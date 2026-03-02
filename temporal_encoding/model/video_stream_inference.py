"""
VideoStreamingInference — Streaming VLM Inference (Chunk-Local / Append 模式)

关键设计：
  1) 首帧包含 system+user+vision，后续帧仅追加 vision tokens
  2) Position 由 StreamQwenModel 内部自动跟踪（append 模式 3 分支）
  3) ask()/ask_choice() 使用 KVCacheManager snapshot/restore，
     同时保存/恢复模型的 stream_state，防止污染视频缓存
  4) (v2) 支持 KV Cache 淘汰策略，控制显存增长，实现无限长度视频流
     参考 StreamingVLM (MIT-HAN-Lab) + StreamingLLM + LOOK-M

Chunk-Local 假设：
  - ViT 只在 chunk 内建模，跨 chunk 时序由 LLM+KV+RoPE 负责
  - temporal_patch_size=2，每个 temporal chunk 融合 2 帧

推荐 chunk 大小：
  - 2 帧 (as_video=True, fps=1-2): 最低延迟，T=1 temporal grid
  - 4 帧 (as_video=True, fps=2-4): 延迟/质量均衡推荐，T=2
  - 6-8 帧 (as_video=True, fps=2-4): 更高吞吐，适合准实时
  - 单帧 image 模式: 最简单但效率较低（1帧被复制为2帧凑对 temporal_patch_size）

注意：
  - 不再使用 manual_time / VideoMetaCalculator
  - 若要输入多帧 chunk，请使用 as_video=True，并传入帧列表
  - 多帧 chunk 的帧数建议为 temporal_patch_size(2) 的倍数，避免被帧填充
"""

import gc
import time
from typing import List, Optional

import torch

from .cache_manager import KVCacheManager
from .kv_cache_eviction import EvictionConfig


class VideoStreamingInference:
    def __init__(
        self,
        model,
        processor,
        device: str = "cuda",
        eviction_config: Optional[EvictionConfig] = None,
    ):
        self.model = model
        self.processor = processor
        self.device = device

        self.cache_manager = KVCacheManager(eviction_config=eviction_config)
        self.frame_count = 0      # chunk 计数
        self.total_frames = 0     # 实际帧数累计
        self._system_prompt_added = False
        self._chunk_counter = 0   # 淘汰间隔计数器

        # 统一的系统提示
        self.system_prompt = (
            "You are a concise video analyst. Answer briefly and directly. "
            "Focus on visible facts only. Avoid speculation, avoid repetition. "
            # "Strictly limit the response to at most 60 tokens."
        )

        eviction_str = "OFF"
        if eviction_config is not None:
            sink_str = "auto" if eviction_config.sink_size == 0 else str(eviction_config.sink_size)
            win_str = "auto" if eviction_config.window_size == 0 else str(eviction_config.window_size)
            eviction_str = (
                f"ON (max={eviction_config.max_cache_tokens}, "
                f"sink={sink_str}, window={win_str})"
            )
        print(f"✅ VideoStreamingInference Engine Initialized (Chunk-Local / Append Mode).")
        print(f"   KV Cache Eviction: {eviction_str}")

    # ── Prompt 处理 ────────────────────────────────────────────

    @staticmethod
    def _extract_vision_segment(text_prompt: str) -> str:
        """从 chat template 中裁剪出 <|vision_start|>...<|vision_end|> 片段。"""
        start_tok = "<|vision_start|>"
        end_tok = "<|vision_end|>"
        if start_tok in text_prompt and end_tok in text_prompt:
            head = text_prompt.split(start_tok, 1)[1]
            body = head.split(end_tok, 1)[0]
            return f"{start_tok}{body}{end_tok}"
        return text_prompt

    @staticmethod
    def _extract_user_vision_turn(text_prompt: str) -> str:
        """
        从 chat template 中提取用户 turn 中的视觉内容，保留对话结构标记。

        返回: <|im_start|>user\n<|vision_start|>...<|vision_end|><|im_end|>\n
        这样后续 chunk 的 token 分布与首帧相似，减少 OOD 降质。
        """
        im_start = "<|im_start|>"
        im_end = "<|im_end|>"
        vision_start = "<|vision_start|>"
        vision_end = "<|vision_end|>"

        # 找到包含 vision 的 user turn
        if vision_start in text_prompt and vision_end in text_prompt:
            # 提取 vision segment
            head = text_prompt.split(vision_start, 1)[1]
            body = head.split(vision_end, 1)[0]
            vision_seg = f"{vision_start}{body}{vision_end}"
            # 包裹在 user turn 结构中
            return f"{im_start}user\n{vision_seg}{im_end}\n"

        return text_prompt

    def _build_frame_prompt(self, as_video: bool, vision_payload, text_content: str) -> str:
        if not self._system_prompt_added:
            messages = [
                {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
            ]
            if as_video:
                messages.append(
                    {"role": "user", "content": [
                        {"type": "video", "video": vision_payload},
                        {"type": "text", "text": text_content},
                    ]}
                )
            else:
                messages.append(
                    {"role": "user", "content": [
                        {"type": "image", "image": vision_payload},
                        {"type": "text", "text": text_content},
                    ]}
                )

            text_prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            self._system_prompt_added = True
            return text_prompt

        # 后续帧：保留 <|im_start|>user\n...<|im_end|> 对话结构包裹
        # 减少裸 vision token 带来的 OOD 效应
        if as_video:
            messages = [
                {"role": "user", "content": [{"type": "video", "video": vision_payload}]}
            ]
        else:
            messages = [
                {"role": "user", "content": [{"type": "image", "image": vision_payload}]}
            ]
        text_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        return self._extract_user_vision_turn(text_prompt)

    # ── Reset ──────────────────────────────────────────────────

    def reset(self):
        self.cache_manager.clear()
        self.frame_count = 0
        self.total_frames = 0
        self._system_prompt_added = False
        self._chunk_counter = 0
        if hasattr(self.model, "reset_stream_state"):
            self.model.reset_stream_state()
        gc.collect()
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
        print("🔄 Memory Reset.")

    # ── 追加帧 / Chunk ─────────────────────────────────────────

    def append_frame(
        self,
        image,
        text_content: str = "Frame processed.",
        as_video: bool = False,
        fps: Optional[float] = None,
    ) -> str:
        """
        Phase 1: Stream Encoding (Append)

        Args:
            image: 单帧 PIL Image；或当 as_video=True 时为帧列表 (List[PIL.Image])
            text_content: 首帧附带的文本描述（后续帧被忽略）
            as_video: True → 使用视频 token（推荐多帧 chunk）
            fps: 采样帧率（仅 as_video=True 时有效）
        """
        if as_video and not isinstance(image, (list, tuple)):
            # 允许单帧视频作为特例
            image = [image]
        if (not as_video) and isinstance(image, (list, tuple)):
            raise ValueError("When passing multiple frames, set as_video=True.")

        # 1) 构造 prompt
        text_prompt = self._build_frame_prompt(as_video, image, text_content)

        # 2) Processor 输入
        if as_video:
            videos_kwargs = {"fps": fps} if fps is not None else None
            inputs = self.processor(
                text=[text_prompt],
                videos=[image],
                padding=True,
                return_tensors="pt",
                **({"videos_kwargs": videos_kwargs} if videos_kwargs is not None else {}),
            ).to(self.device)
        else:
            inputs = self.processor(
                text=[text_prompt],
                images=[image],
                padding=True,
                return_tensors="pt",
            ).to(self.device)

        # 3) 构造 Attention Mask (包含 past KV cache 长度)
        full_mask = self.cache_manager.build_full_attention_mask(
            inputs.attention_mask,
            cache_override=self.cache_manager.cache,
        )
        model_inputs = {k: v for k, v in inputs.items()}
        model_inputs["attention_mask"] = full_mask

        # 4) Forward（position 由模型内部自动计算）
        with torch.inference_mode():
            outputs = self.model(
                **model_inputs,
                past_key_values=self.cache_manager.cache,
                use_cache=True,
            )
            self.cache_manager.cache = self.cache_manager.detach(outputs.past_key_values)
            del outputs

        # 5) Token Tracking (Level 2/3 需要)
        if "input_ids" in inputs:
            self.cache_manager.track_tokens(inputs["input_ids"], is_new_chunk=True)

        # 6) 首 chunk 自动检测 sink_size
        cache_len_after = self.cache_manager.get_seq_length()
        if self.cache_manager.eviction_enabled and self.frame_count == 0:
            self.cache_manager.set_first_chunk_info(cache_len_after)
            evictor = self.cache_manager.evictor
            if evictor is not None:
                print(
                    f"   \U0001f4cd Auto-sink detected: first chunk = {cache_len_after} tokens"
                    f" (sink={evictor.effective_sink_size}, window={evictor.effective_window_size})"
                )
            self._prev_cache_len = cache_len_after
        elif self.cache_manager.eviction_enabled:
            # 更新 chunk 统计
            chunk_tokens = cache_len_after - getattr(self, '_prev_cache_len', 0)
            if chunk_tokens > 0 and self.cache_manager.evictor is not None:
                self.cache_manager.evictor.update_chunk_stats(chunk_tokens)
            self._prev_cache_len = cache_len_after

        # 7) KV Cache Eviction (如果启用)
        eviction_info = {"evicted": False}
        if self.cache_manager.eviction_enabled:
            self._chunk_counter += 1
            evictor = self.cache_manager.evictor
            if evictor is not None:
                interval = evictor.config.eviction_interval
                if self._chunk_counter % interval == 0:
                    eviction_info = self.cache_manager.evict_if_needed()
                    if eviction_info.get("evicted"):
                        print(
                            f"  ✂️ Eviction: {eviction_info['tokens_before']} → "
                            f"{eviction_info['tokens_after']} tokens "
                            f"(-{eviction_info['tokens_removed']})"
                        )

        self.frame_count += 1
        n_frames = len(image) if as_video and isinstance(image, (list, tuple)) else 1
        self.total_frames += n_frames
        cache_len = self.cache_manager.get_seq_length()
        return f"Chunk {self.frame_count - 1} encoded ({n_frames} frame(s), cache_len={cache_len})"

    # ── Ask ────────────────────────────────────────────────────

    def ask(
        self,
        question: str,
        max_new_tokens: int = 256,
        min_new_tokens: int = 1,
        update_state: bool = False,
        do_sample: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        """
        Phase 2: Interaction (Chunk Prefill + Decode)

        - 问题 Prefill → Branch 2 (chunk prefill + offset)
        - 逐 token Decode → Branch 3 (last_cache_position + 1)
        """
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        t_start = time.time()

        # Snapshot: 保护视频 KV Cache + 模型 stream_state
        self.cache_manager.snapshot(self.model)

        # 1) 构造问题 Prompt（不重复 system prompt）
        messages = [
            {"role": "user", "content": [{"type": "text", "text": question}]}
        ]
        text_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text_prompt], images=None, padding=True, return_tensors="pt"
        ).to(self.device)

        input_ids = inputs.input_ids

        # 2) 构造 Attention Mask (包含 Video 历史)
        full_mask = self.cache_manager.build_full_attention_mask(
            inputs.attention_mask,
            cache_override=self.cache_manager.cache,
        )

        current_cache = self.cache_manager.cache

        def _select_token(logits):
            if not do_sample:
                return torch.argmax(logits, dim=-1).unsqueeze(-1)
            temp = max(1e-5, float(temperature))
            scaled = logits / temp
            probs = torch.softmax(scaled, dim=-1)
            if top_p is not None and 0 < top_p < 1:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cum = torch.cumsum(sorted_probs, dim=-1)
                mask = cum > top_p
                mask[..., 0] = False
                sorted_probs = sorted_probs.masked_fill(mask, 0.0)
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                next_idx = torch.multinomial(sorted_probs, num_samples=1)
                return sorted_idx.gather(-1, next_idx)
            return torch.multinomial(probs, num_samples=1)

        # 3) Prefill
        with torch.inference_mode():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=full_mask,
                past_key_values=current_cache,
                use_cache=True,
            )
            current_cache = self.cache_manager.detach(outputs.past_key_values)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = _select_token(next_token_logits)

        # TTFT: 首 token 在 prefill 完成后即可确定
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        t_first_token = time.time()

        # 4) Decode loop
        generated_ids: List[int] = []
        max_new_tokens = max(1, int(max_new_tokens))
        min_new_tokens = max(1, int(min_new_tokens))
        min_new_tokens = min(min_new_tokens, max_new_tokens)
        eos_token_id = self.processor.tokenizer.eos_token_id

        curr_input = next_token
        last_next_token_logits = next_token_logits
        curr_mask = torch.cat([full_mask, torch.ones((1, 1), device=self.device)], dim=1)

        with torch.inference_mode():
            if curr_input.item() == eos_token_id and min_new_tokens > 0:
                tmp_logits = last_next_token_logits.clone()
                tmp_logits[0, eos_token_id] = -1e9
                curr_input = _select_token(tmp_logits)

            if curr_input.item() != eos_token_id:
                generated_ids.append(curr_input.item())

                outputs = self.model(
                    input_ids=curr_input,
                    attention_mask=curr_mask,
                    past_key_values=current_cache,
                    use_cache=True,
                )

                current_cache = self.cache_manager.detach(outputs.past_key_values)
                next_token_logits = outputs.logits[:, -1, :]
                curr_input = _select_token(next_token_logits)
                last_next_token_logits = next_token_logits

                curr_mask = torch.cat([curr_mask, torch.ones((1, 1), device=self.device)], dim=1)

            for _ in range(max_new_tokens - 1):
                if curr_input.item() == eos_token_id:
                    if len(generated_ids) >= min_new_tokens:
                        break
                    tmp_logits = last_next_token_logits.clone()
                    tmp_logits[0, eos_token_id] = -1e9
                    curr_input = _select_token(tmp_logits)
                generated_ids.append(curr_input.item())

                outputs = self.model(
                    input_ids=curr_input,
                    attention_mask=curr_mask,
                    past_key_values=current_cache,
                    use_cache=True,
                )

                current_cache = self.cache_manager.detach(outputs.past_key_values)
                next_token_logits = outputs.logits[:, -1, :]
                curr_input = _select_token(next_token_logits)
                last_next_token_logits = next_token_logits

                curr_mask = torch.cat([curr_mask, torch.ones((1, 1), device=self.device)], dim=1)

        output_text = self.processor.decode(generated_ids, skip_special_tokens=True)

        if update_state:
            self.cache_manager.cache = current_cache
            self.cache_manager.discard_snapshot()
        else:
            # 恢复 KV Cache + 模型 stream_state
            self.cache_manager.restore(self.model)

        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        t_end = time.time()

        metrics = {
            "ttft": t_first_token - t_start,
            "total_latency": t_end - t_start,
        }
        return output_text, metrics

    def ask_stream(
        self,
        question: str,
        max_new_tokens: int = 256,
        min_new_tokens: int = 1,
        update_state: bool = False,
        do_sample: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        """
        流式问答：按 token 增量产出文本，最后返回完整 metrics。

        Yields:
            {
              "type": "token", "delta": str, "text": str, "ttft": float|None
            }
            {
              "type": "final", "text": str, "metrics": {...}
            }
        """
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        t_start = time.time()

        # Snapshot: 保护视频 KV Cache + 模型 stream_state
        self.cache_manager.snapshot(self.model)

        messages = [{"role": "user", "content": [{"type": "text", "text": question}]}]
        text_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text_prompt], images=None, padding=True, return_tensors="pt"
        ).to(self.device)

        input_ids = inputs.input_ids
        full_mask = self.cache_manager.build_full_attention_mask(
            inputs.attention_mask,
            cache_override=self.cache_manager.cache,
        )
        current_cache = self.cache_manager.cache

        def _select_token(logits):
            if not do_sample:
                return torch.argmax(logits, dim=-1).unsqueeze(-1)
            temp = max(1e-5, float(temperature))
            scaled = logits / temp
            probs = torch.softmax(scaled, dim=-1)
            if top_p is not None and 0 < top_p < 1:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cum = torch.cumsum(sorted_probs, dim=-1)
                mask = cum > top_p
                mask[..., 0] = False
                sorted_probs = sorted_probs.masked_fill(mask, 0.0)
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                next_idx = torch.multinomial(sorted_probs, num_samples=1)
                return sorted_idx.gather(-1, next_idx)
            return torch.multinomial(probs, num_samples=1)

        with torch.inference_mode():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=full_mask,
                past_key_values=current_cache,
                use_cache=True,
            )
            current_cache = self.cache_manager.detach(outputs.past_key_values)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = _select_token(next_token_logits)

        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        t_first_token = time.time()

        generated_ids: List[int] = []
        max_new_tokens = max(1, int(max_new_tokens))
        min_new_tokens = max(1, int(min_new_tokens))
        min_new_tokens = min(min_new_tokens, max_new_tokens)
        eos_token_id = self.processor.tokenizer.eos_token_id

        curr_input = next_token
        last_next_token_logits = next_token_logits
        curr_mask = torch.cat([full_mask, torch.ones((1, 1), device=self.device)], dim=1)

        streamed_text = ""
        ttft_sent = False

        with torch.inference_mode():
            if curr_input.item() == eos_token_id and min_new_tokens > 0:
                tmp_logits = last_next_token_logits.clone()
                tmp_logits[0, eos_token_id] = -1e9
                curr_input = _select_token(tmp_logits)

            if curr_input.item() != eos_token_id:
                generated_ids.append(curr_input.item())
                new_text = self.processor.decode(generated_ids, skip_special_tokens=True)
                delta = new_text[len(streamed_text):]
                streamed_text = new_text
                if delta:
                    yield {
                        "type": "token",
                        "delta": delta,
                        "text": streamed_text,
                        "ttft": (t_first_token - t_start) if not ttft_sent else None,
                    }
                    ttft_sent = True

                outputs = self.model(
                    input_ids=curr_input,
                    attention_mask=curr_mask,
                    past_key_values=current_cache,
                    use_cache=True,
                )

                current_cache = self.cache_manager.detach(outputs.past_key_values)
                next_token_logits = outputs.logits[:, -1, :]
                curr_input = _select_token(next_token_logits)
                last_next_token_logits = next_token_logits
                curr_mask = torch.cat([curr_mask, torch.ones((1, 1), device=self.device)], dim=1)

            for _ in range(max_new_tokens - 1):
                if curr_input.item() == eos_token_id:
                    if len(generated_ids) >= min_new_tokens:
                        break
                    tmp_logits = last_next_token_logits.clone()
                    tmp_logits[0, eos_token_id] = -1e9
                    curr_input = _select_token(tmp_logits)

                generated_ids.append(curr_input.item())
                new_text = self.processor.decode(generated_ids, skip_special_tokens=True)
                delta = new_text[len(streamed_text):]
                streamed_text = new_text
                if delta:
                    yield {
                        "type": "token",
                        "delta": delta,
                        "text": streamed_text,
                        "ttft": None,
                    }

                outputs = self.model(
                    input_ids=curr_input,
                    attention_mask=curr_mask,
                    past_key_values=current_cache,
                    use_cache=True,
                )

                current_cache = self.cache_manager.detach(outputs.past_key_values)
                next_token_logits = outputs.logits[:, -1, :]
                curr_input = _select_token(next_token_logits)
                last_next_token_logits = next_token_logits
                curr_mask = torch.cat([curr_mask, torch.ones((1, 1), device=self.device)], dim=1)

        output_text = self.processor.decode(generated_ids, skip_special_tokens=True)

        if update_state:
            self.cache_manager.cache = current_cache
            self.cache_manager.discard_snapshot()
        else:
            self.cache_manager.restore(self.model)

        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        t_end = time.time()

        metrics = {
            "ttft": t_first_token - t_start,
            "total_latency": t_end - t_start,
        }
        yield {
            "type": "final",
            "text": output_text,
            "metrics": metrics,
        }

    # ── Ask Choice ─────────────────────────────────────────────

    def ask_choice(self, question: str, choices: List[str]):
        """
        Multiple-choice querying with log-prob scoring.

        对问题做一次 prefill，然后对每个选项逐 token 累加 log-prob。
        多 token 选项使用独立 cache 副本 + 独立模型状态。
        """
        # Snapshot: 保护视频 KV Cache + 模型 stream_state
        self.cache_manager.snapshot(self.model)

        messages = [
            {"role": "user", "content": [{"type": "text", "text": question}]}
        ]
        text_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text_prompt], images=None, padding=True, return_tensors="pt"
        ).to(self.device)
        input_ids = inputs.input_ids

        full_mask = self.cache_manager.build_full_attention_mask(
            inputs.attention_mask,
            cache_override=self.cache_manager.cache,
        )

        base_cache = self.cache_manager.cache

        # Prefill 问题部分 → Branch 2
        with torch.inference_mode():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=full_mask,
                past_key_values=base_cache,
                use_cache=True,
            )
            base_cache = self.cache_manager.detach(outputs.past_key_values)
            next_token_logits = outputs.logits[:, -1, :]

        # 保存 prefill 后的模型状态（每个选项从此分叉）
        post_prefill_state = self.model.stream_state

        tokenizer = self.processor.tokenizer
        log_probs = torch.log_softmax(next_token_logits, dim=-1)

        best_choice = None
        best_score = None

        for choice in choices:
            token_ids = tokenizer(choice, add_special_tokens=False).input_ids
            if len(token_ids) == 0:
                continue

            score = log_probs[0, token_ids[0]].item()

            if len(token_ids) > 1:
                # 独立 cache 副本 + 独立模型状态
                temp_cache = self.cache_manager.clone(base_cache)
                self.model.stream_state = post_prefill_state
                curr_mask = torch.cat([full_mask, torch.ones((1, 1), device=self.device)], dim=1)
                curr_input = torch.tensor([[token_ids[0]]], device=self.device)

                with torch.inference_mode():
                    for tid in token_ids[1:]:
                        outputs = self.model(
                            input_ids=curr_input,
                            attention_mask=curr_mask,
                            past_key_values=temp_cache,
                            use_cache=True,
                        )
                        temp_cache = self.cache_manager.detach(outputs.past_key_values)
                        logits = outputs.logits[:, -1, :]
                        lp = torch.log_softmax(logits, dim=-1)
                        score += lp[0, tid].item()

                        curr_input = torch.tensor([[tid]], device=self.device)
                        curr_mask = torch.cat(
                            [curr_mask, torch.ones((1, 1), device=self.device)], dim=1
                        )

            if best_score is None or score > best_score:
                best_score = score
                best_choice = choice

        # 恢复视频 KV Cache + 模型 stream_state
        self.cache_manager.restore(self.model)

        return best_choice if best_choice is not None else ""

    # ── 便捷方法 ──────────────────────────────────────────────

    def append_video_chunk(
        self,
        frames: List,
        fps: float = 2.0,
        text_content: str = "Video chunk processed.",
    ) -> str:
        """
        追加多帧视频 chunk 的便捷方法。

        Args:
            frames: PIL Image 列表，建议帧数为 temporal_patch_size(2) 的倍数
            fps: 帧率（影响 LLM M-RoPE 中的时间位置编码间距）
            text_content: 首帧附带的文本描述

        Returns:
            编码状态字符串

        推荐用法:
            # 2 帧 chunk (T=1, 最低延迟)
            engine.append_video_chunk([frame0, frame1], fps=2.0)

            # 4 帧 chunk (T=2, 延迟/质量均衡)
            engine.append_video_chunk([f0, f1, f2, f3], fps=4.0)
        """
        if not isinstance(frames, (list, tuple)) or len(frames) == 0:
            raise ValueError("frames must be a non-empty list of PIL Images.")
        if len(frames) % 2 != 0:
            print(
                f"⚠️ {len(frames)} frames is not a multiple of temporal_patch_size=2. "
                f"Last frame will be duplicated by the processor."
            )
        return self.append_frame(frames, text_content=text_content, as_video=True, fps=fps)

    @staticmethod
    def _measure_cache_bytes(cache) -> int:
        """
        计算 KV Cache 的总显存占用（字节）。

        兼容 3 种 DynamicCache 内部结构:
          ① transformers ≥ 4.50: cache.layers → list[DynamicLayer]
             每层有 .key_state / .value_state (Tensor)
          ② transformers < 4.50: cache.key_cache / cache.value_cache → list[Tensor]
          ③ Tuple-of-tuples: ((K, V), (K, V), ...)
        """
        if cache is None:
            return 0

        total = 0

        # ─── DynamicCache 路径 ───
        if hasattr(cache, "get_seq_length"):
            try:
                # 策略 1: 新版 .layers 属性（每层是 DynamicLayer 对象）
                if hasattr(cache, "layers") and len(getattr(cache, "layers", [])) > 0:
                    for layer in cache.layers:
                        for attr in ("key_state", "value_state"):
                            t = getattr(layer, attr, None)
                            if t is not None and hasattr(t, "nelement"):
                                total += t.nelement() * t.element_size()
                    if total > 0:
                        return total

                # 策略 2: 旧版 key_cache / value_cache 列表
                for attr in ("key_cache", "value_cache"):
                    lst = getattr(cache, attr, None)
                    if lst is not None:
                        for t in lst:
                            if hasattr(t, "nelement"):
                                total += t.nelement() * t.element_size()
                if total > 0:
                    return total

                # 策略 3: 通用回退 — 通过 __getitem__ 逐层提取
                try:
                    n = len(cache)
                    for i in range(n):
                        layer_kv = cache[i]
                        if isinstance(layer_kv, (tuple, list)):
                            for t in layer_kv:
                                if hasattr(t, "nelement"):
                                    total += t.nelement() * t.element_size()
                except (TypeError, IndexError):
                    pass
            except Exception:
                pass
            return total

        # ─── Tuple-of-tuples 路径 ───
        if isinstance(cache, (tuple, list)):
            for layer in cache:
                if isinstance(layer, (tuple, list)):
                    for t in layer:
                        if hasattr(t, "nelement"):
                            total += t.nelement() * t.element_size()
        return total

    def get_cache_info(self) -> dict:
        """返回当前 KV Cache 状态信息（含淘汰统计）。"""
        cache_len = self.cache_manager.get_seq_length()
        cache = self.cache_manager.cache
        mem_bytes = self._measure_cache_bytes(cache)
        mem_gb = mem_bytes / (1024 ** 3)

        stream_state = None
        if hasattr(self.model, "stream_state"):
            stream_state = {
                "last_cache_position": self.model.stream_state["last_cache_position"],
                "rope_deltas": (
                    self.model.stream_state["rope_deltas"].tolist()
                    if self.model.stream_state["rope_deltas"] is not None
                    else None
                ),
            }

        info = {
            "chunks_encoded": self.frame_count,
            "total_frames": self.total_frames,
            "cache_seq_length": cache_len,
            "cache_memory_gb": round(mem_gb, 4),
            "stream_state": stream_state,
        }

        # 淘汰统计
        eviction_stats = self.cache_manager.get_eviction_stats()
        if eviction_stats:
            info["eviction_stats"] = eviction_stats

        # 淘汰策略说明（用于前端状态栏展示）
        if self.cache_manager.evictor is not None:
            cfg = self.cache_manager.evictor.config
            if cfg.enable_frame_importance:
                strategy = "L3-importance"
            elif cfg.enable_temporal_sampling:
                strategy = "L2-mid-anchors"
            else:
                strategy = "L1-sink-window"
            info["eviction_strategy"] = strategy
            info["eviction_config"] = {
                "max_cache_tokens": int(cfg.max_cache_tokens),
                "enable_temporal_sampling": bool(cfg.enable_temporal_sampling),
                "mid_retention_ratio": float(cfg.mid_retention_ratio),
                "enable_sink_boundary_guard": bool(cfg.enable_sink_boundary_guard),
                "sink_boundary_guard_tokens": int(cfg.sink_boundary_guard_tokens),
            }

        return info
