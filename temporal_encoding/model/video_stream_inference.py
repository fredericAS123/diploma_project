"""
VideoStreamingInference — Streaming VLM Inference (Chunk-Local / Append 模式)

关键设计：
  1) 首帧包含 system+user+vision，后续帧仅追加 vision tokens
  2) Position 由 StreamQwenModel 内部自动跟踪（append 模式 3 分支）
  3) ask()/ask_choice() 使用 KVCacheManager snapshot/restore，
     同时保存/恢复模型的 stream_state，防止污染视频缓存
  4) 累积所有历史 KV Cache，不做 sliding window 或 eviction

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


class VideoStreamingInference:
    def __init__(self, model, processor, device: str = "cuda"):
        self.model = model
        self.processor = processor
        self.device = device

        self.cache_manager = KVCacheManager()
        self.frame_count = 0      # chunk 计数
        self.total_frames = 0     # 实际帧数累计
        self._system_prompt_added = False

        # 统一的系统提示
        self.system_prompt = (
            "You are a concise video analyst. Answer briefly and directly. "
            "Focus on visible facts only. Avoid speculation, avoid repetition. "
            "Strictly limit the response to at most 60 tokens."
        )

        print("✅ VideoStreamingInference Engine Initialized (Chunk-Local / Append Mode).")

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

        # 后续帧：仅追加视觉 token（避免重复系统/文本）
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
        return self._extract_vision_segment(text_prompt)

    # ── Reset ──────────────────────────────────────────────────

    def reset(self):
        self.cache_manager.clear()
        self.frame_count = 0
        self.total_frames = 0
        self._system_prompt_added = False
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

    def get_cache_info(self) -> dict:
        """返回当前 KV Cache 状态信息。"""
        cache_len = self.cache_manager.get_seq_length()
        mem_mb = 0.0
        cache = self.cache_manager.cache
        if cache is not None:
            if hasattr(cache, "get_seq_length"):
                # DynamicCache: 估算 = n_layers × 2(K+V) × seq × heads × dim × dtype_bytes
                try:
                    for kv in cache.key_cache:
                        mem_mb += kv.nelement() * kv.element_size()
                    for kv in cache.value_cache:
                        mem_mb += kv.nelement() * kv.element_size()
                except Exception:
                    pass
            elif isinstance(cache, (tuple, list)):
                for layer in cache:
                    for t in layer:
                        mem_mb += t.nelement() * t.element_size()
            mem_mb /= (1024 * 1024)

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

        return {
            "chunks_encoded": self.frame_count,
            "total_frames": self.total_frames,
            "cache_seq_length": cache_len,
            "cache_memory_mb": round(mem_mb, 2),
            "stream_state": stream_state,
        }
