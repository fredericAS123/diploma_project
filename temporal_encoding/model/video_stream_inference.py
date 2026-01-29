import torch
from transformers import AutoProcessor
import gc
import time

class VideoStreamingInference:
    def __init__(self, model, processor, device="cuda"):
        self.model = model
        self.processor = processor
        self.device = device
        self.video_cache = None 
        self.current_time_step = 0
        self.current_frame_index = -1
        self.video_duration_sec = None
        self.native_temporal_patches = None
        self.native_fps = 1.0
        self.native_text_len = None
        self.last_manual_time = None
        self.fps = None
        self.tokens_per_second = getattr(getattr(model, "config", None), "vision_config", None)
        if self.tokens_per_second is not None:
            self.tokens_per_second = getattr(self.tokens_per_second, "tokens_per_second", None)
        self._system_prompt_added = False
        print(f"✅ VideoStreamingInference Engine Initialized (Manual Loop).")

    def set_video_fps(self, fps: float):
        if fps is not None and fps > 0:
            self.fps = float(fps)

    def set_video_meta(
        self,
        duration_sec: float,
        temporal_patches: int,
        native_fps: float = 1.0,
        text_len: int | None = None,
    ):
        if duration_sec is not None and duration_sec > 0:
            self.video_duration_sec = float(duration_sec)
        if temporal_patches is not None and temporal_patches > 0:
            self.native_temporal_patches = int(temporal_patches)
        if native_fps is not None and native_fps > 0:
            self.native_fps = float(native_fps)
        if text_len is not None and text_len > 0:
            self.native_text_len = int(text_len)
    def _detach_past(self, past_key_values):
        if past_key_values is None:
            return None
        if hasattr(past_key_values, "get_seq_length"):
            return past_key_values
        return tuple(tuple(p.detach() for p in layer) for layer in past_key_values)

    def _get_past_len(self, past_key_values):
        if past_key_values is None:
            return 0
        if hasattr(past_key_values, "get_seq_length"):
            return past_key_values.get_seq_length()
        return past_key_values[0][0].shape[-2]

    def _build_full_attention_mask(self, attention_mask, past_len):
        if past_len is None or past_len == 0:
            return attention_mask
        past_mask = torch.ones(
            (attention_mask.shape[0], past_len),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        return torch.cat([past_mask, attention_mask], dim=1)

    def reset(self):
        self.video_cache = None
        self.current_time_step = 0
        self.current_frame_index = -1
        self.last_manual_time = None
        self._system_prompt_added = False
        gc.collect()
        torch.cuda.empty_cache()
        print("🔄 Memory Reset.")

    def _compute_manual_time(self, frame_time_sec=None, frame_index=None):
        if self.tokens_per_second is None:
            return None
        if frame_time_sec is not None:
            if self.fps:
                self.current_frame_index = int(round(float(frame_time_sec) * float(self.fps)))
            return int(round(float(frame_time_sec) * self.tokens_per_second))
        if frame_index is not None and self.fps:
            self.current_frame_index = int(frame_index)
            return int(round(float(frame_index) / float(self.fps) * self.tokens_per_second))
        if self.fps:
            self.current_frame_index += 1
            return int(round(float(self.current_frame_index) / float(self.fps) * self.tokens_per_second))
        return None

    def append_frame(
        self,
        image,
        manual_time=None,
        text_content="Frame processed.",
        frame_time_sec=None,
        frame_index=None,
        frame_frac=None,
    ):
        """Phase 1: Stream Encoding"""
        # 构造 Prompt
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": text_content}]}]
        text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        
        if self.video_cache is not None and "<|im_start|>system" in text_prompt:
             text_prompt = "<|im_start|>user" + text_prompt.split("<|im_start|>user")[-1]

        inputs = self.processor(text=[text_prompt], images=[image], padding=True, return_tensors="pt").to(self.device)

        if (
            manual_time is None
            and self.tokens_per_second is not None
            and self.video_duration_sec
            and self.native_temporal_patches
        ):
            input_ids = inputs.input_ids[0].tolist()
            vision_start_id = getattr(self.model.config, "vision_start_token_id", 151652)
            image_token_id = getattr(self.model.config, "image_token_id", 151655)
            text_len = self.native_text_len
            for idx, tid in enumerate(input_ids):
                if tid == vision_start_id and idx + 1 < len(input_ids) and input_ids[idx + 1] == image_token_id:
                    text_len = text_len if text_len is not None else idx + 1
                    break
            if text_len is None:
                for idx, tid in enumerate(input_ids):
                    if tid == image_token_id:
                        text_len = text_len if text_len is not None else idx
                        break

            temporal_patch_size = getattr(self.model.config.vision_config, "temporal_patch_size", 1)
            if frame_frac is not None:
                frac = float(frame_frac)
            elif frame_time_sec is not None and self.video_duration_sec > 0:
                frac = float(frame_time_sec) / float(self.video_duration_sec)
            elif frame_index is not None and self.fps and self.video_duration_sec > 0:
                frac = (float(frame_index) / float(self.fps)) / float(self.video_duration_sec)
            else:
                frac = 0.0

            grid_index = int(round((self.native_temporal_patches - 1) * frac))
            interval = self.tokens_per_second * (temporal_patch_size / float(self.native_fps))
            if text_len is not None:
                manual_time = int(round(text_len + grid_index * interval))

        if manual_time is None:
            manual_time = self._compute_manual_time(frame_time_sec=frame_time_sec, frame_index=frame_index)

        self.last_manual_time = manual_time
        if manual_time is not None:
            self.current_time_step = max(self.current_time_step, int(manual_time))
        else:
            self.current_time_step += 1
        target_t = self.current_time_step


        past_len = self._get_past_len(self.video_cache)
        full_mask = self._build_full_attention_mask(inputs.attention_mask, past_len)
        model_inputs = {k: v for k, v in inputs.items()}
        model_inputs["attention_mask"] = full_mask
        
        with torch.inference_mode():
            # 纯 Forward，存入 Memory
            outputs = self.model(
                **model_inputs,
                past_key_values=self.video_cache,
                manual_time=target_t,
                use_cache=True
            )
            self.video_cache = self._detach_past(outputs.past_key_values)
            del outputs
        
        return f"Frame encoded at T={target_t}"

    def ask(
        self,
        question,
        manual_time=None,
        max_new_tokens=256,
        min_new_tokens=1,
        update_state=False,
        do_sample=False,
        temperature=0.7,
        top_p=0.9,
    ):
        """Phase 2: Interaction (Manual Prefill & Decode)"""
        if manual_time is None:
            # 默认提问发生在视频之后
            manual_time = self.current_time_step + 1
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        t_start = time.time()
            
        # 1. 构造问题 Prompt
        SYSTEM_PROMPT = (
            "You are a concise video analyst. Answer briefly and directly. "
            "Focus on visible facts only. Avoid speculation, avoid repetition. "
            "Strictly limit the response to at most 60 tokens."
        )
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "text", "text": question}]},
        ]
        text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # Keep system prompt; do not strip to last user block.
        
        inputs = self.processor(text=[text_prompt], images=None, padding=True, return_tensors="pt").to(self.device)
        
        input_ids = inputs.input_ids
        
        # 2. 构造 Attention Mask (必须显式包含 Video 历史)
        past_len = self._get_past_len(self.video_cache)
        full_mask = self._build_full_attention_mask(inputs.attention_mask, past_len)
            
        # 3. Step A: Prefill (处理问题文本)
        # 将问题文本一次性输入，计算 KV Cache，且将其位置平移到 manual_time
        current_cache = self.video_cache # Start with video memory
        
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
                manual_time=manual_time, # Shift Text to start at manual_time
                use_cache=True
            )
            
            current_cache = self._detach_past(outputs.past_key_values)
            # 获取最后一个 token 的 logits 用于预测第一个回复 token
            next_token_logits = outputs.logits[:, -1, :]
            next_token = _select_token(next_token_logits)
            
        # 4. Step B: Greedy Decoding Loop (逐个生成回复)
        generated_ids = []
        max_new_tokens = max(1, int(max_new_tokens))
        min_new_tokens = max(1, int(min_new_tokens))
        min_new_tokens = min(min_new_tokens, max_new_tokens)
        eos_token_id = self.processor.tokenizer.eos_token_id
        
        # 维护当前的绝对时间 ID
        # 问题文本长度
        prompt_len = input_ids.shape[1]
        # 回复的起始时间 = manual_time + len(Question)
        current_token_time = manual_time + prompt_len
        
        curr_input = next_token
        last_next_token_logits = next_token_logits
        # Mask 也要随之增长
        curr_mask = torch.cat([full_mask, torch.ones((1, 1), device=self.device)], dim=1)

        t_first_token = None

        with torch.inference_mode():
            # First decode step (for TTFT)
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
                    manual_time=current_token_time,
                    use_cache=True,
                )

                current_cache = self._detach_past(outputs.past_key_values)
                next_token_logits = outputs.logits[:, -1, :]
                curr_input = _select_token(next_token_logits)
                last_next_token_logits = next_token_logits

                current_token_time += 1
                curr_mask = torch.cat([curr_mask, torch.ones((1, 1), device=self.device)], dim=1)

                if self.device.startswith("cuda"):
                    torch.cuda.synchronize()
                t_first_token = time.time()

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
                    manual_time=current_token_time,
                    use_cache=True
                )

                current_cache = self._detach_past(outputs.past_key_values)
                next_token_logits = outputs.logits[:, -1, :]
                curr_input = _select_token(next_token_logits)
                last_next_token_logits = next_token_logits

                current_token_time += 1
                curr_mask = torch.cat([curr_mask, torch.ones((1, 1), device=self.device)], dim=1)
                
        output_text = self.processor.decode(generated_ids, skip_special_tokens=True)
        # 可选：更新全局时间轴（默认不更新，避免问答污染流式时间轴）
        if update_state:
            self.current_time_step = max(self.current_time_step, current_token_time)

        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        t_end = time.time()
        if t_first_token is None:
            t_first_token = t_end
        metrics = {
            "ttft": t_first_token - t_start,
            "total_latency": t_end - t_start,
        }
        return output_text, metrics

    def ask_choice(self, question, choices, manual_time=None):
        """Multiple-choice querying with log-prob scoring (more stable than free decoding)."""
        if manual_time is None:
            manual_time = self.current_time_step + 1

        messages = [{"role": "user", "content": [{"type": "text", "text": question}]}]
        text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # Keep system prompt; do not strip to last user block.

        inputs = self.processor(text=[text_prompt], images=None, padding=True, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids

        past_len = self._get_past_len(self.video_cache)
        full_mask = self._build_full_attention_mask(inputs.attention_mask, past_len)

        current_cache = self.video_cache

        with torch.inference_mode():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=full_mask,
                past_key_values=current_cache,
                manual_time=manual_time,
                use_cache=True,
            )

            current_cache = self._detach_past(outputs.past_key_values)
            next_token_logits = outputs.logits[:, -1, :]

        # Log-prob scoring for each choice
        tokenizer = self.processor.tokenizer
        log_probs = torch.log_softmax(next_token_logits, dim=-1)

        prompt_len = input_ids.shape[1]
        base_time = manual_time + prompt_len

        best_choice = None
        best_score = None

        for choice in choices:
            token_ids = tokenizer(choice, add_special_tokens=False).input_ids
            if len(token_ids) == 0:
                continue

            # Score first token from prefill logits
            score = log_probs[0, token_ids[0]].item()

            # If multi-token, roll forward to score remaining tokens
            if len(token_ids) > 1:
                temp_cache = current_cache
                curr_mask = torch.cat([full_mask, torch.ones((1, 1), device=self.device)], dim=1)
                curr_time = base_time
                curr_input = torch.tensor([[token_ids[0]]], device=self.device)

                with torch.inference_mode():
                    for tid in token_ids[1:]:
                        outputs = self.model(
                            input_ids=curr_input,
                            attention_mask=curr_mask,
                            past_key_values=temp_cache,
                            manual_time=curr_time,
                            use_cache=True,
                        )
                        temp_cache = self._detach_past(outputs.past_key_values)
                        logits = outputs.logits[:, -1, :]
                        lp = torch.log_softmax(logits, dim=-1)
                        score += lp[0, tid].item()

                        curr_input = torch.tensor([[tid]], device=self.device)
                        curr_time += 1
                        curr_mask = torch.cat([curr_mask, torch.ones((1, 1), device=self.device)], dim=1)

            if best_score is None or score > best_score:
                best_score = score
                best_choice = choice

        return best_choice if best_choice is not None else ""