import torch
from transformers import AutoProcessor
import gc

class VideoStreamingInference:
    def __init__(self, model, processor, device="cuda"):
        self.model = model
        self.processor = processor
        self.device = device
        self.video_cache = None 
        self.current_time_step = 0
        print(f"✅ VideoStreamingInference Engine Initialized (Manual Loop).")

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
        gc.collect()
        torch.cuda.empty_cache()
        print("🔄 Memory Reset.")

    def append_frame(self, image, manual_time=None, text_content="Frame processed."):
        """Phase 1: Stream Encoding"""
        if manual_time is not None:
            self.current_time_step = manual_time
        else:
            self.current_time_step += 1
        target_t = self.current_time_step
        
        # 构造 Prompt
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": text_content}]}]
        text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        
        if self.video_cache is not None and "<|im_start|>system" in text_prompt:
             text_prompt = "<|im_start|>user" + text_prompt.split("<|im_start|>user")[-1]

        inputs = self.processor(text=[text_prompt], images=[image], padding=True, return_tensors="pt").to(self.device)

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

    def ask(self, question, manual_time=None):
        """Phase 2: Interaction (Manual Prefill & Decode)"""
        if manual_time is None:
            # 默认提问发生在视频之后
            manual_time = self.current_time_step + 1
            
        # 1. 构造问题 Prompt
        messages = [{"role": "user", "content": [{"type": "text", "text": question}]}]
        text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if "<|im_start|>user" in text_prompt:
            text_prompt = "<|im_start|>user" + text_prompt.split("<|im_start|>user")[-1]
        
        inputs = self.processor(text=[text_prompt], images=None, padding=True, return_tensors="pt").to(self.device)
        
        input_ids = inputs.input_ids
        
        # 2. 构造 Attention Mask (必须显式包含 Video 历史)
        past_len = self._get_past_len(self.video_cache)
        full_mask = self._build_full_attention_mask(inputs.attention_mask, past_len)
            
        # 3. Step A: Prefill (处理问题文本)
        # 将问题文本一次性输入，计算 KV Cache，且将其位置平移到 manual_time
        current_cache = self.video_cache # Start with video memory
        
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
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            
        # 4. Step B: Greedy Decoding Loop (逐个生成回复)
        generated_ids = []
        max_new_tokens = 50
        eos_token_id = self.processor.tokenizer.eos_token_id
        
        # 维护当前的绝对时间 ID
        # 问题文本长度
        prompt_len = input_ids.shape[1]
        # 回复的起始时间 = manual_time + len(Question)
        current_token_time = manual_time + prompt_len
        
        curr_input = next_token
        # Mask 也要随之增长
        curr_mask = torch.cat([full_mask, torch.ones((1, 1), device=self.device)], dim=1)
        
        with torch.inference_mode():
            for _ in range(max_new_tokens):
                # 检查 EOS
                if curr_input.item() == eos_token_id:
                    break
                generated_ids.append(curr_input.item())
                
                # Forward 单个 Token
                outputs = self.model(
                    input_ids=curr_input,
                    attention_mask=curr_mask,
                    past_key_values=current_cache,
                    manual_time=current_token_time, # 关键：每一步时间递增！
                    use_cache=True
                )
                
                current_cache = self._detach_past(outputs.past_key_values)
                next_token_logits = outputs.logits[:, -1, :]
                curr_input = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                
                # 更新状态
                current_token_time += 1
                curr_mask = torch.cat([curr_mask, torch.ones((1, 1), device=self.device)], dim=1)
                
        output_text = self.processor.decode(generated_ids, skip_special_tokens=True)
        # 更新全局时间轴
        self.current_time_step = max(self.current_time_step, current_token_time)
        return output_text

    def ask_choice(self, question, choices, manual_time=None):
        """Multiple-choice querying with log-prob scoring (more stable than free decoding)."""
        if manual_time is None:
            manual_time = self.current_time_step + 1

        messages = [{"role": "user", "content": [{"type": "text", "text": question}]}]
        text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if "<|im_start|>user" in text_prompt:
            text_prompt = "<|im_start|>user" + text_prompt.split("<|im_start|>user")[-1]

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