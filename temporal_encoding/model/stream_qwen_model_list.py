# 文件路径: temporal_encoding/model/stream_qwen_model.py
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLConfig
from typing import Optional, List, Union, Tuple
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLCausalLMOutputWithPast

class StreamQwenModel(Qwen2_5_VLForConditionalGeneration):
    def __init__(self, config: Qwen2_5_VLConfig):
        super().__init__(config)
        self.vision_start_token_id = getattr(config, "vision_start_token_id", 151652)
        self.vision_end_token_id = getattr(config, "vision_end_token_id", 151653)
        print(f"[StreamQwenModel] Initialized. Vision Start ID: {self.vision_start_token_id}")

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        manual_time_list: Optional[List[int]] = None,
        **kwargs,
    ) -> Union[Tuple, Qwen2VLCausalLMOutputWithPast]:

        # 1. 生成默认 Position IDs
        if position_ids is None and input_ids is not None:
            if past_key_values is None or past_key_values.get_seq_length() == 0:
                pos_ids, rope_deltas_calc = self.model.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask,
                )
                position_ids = pos_ids
                if rope_deltas is None:
                    rope_deltas = rope_deltas_calc

        # 2. 执行时间注入 (v14: 全维度平移 + 无Delta干扰)
        if manual_time_list is not None and input_ids is not None:
            has_vision = (input_ids == self.vision_start_token_id).any()
            
            if has_vision:
                modified_pos_ids = position_ids.clone()
                injected_any = False
                seq_len = input_ids.shape[1]
                
                print(f"\n{'='*20} AM-ROPE INJECTION LOG (v14) {'='*20}")
                
                for b in range(input_ids.shape[0]):
                    start_indices = (input_ids[b] == self.vision_start_token_id).nonzero(as_tuple=True)[0]
                    end_indices = (input_ids[b] == self.vision_end_token_id).nonzero(as_tuple=True)[0]
                    
                    num_blocks = len(start_indices)
                    if num_blocks > 0:
                        print(f"[Batch {b}] Found {num_blocks} blocks. Target times: {manual_time_list}")
                        limit = min(num_blocks, len(manual_time_list))
                        
                        last_e_idx = -1
                        last_injected_t = -1
                        max_h_w = 0 
                        
                        # A. 注入图片时间
                        for i in range(limit):
                            s_idx = start_indices[i]
                            e_idx = end_indices[i] if i < len(end_indices) else seq_len - 1
                            target_time = manual_time_list[i]
                            
                            # 获取当前块的最大空间坐标，作为文字平移的基准
                            current_block_pos = modified_pos_ids[:, b, s_idx:e_idx+1]
                            max_h_w = max(max_h_w, current_block_pos[1].max().item(), current_block_pos[2].max().item())

                            # 强行修改 T 维度
                            modified_pos_ids[0, b, s_idx : e_idx + 1] = target_time
                            
                            last_e_idx = max(last_e_idx, e_idx)
                            last_injected_t = target_time
                            injected_any = True

                        # B. 平移后续文字 (T, H, W 全移)
                        if last_e_idx < seq_len - 1 and last_injected_t >= 0:
                            text_start_idx = last_e_idx + 1
                            
                            # T轴目标: 紧接最后一张图
                            target_t = last_injected_t + 1
                            
                            # H/W轴目标: 放在图片空间之外的一个安全小数值
                            # 注意：不能比图片内部的 Grid 小，否则会重叠。
                            # 假设 max_h_w 是图片最大的 grid index (比如 28)，那我们从 30 开始就很安全
                            # 同时为了保持 monotonic 性质，加上 target_t 是个好习惯
                            target_hw = max_h_w + 2 # 紧贴着图片的空间结束点
                            
                            curr_t = modified_pos_ids[0, b, text_start_idx].item()
                            curr_h = modified_pos_ids[1, b, text_start_idx].item()
                            curr_w = modified_pos_ids[2, b, text_start_idx].item()

                            offset_t = target_t - curr_t
                            offset_h = target_hw - curr_h
                            offset_w = target_hw - curr_w
                            
                            # 应用偏移
                            modified_pos_ids[0, b, text_start_idx:] += offset_t
                            modified_pos_ids[1, b, text_start_idx:] += offset_h
                            modified_pos_ids[2, b, text_start_idx:] += offset_w
                            
                            print(f"  >> 📝 Text Shift: Snapped to T={target_t}, HW={target_hw}")
                            print(f"     Offsets: T={offset_t}, H={offset_h}, W={offset_w}")

                if injected_any:
                    print(f"{'='*60}\n")
                    position_ids = modified_pos_ids
                    # 【再次确认】绝对不更新 rope_deltas

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            rope_deltas=rope_deltas,
            cache_position=cache_position,
            **kwargs,
        )

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        manual_time_list = kwargs.get("manual_time_list", None)
        model_inputs = super().prepare_inputs_for_generation(input_ids, **kwargs)
        if manual_time_list is not None:
            model_inputs["manual_time_list"] = manual_time_list
        return model_inputs