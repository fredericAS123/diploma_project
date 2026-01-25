import torch
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLConfig
from transformers.modeling_outputs import ModelOutput
from typing import Optional, List, Union, Tuple
from dataclasses import dataclass

@dataclass
class StreamQwenModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None

class StreamQwenModel(Qwen2_5_VLForConditionalGeneration):
    def __init__(self, config: Qwen2_5_VLConfig):
        super().__init__(config)
        self.vision_start_token_id = getattr(config, "vision_start_token_id", 151652)
        self.vision_end_token_id = getattr(config, "vision_end_token_id", 151653)
        print(f"[StreamQwenModel] Initialized. Vision IDs: {self.vision_start_token_id}-{self.vision_end_token_id}")

    def _apply_time_shift(self, position_ids, input_ids, target_time):
        """
        Qwen2.5-VL 使用 mRoPE (3D). 我们采用一致的“全局时间平移”策略：
        - 文本 token: 3 维坐标整体平移，保持相对序列一致性。
        - 视觉 token: 仅平移时间维 (dim=0)，保留空间维 (dim=1,2) 以避免几何错位。
        """
        modified_pos_ids = position_ids.clone()
        batch_size = input_ids.shape[0]
        target_time = int(target_time)

        for b in range(batch_size):
            has_vision_b = (input_ids[b] == self.vision_start_token_id).any()

            if has_vision_b:
                start_indices = (input_ids[b] == self.vision_start_token_id).nonzero(as_tuple=True)[0]
                end_indices = (input_ids[b] == self.vision_end_token_id).nonzero(as_tuple=True)[0]

                if len(start_indices) == 0 or len(end_indices) == 0:
                    continue

                s_idx = start_indices[0]
                e_idx = end_indices[0]

                anchor_time = modified_pos_ids[0, b, s_idx].item()
                delta = target_time - anchor_time
                if delta == 0:
                    continue

                # 1) 先对全部 token 的 3 维坐标做整体平移（保证文本序列相对关系不被破坏）
                modified_pos_ids[:, b, :] += delta

                # 2) 再恢复视觉 token 的空间维度 (dim=1,2)，只保留时间维平移
                if modified_pos_ids.shape[0] > 1:
                    modified_pos_ids[1:, b, s_idx : e_idx + 1] = position_ids[1:, b, s_idx : e_idx + 1]
            else:
                # 纯文本：以第一个 token 的时间为锚点进行整体平移
                anchor_time = modified_pos_ids[0, b, 0].item()
                delta = target_time - anchor_time
                if delta != 0:
                    modified_pos_ids[:, b, :] += delta

        return modified_pos_ids

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
        manual_time: Optional[Union[float, int]] = None, 
        **kwargs,
    ) -> Union[Tuple, StreamQwenModelOutput]:

        # 1. 计算原生 Position IDs
        if position_ids is None and input_ids is not None:
            pos_ids, rope_deltas_calc = self.model.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                attention_mask,
            )
            position_ids = pos_ids
            if rope_deltas is None:
                rope_deltas = rope_deltas_calc
            
            # 仅当 position_ids 是内部生成时，应用 shift
            if manual_time is not None:
                # print(f"[Forward] Shift -> {manual_time}")
                position_ids = self._apply_time_shift(position_ids, input_ids, int(manual_time))

        outputs = super().forward(
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
        
        if return_dict:
            return StreamQwenModelOutput(
                loss=outputs.loss,
                logits=outputs.logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                rope_deltas=outputs.rope_deltas
            )
        return outputs