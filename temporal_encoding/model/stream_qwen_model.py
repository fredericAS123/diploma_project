"""
StreamQwenModel — Append 模式流式 mRoPE 位置追踪。

参考 StreamingVLM (mit-han-lab/streaming-vlm) 的 append 模式实现：
  Branch 1 (首次 Prefill)：无 KV Cache → 标准 get_rope_index
  Branch 2 (Chunk Prefill)：有 Cache + seq_len > 1 → 局部 get_rope_index + 全局统一偏移
  Branch 3 (Decode)：有 Cache + seq_len == 1 → last_cache_position + 1

关键语义（源自官方 Qwen2.5-VL get_rope_index）：
  视觉 token 位置 = stack([t_index, h_index, w_index]) + text_len + st_idx
  偏移对 3 维统一施加 — 不需要单独恢复空间维。

流式状态（stream_state）：
  - _last_cache_position: 时间轴上最后一个 token 的位置 ID
  - _rope_deltas: 首次 prefill 计算出的 mRoPE 偏移量
  snapshot/restore 时需要一并保存/恢复。
"""

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
    """
    Qwen2.5-VL 流式推理扩展。

    内部自动追踪 mRoPE 位置，外部无需传入 manual_time。
    通过 stream_state 属性实现状态快照/恢复。
    """

    def __init__(self, config: Qwen2_5_VLConfig):
        super().__init__(config)
        self._last_cache_position: int = -1
        self._rope_deltas: Optional[torch.Tensor] = None
        print(
            f"[StreamQwenModel] Initialized. "
            f"vision_start={getattr(config, 'vision_start_token_id', 'N/A')}, "
            f"vision_end={getattr(config, 'vision_end_token_id', 'N/A')}"
        )

    # ── Stream State ───────────────────────────────────────────

    @property
    def stream_state(self) -> dict:
        """导出流式状态（用于 snapshot）。"""
        return {
            "last_cache_position": self._last_cache_position,
            "rope_deltas": (
                self._rope_deltas.clone()
                if self._rope_deltas is not None
                else None
            ),
        }

    @stream_state.setter
    def stream_state(self, state: dict):
        """恢复流式状态（用于 restore）。"""
        self._last_cache_position = state["last_cache_position"]
        rd = state["rope_deltas"]
        self._rope_deltas = rd.clone() if rd is not None else None

    def reset_stream_state(self):
        """重置流式状态（新视频开始时调用）。"""
        self._last_cache_position = -1
        self._rope_deltas = None

    # ── Version-compatible get_rope_index ─────────────────────

    def _get_rope_index(
        self,
        input_ids: torch.LongTensor,
        image_grid_thw: Optional[torch.LongTensor],
        video_grid_thw: Optional[torch.LongTensor],
        second_per_grid_ts: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        兼容不同 transformers 版本的 get_rope_index 调用。

        - 5 参数版本（含 second_per_grid_ts）
        - 4 参数版本（不含 second_per_grid_ts）
        """
        target = self.model if hasattr(self, "model") and hasattr(self.model, "get_rope_index") else self
        try:
            return target.get_rope_index(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                attention_mask=attention_mask,
            )
        except TypeError:
            return target.get_rope_index(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
            )

    # ── 缓存检测 ────────────────────────────────────────────────

    @staticmethod
    def _has_cache(past_key_values) -> bool:
        """检查 past_key_values 是否包含有效缓存。"""
        if past_key_values is None:
            return False
        if hasattr(past_key_values, "get_seq_length"):
            return past_key_values.get_seq_length() > 0
        if isinstance(past_key_values, (tuple, list)) and len(past_key_values) > 0:
            first = past_key_values[0]
            if isinstance(first, (tuple, list)) and len(first) > 0:
                return first[0].shape[2] > 0
        return False

    # ── 3 分支 Position IDs ────────────────────────────────────

    def _compute_streaming_position_ids(
        self,
        input_ids: torch.LongTensor,
        image_grid_thw: Optional[torch.LongTensor],
        video_grid_thw: Optional[torch.LongTensor],
        second_per_grid_ts: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        3 分支 position_ids 计算。

        Branch 1 (首次 Prefill): 无缓存 → 标准 get_rope_index
        Branch 2 (Chunk Prefill): 有缓存 + 多 token → 局部 get_rope_index + 全局统一偏移
        Branch 3 (Decode): 有缓存 + 单 token → last_cache_position + 1

        Returns:
            position_ids: [3, batch, seq_len]
            rope_deltas: [batch, 1] 或 None
        """
        has_cache = self._has_cache(past_key_values)
        seq_len = input_ids.shape[1]
        batch_size = input_ids.shape[0]

        if not has_cache:
            # ── Branch 1: 首次 Prefill ──
            position_ids, rope_deltas = self._get_rope_index(
                input_ids, image_grid_thw, video_grid_thw, second_per_grid_ts, attention_mask,
            )
            self._rope_deltas = rope_deltas

        elif seq_len > 1:
            # ── Branch 2: Chunk Prefill ──
            # attention_mask 可能包含 past_len + new_len；截取为仅 new_len
            local_mask = attention_mask
            if attention_mask is not None and attention_mask.shape[1] != seq_len:
                local_mask = attention_mask[:, -seq_len:]

            position_ids, _ = self._get_rope_index(
                input_ids, image_grid_thw, video_grid_thw, second_per_grid_ts, local_mask,
            )
            # 全局统一偏移（3 维同时 +offset）
            offset = self._last_cache_position + 1
            position_ids = position_ids.clone()
            position_ids += offset

        else:
            # ── Branch 3: Decode (单 token) ──
            next_pos = self._last_cache_position + 1
            position_ids = torch.full(
                (3, batch_size, 1),
                next_pos,
                dtype=torch.long,
                device=input_ids.device,
            )

        # 更新追踪：取 3 维的全局最大值（与 get_rope_index 中 st_idx = .max()+1 语义一致）
        # 对文本 token，3 维相同；对视觉 token，T 可能 ≠ H ≠ W
        self._last_cache_position = int(position_ids[:, 0, -1].max().item())

        return position_ids, self._rope_deltas

    # ── Forward ─────────────────────────────────────────────────

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
        second_per_grid_ts: Optional[torch.Tensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, StreamQwenModelOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 流式 3 分支计算 position_ids
        if position_ids is None and input_ids is not None:
            position_ids, rope_deltas_computed = self._compute_streaming_position_ids(
                input_ids, image_grid_thw, video_grid_thw,
                second_per_grid_ts, attention_mask, past_key_values,
            )
            if rope_deltas is None:
                rope_deltas = rope_deltas_computed

        # 调用父类 forward（position_ids 已设置，父类跳过 get_rope_index）
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
            second_per_grid_ts=second_per_grid_ts,
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
                rope_deltas=rope_deltas,  # 使用我们计算的值，而非父类 self.model.rope_deltas（可能为 None/过期）
            )
        return outputs
