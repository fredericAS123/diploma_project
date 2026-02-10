"""
KV Cache Manager for Streaming VLM Inference.

解决 ask/ask_choice 污染 video_cache 的问题。
提供 snapshot/restore 机制保护缓存状态。

设计原理：
  - 在 Append 模式下，ask() 和 ask_choice() 会在视频缓存之上做 prefill + decode，
    但不应修改视频缓存，以便 QA 后继续追加帧。
  - 通过 snapshot()/restore() 显式保护。
  - snapshot/restore 同时保存 model.stream_state，确保 mRoPE 位置追踪一致。

参考 StreamingVLM 的 KV cache 管理模式（全量累积，不做 eviction）。
"""

import copy
import torch


class KVCacheManager:
    """管理 KV 缓存生命周期，支持 snapshot/restore（含模型流式状态）。"""

    def __init__(self):
        self._cache = None
        self._snapshot = None
        self._snapshot_stream_state = None

    # ── 属性 ────────────────────────────────────────────────────

    @property
    def cache(self):
        return self._cache

    @cache.setter
    def cache(self, value):
        self._cache = value

    # ── 序列长度 ────────────────────────────────────────────────

    def get_seq_length(self) -> int:
        """返回缓存中当前的序列长度。"""
        if self._cache is None:
            return 0
        # transformers DynamicCache
        if hasattr(self._cache, "get_seq_length"):
            return self._cache.get_seq_length()
        # Tuple-of-tuples: ((k, v), (k, v), ...)
        if isinstance(self._cache, (tuple, list)) and len(self._cache) > 0:
            first_layer = self._cache[0]
            if isinstance(first_layer, (tuple, list)) and len(first_layer) > 0:
                return first_layer[0].shape[2]  # [batch, heads, seq, dim]
        return 0

    # ── Detach ──────────────────────────────────────────────────

    @staticmethod
    def detach(past_key_values):
        """从计算图中分离 past_key_values。"""
        if past_key_values is None:
            return None
        # DynamicCache 已由 transformers 管理
        if hasattr(past_key_values, "get_seq_length"):
            return past_key_values
        # Tuple-of-tuples
        return tuple(
            tuple(t.detach() for t in layer)
            for layer in past_key_values
        )

    # ── Clone ───────────────────────────────────────────────────

    @staticmethod
    def clone(cache):
        """深拷贝 KV 缓存（用于多选题每个选项的独立评分等）。"""
        if cache is None:
            return None
        if hasattr(cache, "get_seq_length"):
            return copy.deepcopy(cache)
        return tuple(
            tuple(t.clone() for t in layer)
            for layer in cache
        )

    # ── Snapshot / Restore ──────────────────────────────────────

    def snapshot(self, model=None):
        """
        保存当前缓存的深拷贝 + 模型流式状态。
        在 ask()/ask_choice() 之前调用以保护视频缓存。

        Args:
            model: StreamQwenModel 实例（用于保存 stream_state）
        """
        self._snapshot = self.clone(self._cache)
        if model is not None and hasattr(model, 'stream_state'):
            self._snapshot_stream_state = model.stream_state
        else:
            self._snapshot_stream_state = None

    def restore(self, model=None):
        """
        将缓存和模型状态恢复到上次快照。

        Args:
            model: StreamQwenModel 实例（用于恢复 stream_state）
        """
        if self._snapshot is not None:
            self._cache = self._snapshot
            self._snapshot = None
            if (
                model is not None
                and hasattr(model, 'stream_state')
                and self._snapshot_stream_state is not None
            ):
                model.stream_state = self._snapshot_stream_state
            self._snapshot_stream_state = None

    def discard_snapshot(self):
        """丢弃快照但不恢复（当 update_state=True 时使用）。"""
        self._snapshot = None
        self._snapshot_stream_state = None

    # ── Attention Mask ──────────────────────────────────────────

    def build_full_attention_mask(
        self,
        new_mask: torch.Tensor,
        cache_override=None,
    ) -> torch.Tensor:
        """
        构建覆盖 [past_len + new_seq_len] 的 attention mask。

        Args:
            new_mask: shape [batch, new_seq_len]
            cache_override: 若提供，用此缓存测量 past_len
        """
        c = cache_override if cache_override is not None else self._cache
        past_len = 0
        if c is not None:
            if hasattr(c, "get_seq_length"):
                past_len = c.get_seq_length()
            elif isinstance(c, (tuple, list)) and len(c) > 0:
                first = c[0]
                if isinstance(first, (tuple, list)) and len(first) > 0:
                    past_len = first[0].shape[2]

        if past_len == 0:
            return new_mask

        past_mask = torch.ones(
            (new_mask.shape[0], past_len),
            dtype=new_mask.dtype,
            device=new_mask.device,
        )
        return torch.cat([past_mask, new_mask], dim=1)

    # ── Clear ───────────────────────────────────────────────────

    def clear(self):
        """释放所有缓存内存。"""
        self._cache = None
        self._snapshot = None
        self._snapshot_stream_state = None
