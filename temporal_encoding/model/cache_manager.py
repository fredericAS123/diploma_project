"""
KV Cache Manager for Streaming VLM Inference.

解决 ask/ask_choice 污染 video_cache 的问题。
提供 snapshot/restore 机制保护缓存状态。

设计原理：
  - 在 Append 模式下，ask() 和 ask_choice() 会在视频缓存之上做 prefill + decode，
    但不应修改视频缓存，以便 QA 后继续追加帧。
  - 通过 snapshot()/restore() 显式保护。
  - snapshot/restore 同时保存 model.stream_state，确保 mRoPE 位置追踪一致。

KV Cache 淘汰集成 (v2):
  通过 KVCacheEvictor 实现 3 级递进淘汰策略，
  在 append_frame 后自动检查并触发淘汰，控制显存增长。
  参考 StreamingVLM (MIT-HAN-Lab) + StreamingLLM + LOOK-M。

注意: 淘汰操作不修改 model._last_cache_position —— 
      在 append 模式下，剩余 token 的 position ID 不变，
      新 token 仍从 _last_cache_position + 1 继续编号。
"""

import copy
import torch
from typing import Optional

from .kv_cache_eviction import KVCacheEvictor, EvictionConfig


class KVCacheManager:
    """管理 KV 缓存生命周期，支持 snapshot/restore + KV Cache 淘汰。"""

    def __init__(self, eviction_config: Optional[EvictionConfig] = None):
        self._cache = None
        self._snapshot = None
        self._snapshot_stream_state = None
        self._snapshot_tracker_state = None

        # ── 淘汰器 ──
        self._evictor: Optional[KVCacheEvictor] = None
        if eviction_config is not None:
            self._evictor = KVCacheEvictor(eviction_config)

    @property
    def evictor(self) -> Optional[KVCacheEvictor]:
        return self._evictor

    @property
    def eviction_enabled(self) -> bool:
        return self._evictor is not None

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
        保存当前缓存的深拷贝 + 模型流式状态 + token tracker 状态。
        在 ask()/ask_choice() 之前调用以保护视频缓存。

        Args:
            model: StreamQwenModel 实例（用于保存 stream_state）
        """
        self._snapshot = self.clone(self._cache)
        if model is not None and hasattr(model, 'stream_state'):
            self._snapshot_stream_state = model.stream_state
        else:
            self._snapshot_stream_state = None

        # 保存 token tracker 状态
        if self._evictor is not None and self._evictor.token_tracker is not None:
            self._snapshot_tracker_state = self._evictor.token_tracker.snapshot()
        else:
            self._snapshot_tracker_state = None

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

            # 恢复 token tracker 状态
            if (
                self._evictor is not None
                and self._evictor.token_tracker is not None
                and self._snapshot_tracker_state is not None
            ):
                self._evictor.token_tracker.restore(self._snapshot_tracker_state)
            self._snapshot_tracker_state = None

    def discard_snapshot(self):
        """丢弃快照但不恢复（当 update_state=True 时使用）。"""
        self._snapshot = None
        self._snapshot_stream_state = None

    # ── KV Cache Eviction ─────────────────────────────────────

    def set_first_chunk_info(self, first_chunk_cache_len: int):
        """记录首 chunk 的 cache 长度, 用于自动确定 sink_size。"""
        if self._evictor is not None:
            self._evictor.set_first_chunk_info(first_chunk_cache_len)

    def evict_if_needed(self) -> dict:
        """
        检查 KV Cache 是否超过预算，超过则执行淘汰。

        返回淘汰信息字典。不修改 model._last_cache_position。
        在 append 模式下，position ID 单调递增，淘汰不影响后续编码。

        Returns:
            dict with keys: evicted (bool), tokens_before, tokens_after, tokens_removed
        """
        if self._evictor is None or self._cache is None:
            return {"evicted": False}

        seq_len = self.get_seq_length()
        if not self._evictor.should_evict(seq_len):
            return {"evicted": False, "seq_len": seq_len}

        self._cache = self._evictor.evict(self._cache, seq_len)
        new_len = self.get_seq_length()

        return {
            "evicted": True,
            "tokens_before": seq_len,
            "tokens_after": new_len,
            "tokens_removed": seq_len - new_len,
        }

    def track_tokens(self, input_ids: torch.LongTensor, is_new_chunk: bool = True):
        """
        将 input_ids 中的 token 类型信息追加到 token tracker。

        Args:
            input_ids: [batch=1, seq_len] 的 token ID
            is_new_chunk: 是否为新的 chunk
        """
        if self._evictor is not None and self._evictor.token_tracker is not None:
            self._evictor.token_tracker.append_tokens(
                input_ids, self._evictor.config, is_new_chunk=is_new_chunk
            )

    def get_eviction_stats(self) -> dict:
        """获取淘汰统计信息。"""
        if self._evictor is not None:
            return self._evictor.get_stats()
        return {}

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
        self._snapshot_tracker_state = None
        if self._evictor is not None:
            self._evictor.reset()
