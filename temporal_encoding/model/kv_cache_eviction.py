"""
KV Cache Eviction Strategies for Streaming VLM Inference.

参考实现:
  - StreamingVLM (MIT-HAN-Lab, https://github.com/mit-han-lab/streaming-vlm)
    ├─ prune_id_and_kv_cache(): 按索引范围删除 KV 条目 (torch.index_select)
    ├─ process_past_kv(): 对话轮次级别的视觉/文本 KV 裁剪
    └─ text_sink + text_sliding_window: 前 N + 后 M 保留策略
  - StreamingLLM (arXiv:2309.17453): Attention Sink + Sliding Window
  - LOOK-M (arXiv:2406.18139): 多模态 KV Cache 压缩

适配说明:
  我们的流式视频系统与 StreamingVLM 有关键区别:
    1) ask() 使用 snapshot/restore, QA 文本不进入视频 KV cache
    2) 因此视频 KV cache 中几乎 100% 为视觉 token (每 chunk <0.13% 文本 wrapper)
    3) 淘汰粒度以 "chunk"(视频帧组) 为单位更合理, 而非 token 级

  关键参数依据 (来自 test_step10 在 RTX 4090 24GB 上的实测):
    - 模型 VRAM (Qwen2.5-VL-3B, bf16): ~7.1 GB
    - KV cache 每 token: ~36 KB (across 36 layers)
    - 1920×1080, 4帧/chunk: ~5,389 tokens/chunk, ~0.185 GB/chunk
    - 30 chunks (120帧): cache 161,719 tokens, VRAM reserved 22.89 GB → 极限
    - 40 chunks (160帧): OOM
    - 安全 cache 预算: ~100,000 tokens

三级递进:
  Level 1: Attention Sink + Sliding Window (首 chunk 保留 + 最近 N token)
  Level 2: Sink + Window + 均匀时序采样 (中间区域按 chunk 均匀保留, 增强时序覆盖)
  Level 3: Sink + Window + 帧级重要性评分 (保留最重要帧 + 最近帧)
"""

import torch
import math
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass, field


@dataclass
class EvictionConfig:
    """KV Cache 淘汰策略配置。

    参数设计依据:
      - max_cache_tokens: 此值为需要实验调优的核心超参数。
        淘汰发生在 forward pass 之后, 因此峰值 cache = max_cache_tokens + 1个chunk。
        test_step10 实测 (4090 24GB, 1920×1080, 4帧/chunk ≈ 5,389 tokens/chunk):
          · 161,719 tokens → reserved 22.89 GB (能跑, 但极限)
          · ~215,000 tokens → OOM
          · 安全峰值 ≈ 155K tokens → max_cache_tokens ≈ 150,000 (激进)
        推荐通过实验确定最优值:
          · 100,000 = 保守 (~3.4 GB cache, total ~10.5 GB, 大量余量)
          · 130,000 = 中等 (~4.5 GB cache, total ~11.6 GB)
          · 150,000 = 激进 (~5.2 GB cache, total ~12.3 GB, peak ~155K 接近极限)
        注意 sink 过大会压缩 window, 影响近期信息保留和回答质量。
      - sink_size=0: 自动检测 — 首次 append_frame 后以 cache 长度作为 sink。
        首 chunk 包含 system prompt + 首帧视觉 token, 构成 attention sink。
        1920×1080 4帧 chunk ≈ 5,438 tokens; 2帧 ≈ 2,750 tokens。
        不可硬编码, 因视频分辨率/chunk帧数不同而异。
      - window_size=0: 自动计算 = max_cache_tokens - sink_size。
        等价于: sink 后的全部预算给 sliding window。
    """

    # ── 通用参数 ──
    max_cache_tokens: int = 100_000
    """KV Cache 最大允许 token 数。超过此值触发淘汰。
    默认 100,000 为保守值 (~3.4 GB cache), 建议通过实验调优。
    调优范围 (4090 24GB): 100K (保守) ~ 150K (激进)。
    注意: 峰值 cache = max_cache_tokens + chunk_tokens, 需留前向激活余量。
    窗口大小 = max - sink, 过小会导致近期信息不足、回答质量下降。"""

    sink_size: int = 0
    """Attention Sink: 保留的首部 token 数。
    0 = 自动检测 (推荐): 首次 append_frame 后自动设为 cache 长度。
    >0 = 手动指定 (仅在已知首 chunk token 数时使用)。"""

    window_size: int = 0
    """Sliding Window: 保留的尾部 token 数。
    0 = 自动计算: max_cache_tokens - effective_sink_size。
    >0 = 手动指定。注意: sink_size + window_size 必须 < max_cache_tokens。"""

    # ── Level 2: 均匀时序采样参数 ──
    enable_temporal_sampling: bool = False
    """是否启用均匀时序采样 (Level 2)。
    启用后, 中间区域不是全部删除, 而是按 chunk 均匀保留, 增强时序覆盖。"""

    mid_retention_ratio: float = 0.3
    """中间区域保留比例。0.3 = 保留 30% 的中间 chunk。
    实际意义: 如果中间有 20 个 chunk, 保留 6 个均匀分布的 chunk。"""

    # ── Level 3: 帧级重要性参数 ──
    enable_frame_importance: bool = False
    """是否启用帧级重要性评分 (Level 3)。"""

    recent_frames_keep: int = 8
    """始终保留的最近帧/chunk 数 (仅 Level 3)。"""

    # ── Token 类型识别 (Level 2/3 需要) ──
    visual_token_id: int = 151656
    """Qwen2.5-VL 的 <|video_pad|> token ID。"""

    vision_start_token_id: int = 151652
    """Qwen2.5-VL 的 <|vision_start|> token ID。"""

    vision_end_token_id: int = 151653
    """Qwen2.5-VL 的 <|vision_end|> token ID。"""

    # ── 淘汰频率 ──
    eviction_interval: int = 1
    """每追加多少个 chunk 后执行一次淘汰检查。
    1 = 每个 chunk 后检查 (推荐, 防止峰值超限)。"""


@dataclass
class EvictionStats:
    """淘汰操作的统计信息。"""
    total_evictions: int = 0
    total_tokens_evicted: int = 0
    last_eviction_seq_len_before: int = 0
    last_eviction_seq_len_after: int = 0
    last_eviction_tokens_removed: int = 0


class TokenTypeTracker:
    """
    跟踪 KV Cache 中每个 token 的模态类型和 chunk 归属。

    在 append_frame 编码时同步更新, 用于 Level 2/3 淘汰决策。

    注意: 在流式视频模式下, KV cache 中几乎全部是视觉 token,
    因为 ask() 使用 snapshot/restore 不污染视频 cache。
    每 chunk 仅有 ~7 个文本 wrapper token (<|im_start|>user, <|im_end|> 等),
    占比 <0.13%。
    """

    def __init__(self):
        self._is_visual: List[bool] = []
        self._chunk_id: List[int] = []
        self._current_chunk: int = -1

    @property
    def length(self) -> int:
        return len(self._is_visual)

    @property
    def is_visual(self) -> List[bool]:
        return self._is_visual

    @property
    def chunk_id(self) -> List[int]:
        return self._chunk_id

    @property
    def current_chunk(self) -> int:
        return self._current_chunk

    def append_tokens(
        self,
        input_ids: torch.LongTensor,
        config: EvictionConfig,
        is_new_chunk: bool = True,
    ):
        """
        根据 input_ids 中的 token 类型标记新增 token。

        Args:
            input_ids: [batch=1, seq_len] 的 token ID
            config: 包含 visual/vision token ID 的配置
            is_new_chunk: 是否为新的 chunk
        """
        if is_new_chunk:
            self._current_chunk += 1

        ids = input_ids[0].tolist() if input_ids.dim() > 1 else input_ids.tolist()

        in_vision_block = False
        for tid in ids:
            if tid == config.vision_start_token_id:
                in_vision_block = True
            elif tid == config.vision_end_token_id:
                in_vision_block = False

            is_vis = in_vision_block or (tid == config.visual_token_id)
            self._is_visual.append(is_vis)
            self._chunk_id.append(self._current_chunk)

    def get_chunk_ranges(self) -> Dict[int, Tuple[int, int]]:
        """返回每个 chunk 的 token 范围 {chunk_id: (start_idx, end_idx)}。"""
        ranges: Dict[int, Tuple[int, int]] = {}
        for i, cid in enumerate(self._chunk_id):
            if cid not in ranges:
                ranges[cid] = (i, i)
            else:
                ranges[cid] = (ranges[cid][0], i)
        return ranges

    def apply_eviction_mask(self, keep_mask: List[bool]):
        """根据保留掩码更新跟踪器（与 KV Cache 同步）。"""
        assert len(keep_mask) == len(self._is_visual), \
            f"Mask length {len(keep_mask)} != tracker length {len(self._is_visual)}"
        self._is_visual = [v for v, k in zip(self._is_visual, keep_mask) if k]
        self._chunk_id = [c for c, k in zip(self._chunk_id, keep_mask) if k]

    def clear(self):
        """重置跟踪器。"""
        self._is_visual.clear()
        self._chunk_id.clear()
        self._current_chunk = -1

    def snapshot(self) -> dict:
        """保存跟踪器状态。"""
        return {
            "is_visual": self._is_visual.copy(),
            "chunk_id": self._chunk_id.copy(),
            "current_chunk": self._current_chunk,
        }

    def restore(self, state: dict):
        """恢复跟踪器状态。"""
        self._is_visual = state["is_visual"]
        self._chunk_id = state["chunk_id"]
        self._current_chunk = state["current_chunk"]


class KVCacheEvictor:
    """
    KV Cache 淘汰器。

    三级递进策略:

    Level 1 (Sink + Sliding Window):
      保留首 chunk (attention sink) + 最近 window token。
      中间全部删除。最简单有效。
      参考: StreamingLLM (arXiv:2309.17453)

    Level 2 (Sink + Window + 均匀时序采样):
      中间区域按 chunk 粒度均匀采样保留。
      保持时序覆盖——即使很早的视频段也有 "锚点" chunk 被保留。

    Level 3 (Frame-Level Importance):
      以帧为单位评估重要性, 保留最重要的 N 帧 + 最近 M 帧。
    """

    def __init__(self, config: EvictionConfig):
        self.config = config
        self.stats = EvictionStats()
        self.token_tracker: Optional[TokenTypeTracker] = None

        # 自动检测的 sink 大小 (首 chunk 的 token 数)
        self._effective_sink_size: Optional[int] = None
        # 自动计算的 window 大小
        self._effective_window_size: Optional[int] = None
        # 每 chunk 的平均 token 数
        self._avg_tokens_per_chunk: Optional[float] = None
        # 首 chunk 已记录标志
        self._first_chunk_recorded: bool = False

        if config.enable_temporal_sampling or config.enable_frame_importance:
            self.token_tracker = TokenTypeTracker()

    def set_first_chunk_info(self, first_chunk_cache_len: int):
        """
        记录首 chunk 编码后的 cache 长度, 用于自动确定 sink_size。

        必须在第一个 append_frame() 完成后调用。

        Args:
            first_chunk_cache_len: 首 chunk 编码后 KV cache 的总 token 数。
                包含 system prompt + 首帧视觉 token + 文本 wrapper。
                例如: 1920×1080 4帧 chunk ≈ 5,438 tokens。
        """
        self._first_chunk_recorded = True

        if self.config.sink_size == 0:
            self._effective_sink_size = first_chunk_cache_len
        else:
            self._effective_sink_size = self.config.sink_size

        self._avg_tokens_per_chunk = float(first_chunk_cache_len)

        if self.config.window_size == 0:
            self._effective_window_size = max(
                0, self.config.max_cache_tokens - self._effective_sink_size
            )
        else:
            self._effective_window_size = self.config.window_size

        # 安全检查
        total_reserved = self._effective_sink_size + self._effective_window_size
        if total_reserved >= self.config.max_cache_tokens:
            # window 自动缩减以确保有淘汰空间
            self._effective_window_size = max(
                0, self.config.max_cache_tokens - self._effective_sink_size - 1
            )

    def update_chunk_stats(self, new_chunk_tokens: int):
        """更新每 chunk 平均 token 数。"""
        if self._avg_tokens_per_chunk is not None:
            self._avg_tokens_per_chunk = (
                0.8 * self._avg_tokens_per_chunk + 0.2 * new_chunk_tokens
            )

    @property
    def effective_sink_size(self) -> int:
        if self._effective_sink_size is not None:
            return self._effective_sink_size
        return self.config.sink_size

    @property
    def effective_window_size(self) -> int:
        if self._effective_window_size is not None:
            return self._effective_window_size
        return self.config.window_size

    def should_evict(self, cache_seq_len: int) -> bool:
        """判断是否需要执行淘汰。"""
        if not self._first_chunk_recorded:
            return False  # 首 chunk 尚未记录, 不淘汰
        return cache_seq_len > self.config.max_cache_tokens

    def evict(self, cache, cache_seq_len: Optional[int] = None) -> object:
        """对 DynamicCache 执行淘汰操作。"""
        if cache is None:
            return cache

        if cache_seq_len is None:
            if hasattr(cache, "get_seq_length"):
                cache_seq_len = cache.get_seq_length()
            else:
                return cache

        if not self.should_evict(cache_seq_len):
            return cache

        cfg = self.config

        if cfg.enable_frame_importance and self.token_tracker is not None:
            return self._evict_frame_importance(cache, cache_seq_len)
        elif cfg.enable_temporal_sampling and self.token_tracker is not None:
            return self._evict_temporal_sampling(cache, cache_seq_len)
        else:
            return self._evict_sink_window(cache, cache_seq_len)

    # ── Level 1: Attention Sink + Sliding Window ──────────────

    def _evict_sink_window(self, cache, seq_len: int):
        """
        Level 1: 保留首 chunk (sink) + 最近 window token, 删除全部中间。

        原理 (StreamingLLM):
          - 首 token 累积了不成比例的高注意力权重 (attention sink)
          - 在我们的系统中, "首部" 扩展为整个首 chunk (system prompt + 首帧视觉)

        实现 (参考 StreamingVLM prune_id_and_kv_cache):
          key_cache[i] = torch.index_select(k, 2, indices_to_keep)
        """
        sink = self.effective_sink_size
        window = self.effective_window_size

        if seq_len <= sink + window:
            return cache

        window = min(window, seq_len - sink)

        indices_to_keep = list(range(sink)) + list(range(seq_len - window, seq_len))
        indices_tensor = torch.tensor(
            indices_to_keep,
            device=self._get_cache_device(cache),
            dtype=torch.long,
        )

        new_cache = self._apply_index_select(cache, indices_tensor)

        if self.token_tracker is not None:
            keep_mask = [False] * seq_len
            for idx in indices_to_keep:
                keep_mask[idx] = True
            self.token_tracker.apply_eviction_mask(keep_mask)

        tokens_removed = seq_len - len(indices_to_keep)
        self._update_stats(seq_len, len(indices_to_keep), tokens_removed)

        return new_cache

    # ── Level 2: Sink + Window + 均匀时序采样 ─────────────────

    def _evict_temporal_sampling(self, cache, seq_len: int):
        """
        Level 2: 中间区域按 chunk 粒度均匀采样保留。

        策略:
        1. 保留首 chunk 全部 token (attention sink)
        2. 保留尾部 window_size 个 token (recent context)
        3. 中间区域: 按 chunk 均匀采样, 保留 mid_retention_ratio 比例的完整 chunk
           — 保持帧内空间关系完整, 保持时序覆盖

        优势 vs Level 1:
          Level 1 中间全删 → 对 "之前发生了什么" 记忆为零
          Level 2 均匀保留锚点 chunk → 即使很早的视频段也有上下文参考
        """
        sink = self.effective_sink_size
        window = self.effective_window_size

        if seq_len <= sink + window:
            return cache

        window = min(window, seq_len - sink)

        tracker = self.token_tracker
        if tracker is None or tracker.length != seq_len:
            return self._evict_sink_window(cache, seq_len)

        mid_start = sink
        mid_end = seq_len - window
        chunk_ranges = tracker.get_chunk_ranges()

        # 筛选完全位于中间区域的 chunk
        mid_chunk_ids = []
        for cid, (cs, ce) in sorted(chunk_ranges.items()):
            if cs >= mid_start and ce < mid_end:
                mid_chunk_ids.append(cid)

        if not mid_chunk_ids:
            return self._evict_sink_window(cache, seq_len)

        # 计算保留数量
        n_mid_chunks = len(mid_chunk_ids)
        mid_budget = self.config.max_cache_tokens - sink - window
        if mid_budget <= 0:
            return self._evict_sink_window(cache, seq_len)

        n_keep = max(1, int(n_mid_chunks * self.config.mid_retention_ratio))

        avg_chunk_tokens = sum(
            chunk_ranges[cid][1] - chunk_ranges[cid][0] + 1
            for cid in mid_chunk_ids
        ) / n_mid_chunks

        while n_keep > 0 and n_keep * avg_chunk_tokens > mid_budget:
            n_keep -= 1

        if n_keep <= 0:
            return self._evict_sink_window(cache, seq_len)

        kept_chunk_ids = set(self._uniform_sample(mid_chunk_ids, n_keep))

        keep_set = set(range(sink))
        keep_set.update(range(seq_len - window, seq_len))
        for cid in kept_chunk_ids:
            cs, ce = chunk_ranges[cid]
            keep_set.update(range(cs, ce + 1))

        indices_to_keep = sorted(keep_set)
        indices_tensor = torch.tensor(
            indices_to_keep,
            device=self._get_cache_device(cache),
            dtype=torch.long,
        )

        new_cache = self._apply_index_select(cache, indices_tensor)

        keep_mask = [i in keep_set for i in range(seq_len)]
        tracker.apply_eviction_mask(keep_mask)

        tokens_removed = seq_len - len(indices_to_keep)
        self._update_stats(seq_len, len(indices_to_keep), tokens_removed)

        return new_cache

    # ── Level 3: Frame-Level Importance ───────────────────────

    def _evict_frame_importance(self, cache, seq_len: int):
        """
        Level 3: 保留最近 N 个 chunk + 中间均匀采样 chunk。

        类似 Level 2, 但额外保证最近 recent_frames_keep 个 chunk 不被淘汰。
        """
        sink = self.effective_sink_size
        window = self.effective_window_size

        if seq_len <= sink + window:
            return cache

        window = min(window, seq_len - sink)

        tracker = self.token_tracker
        if tracker is None or tracker.length != seq_len:
            return self._evict_sink_window(cache, seq_len)

        chunk_ranges = tracker.get_chunk_ranges()
        all_chunk_ids = sorted(chunk_ranges.keys())

        if not all_chunk_ids:
            return self._evict_sink_window(cache, seq_len)

        # 最近 N 个 chunk 一定保留 (不含 sink 中的 chunk)
        recent_chunks = set()
        for cid in reversed(all_chunk_ids):
            cs, _ = chunk_ranges[cid]
            if cs < sink:
                continue
            if len(recent_chunks) >= self.config.recent_frames_keep:
                break
            recent_chunks.add(cid)

        mid_start = sink
        mid_end = seq_len - window
        mid_chunk_ids = []
        for cid in all_chunk_ids:
            if cid in recent_chunks:
                continue
            cs, ce = chunk_ranges[cid]
            if cs >= mid_start and ce < mid_end:
                mid_chunk_ids.append(cid)

        recent_tokens = sum(
            chunk_ranges[c][1] - chunk_ranges[c][0] + 1
            for c in recent_chunks
            if chunk_ranges[c][0] >= sink
        )
        mid_budget = max(0, self.config.max_cache_tokens - sink - window - recent_tokens)

        kept_mid = set()
        if mid_chunk_ids and mid_budget > 0:
            avg_ct = sum(
                chunk_ranges[c][1] - chunk_ranges[c][0] + 1
                for c in mid_chunk_ids
            ) / len(mid_chunk_ids)

            n_keep = max(1, int(mid_budget / avg_ct))
            n_keep = min(n_keep, len(mid_chunk_ids))
            kept_mid = set(self._uniform_sample(mid_chunk_ids, n_keep))

        keep_set = set(range(sink))
        keep_set.update(range(seq_len - window, seq_len))
        for cid in recent_chunks | kept_mid:
            cs, ce = chunk_ranges[cid]
            keep_set.update(range(cs, ce + 1))

        # 超预算保护
        if len(keep_set) > self.config.max_cache_tokens:
            for cid in sorted(kept_mid):
                if len(keep_set) <= self.config.max_cache_tokens:
                    break
                cs, ce = chunk_ranges[cid]
                for i in range(cs, ce + 1):
                    keep_set.discard(i)

        indices_to_keep = sorted(keep_set)
        indices_tensor = torch.tensor(
            indices_to_keep,
            device=self._get_cache_device(cache),
            dtype=torch.long,
        )

        new_cache = self._apply_index_select(cache, indices_tensor)

        keep_mask = [i in keep_set for i in range(seq_len)]
        tracker.apply_eviction_mask(keep_mask)

        tokens_removed = seq_len - len(indices_to_keep)
        self._update_stats(seq_len, len(indices_to_keep), tokens_removed)

        return new_cache

    # ── 工具方法 ──────────────────────────────────────────────

    @staticmethod
    def _uniform_sample(indices: list, n_keep: int) -> list:
        """从 indices 中均匀采样 n_keep 个元素。"""
        if n_keep <= 0:
            return []
        if n_keep >= len(indices):
            return list(indices)
        step = len(indices) / n_keep
        return [indices[int(i * step)] for i in range(n_keep)]

    @staticmethod
    def _get_cache_device(cache) -> torch.device:
        if hasattr(cache, "key_cache") and len(cache.key_cache) > 0:
            t = cache.key_cache[0]
            if hasattr(t, "device"):
                return t.device
        return torch.device("cpu")

    @staticmethod
    def _apply_index_select(cache, indices: torch.Tensor):
        """
        对 DynamicCache 执行 index_select 淘汰。

        参考 StreamingVLM prune_id_and_kv_cache():
          past_key_values.key_cache[i] = torch.index_select(k_layer, 2, indices_tensor)
        """
        if not hasattr(cache, "key_cache") or not hasattr(cache, "value_cache"):
            return cache

        for i in range(len(cache.key_cache)):
            k = cache.key_cache[i]
            v = cache.value_cache[i]
            if hasattr(k, "shape") and k.dim() >= 3:
                cache.key_cache[i] = torch.index_select(k, 2, indices)
                cache.value_cache[i] = torch.index_select(v, 2, indices)

        return cache

    def _update_stats(self, before: int, after: int, removed: int):
        self.stats.total_evictions += 1
        self.stats.total_tokens_evicted += removed
        self.stats.last_eviction_seq_len_before = before
        self.stats.last_eviction_seq_len_after = after
        self.stats.last_eviction_tokens_removed = removed

    def reset(self):
        """重置淘汰器全部状态。"""
        self.stats = EvictionStats()
        self._effective_sink_size = None
        self._effective_window_size = None
        self._avg_tokens_per_chunk = None
        self._first_chunk_recorded = False
        if self.token_tracker is not None:
            self.token_tracker.clear()

    def get_stats(self) -> dict:
        return {
            "total_evictions": self.stats.total_evictions,
            "total_tokens_evicted": self.stats.total_tokens_evicted,
            "effective_sink_size": self.effective_sink_size,
            "effective_window_size": self.effective_window_size,
            "avg_tokens_per_chunk": (
                round(self._avg_tokens_per_chunk) if self._avg_tokens_per_chunk else None
            ),
            "last_eviction": {
                "seq_len_before": self.stats.last_eviction_seq_len_before,
                "seq_len_after": self.stats.last_eviction_seq_len_after,
                "tokens_removed": self.stats.last_eviction_tokens_removed,
            },
        }
