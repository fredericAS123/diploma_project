# 🎓 Diploma Project: Streaming VLM with KV Cache Eviction

基于 Qwen2.5-VL 的流式视频大语言模型（Streaming VLM）推理系统。

实现**持续前向传播保存 KV Cache，在收到用户问题时快速使用已有 KV Cache 进行回答**的核心能力。

**v2 新增**: KV Cache 淘汰策略（参考 StreamingVLM + StreamingLLM + LOOK-M），支持无限长度视频流处理。

---

## 📁 项目结构

```
diploma_project/
│
├── README.md                              # 项目根说明
├── NEXT_STEP_DIRECTION.md                 # 技术方向文档 v3
│
├── temporal_encoding/                     # ⭐ 核心模块：流式推理引擎
│   ├── model/                             # 流式推理核心代码
│   │   ├── __init__.py                    # 模块导出
│   │   ├── stream_qwen_model.py           # 流式 M-RoPE 位置追踪模型
│   │   ├── video_stream_inference.py      # 高层流式推理引擎 (v2: 集成淘汰)
│   │   ├── cache_manager.py               # KV Cache 生命周期管理器 (v2: 淘汰集成)
│   │   └── kv_cache_eviction.py           # 🆕 KV Cache 淘汰策略模块 (3 级递进)
│   │
│   ├── test_step1_cache.py  ~ test_step10 # 已有测试
│   ├── PROJECT_STRUCTURE.md               # 旧版项目结构
│   ├── PROJECT_STRUCTURE_V2.md            # 本文件: 更新后的项目结构
│   ├── TESTING_PROMPT.md                  # 已有测试指南
│   └── EVICTION_EXPERIMENT_PROMPT.md      # 🆕 淘汰策略实验验证 Prompt
│
├── qwen2_5_vl/                            # 参考代码与分析脚本
│   ├── modeling_qwen2_5_vl.py             # Qwen2.5-VL 模型实现源码
│   └── ...                                # 其他分析脚本与报告
│
└── web_demo/                              # Web 演示界面
    ├── main.py                            # Gradio 启动入口
    ├── Qwen_inference.py                  # 推理封装
    └── webui_gradio.py                    # Gradio Web UI
```

---

## 🏗️ 核心架构 (v2: 含淘汰)

### 系统设计

```
[视频流]
   │
   ├─ Chunk 0 ──> append_video_chunk() ──> ViT ──> LLM Prefill ──> KV Cache
   │  └─ 🆕 首 chunk 自动检测 → set_first_chunk_info(cache_len) → sink_size 确定
   │
   ├─ Chunk 1,2,... ──> append_video_chunk() ──> ViT ──> LLM Chunk Prefill ──> KV Cache (累积)
   │  └─ 🆕 update_chunk_stats() 更新平均 chunk token 数
   │
   │  ┌──────────────────────────────────────────────────────────────────┐
   │  │ 🆕 KV Cache Eviction (当 cache_len > max_cache_tokens)          │
   │  │   保留: sink(首 chunk, 自动检测) + window(尾部, 自动计算)        │
   │  │   淘汰: 中间区域 → 显存恒定不增长                                │
   │  │   注: 视频 cache ≈100% 视觉 token (ask 用 snapshot/restore)     │
   │  └──────────────────────────────────────────────────────────────────┘
   │
   └─ 用户提问 ──> ask() ──> Snapshot ──> QA Prefill ──> Decode ──> 答案
                                                                      │
                                                           Restore (保护视频流状态)
```

### 四层架构

| 层级 | 文件 | 职责 |
|------|------|------|
| **Model** | `stream_qwen_model.py` | 3-branch mRoPE 位置追踪，支持连续 chunk 追加 |
| **Eviction** | `kv_cache_eviction.py` | 🆕 3 级淘汰策略: Sink+Window / 均匀时序采样 / 帧级重要性 |
| **Cache Manager** | `cache_manager.py` | KV Cache 生命周期 + snapshot/restore + 淘汰集成 |
| **Inference Engine** | `video_stream_inference.py` | 完整流式推理: append_frame → auto-detect → evict → ask |

---

## 🆕 KV Cache 淘汰模块详解

### 文件: `model/kv_cache_eviction.py`

#### 核心类

| 类 | 职责 |
|----|------|
| `EvictionConfig` | 淘汰策略配置 (max_cache_tokens, sink/window 自动检测, ...) |
| `EvictionStats` | 淘汰统计信息 (总次数, 总淘汰 token 数, ...) |
| `TokenTypeTracker` | Token chunk 跟踪 (chunk_id, 用于 Level 2/3 的 chunk 粒度操作) |
| `KVCacheEvictor` | 淘汰器: set_first_chunk_info() → should_evict() → evict() |

#### 关键参数 (基于 test_step10 实测数据)

| 参数 | 默认值 | 依据 |
|------|--------|------|
| `max_cache_tokens` | 100,000 (需调优) | **核心超参数, 建议实验确定**。峰值 = max + 1 chunk, 不可超 ~155K (4090 24GB) |
| `sink_size` | 0 (自动) | 首次 `append_frame` 后以 cache 长度设定; 1920×1080 4帧 ≈ 5,438 tokens |
| `window_size` | 0 (自动) | = max_cache_tokens - effective_sink_size |
| `eviction_interval` | 1 | 每个 chunk 后检查, 防止峰值超限 |

> **max_cache_tokens 调优指南** (4090 24GB, 1920×1080, 4帧/chunk):
>
> | 设置 | cache 大小 | 总 VRAM (含模型) | window 大小 (sink≈5.4K时) | 说明 |
> |------|-----------|----------------|--------------------------|------|
> | 100,000 | ~3.4 GB | ~10.5 GB | ~94,600 (~17.5 chunk) | 保守, 大量余量 |
> | 130,000 | ~4.5 GB | ~11.6 GB | ~124,600 (~23 chunk) | 中等, 推荐起点 |
> | 150,000 | ~5.2 GB | ~12.3 GB | ~144,600 (~26.8 chunk) | 激进, 接近极限 |
>
> 峰值时 cache 额外增加 1 chunk (~5.4K tokens)。过大可能因 CUDA reserved 碎片 OOM;
> 过小则 window 不足, 丢失近期视频信息, 导致回答质量下降。
> **建议在实验 B 中从 130K 开始, 若稳定则尝试 150K。**

> ⚠️ **sink_size 不可硬编码**: 首 chunk 包含 system prompt + 首帧视觉 token，其 token 数随视频分辨率、chunk 帧数变化。例如 1920×1080 4帧/chunk ≈ 5,438; 2帧/chunk ≈ 2,750; 640×480 会更少。因此必须从实际首 chunk cache 长度自动检测。

#### 容量实测数据 (test_step10, RTX 4090 24GB)

| Chunk 数 | 帧数 | cache tokens | cache 大小 | VRAM allocated | VRAM reserved |
|----------|------|-------------|-----------|----------------|---------------|
| 10 | 40 | 53,939 | 1.85 GB | 8.97 GB | 12.13 GB |
| 20 | 80 | 107,829 | 3.70 GB | 10.82 GB | 18.09 GB |
| 30 | 120 | 161,719 | 5.55 GB | 12.67 GB | **22.89 GB** |
| **40** | **160** | — | — | — | **OOM** ❌ |

- 模型基线: 7.1 GB allocated, 7.33 GB reserved
- 每 chunk (1920×1080, 4帧): ~5,389 tokens, ~0.185 GB cache
- 每 token KV cache: ~36 KB (across 36 layers)
- **无淘汰极限: ~120 帧 (30 chunks)**, 不是 600 帧

#### 3 级递进策略

**Level 1: Attention Sink + Sliding Window**
```
|← sink(首chunk, 自动) →|← 全部淘汰 →|← window(尾部, 自动) →|
|   system + 首帧视觉     |  中间 chunks  |     最近 chunks      |
```
- 参考: StreamingLLM (arXiv:2309.17453)
- sink = 首 chunk 完整保留 (包含 attention sink + 首帧空间信息)
- window = max_cache_tokens - sink (剩余预算全给最近 token)
- 中间区域全部淘汰, 只保留首尾
- 显存预算: 恒定 max_cache_tokens × 36 KB, 不随视频增长
- 实现: `_evict_sink_window()` → `torch.index_select(cache, dim=2, indices)`

**Level 2: Sink + Window + 均匀时序采样**
```
|← sink →|← 均匀采样 chunk (30%) →|← window →|
|  首chunk |  保留部分中间 chunk     |  最近 chunks |
```
- 改进: 中间区域不全部删除, 而是按 chunk 粒度均匀采样保留
- `mid_retention_ratio=0.3`: 保留 30% 的中间 chunk (均匀分布)
- 整 chunk 保留确保帧内空间一致性 (同一帧的 token 不被拆分)
- 需要 `TokenTypeTracker` 记录 chunk 边界
- 实现: `_evict_temporal_sampling()`

> **为什么不用"模态感知"?** 在我们的系统中, `ask()` 使用 snapshot/restore 机制 —— QA 文本 token 不会进入视频 KV cache。因此视频 cache 中 **≈100% 为视觉 token** (每 chunk 仅 <7 个文本 wrapper token, 占比 <0.13%)。"按模态分类处理" 在此场景下无意义。

**Level 3: 帧级重要性评分**
```
|← sink →|← 重要帧(Top-K) →|← 最近 N chunk(全保留) →|← window →|
|  首chunk |  按分数筛选中间帧  |  最近 8 chunk          |  最近 token  |
```
- 以帧/chunk 为单位进行重要性评分 (如基于 attention score)
- 始终保留最近 `recent_frames_keep=8` 个 chunk (时间局部性)
- 需要 `TokenTypeTracker` 提供 chunk 信息
- 实现: `_evict_frame_importance()`

#### 关键设计决策

1. **sink_size 自动检测**: 首次 `append_frame()` 完成后, 调用 `set_first_chunk_info(cache_len)` 记录实际 cache 长度作为 sink_size。后续所有淘汰操作基于此值。这避免了对不同分辨率/帧数组合的硬编码猜测。

2. **淘汰不修改 `_last_cache_position`**: 在 append 模式下, position ID 单调递增。淘汰删除 KV 条目但不改变 position IDs, 新 token 继续从 `_last_cache_position + 1` 编号。这与 StreamingVLM 的 "append" pos_mode 一致。

3. **淘汰操作使用 `torch.index_select`**: 参考 StreamingVLM 的 `prune_id_and_kv_cache()`, 对 `DynamicCache.key_cache[i]` 和 `value_cache[i]` (shape `[batch, heads, seq, dim]`) 在 dim=2 做 index_select。

4. **chunk 粒度而非 token 粒度**: 视频帧的 token 在空间上高度耦合 (同一帧的 patch token 编码了空间关系)。淘汰半帧 token 会破坏空间一致性。因此 Level 2/3 以完整 chunk 为最小淘汰单位。

5. **Attention Mask 自动适配**: `build_full_attention_mask()` 内部查询 `cache.get_seq_length()`, 淘汰后自动缩短, 无需手动调整。

6. **Snapshot/Restore 包含 Tracker**: 在 `ask()` 前 snapshot 同时保存 tracker 状态, restore 时恢复。

---

## 📊 参考实现

### StreamingVLM (MIT-HAN-Lab)
- 仓库: https://github.com/mit-han-lab/streaming-vlm
- 核心方法: 对话轮次级别 KV 裁剪 + text_sink/text_sliding_window
- 本项目复用思路: `torch.index_select` 淘汰方式 + "append" position mode 设计

### StreamingLLM (arXiv:2309.17453)
- Attention Sink + Sliding Window 的理论基础

### LOOK-M (arXiv:2406.18139)
- 多模态 KV Cache 压缩，text-prior 策略
- 参考其分析思路, 但**未采用模态分离策略** (原因: 我们的 cache 几乎 100% 视觉 token)

---

## 🔧 使用方式

### 无淘汰（原有行为，完全兼容）

```python
from temporal_encoding.model import StreamQwenModel, VideoStreamingInference

model = StreamQwenModel.from_pretrained(model_path, ...)
engine = VideoStreamingInference(model, processor)  # 无 eviction_config → 禁用
engine.append_video_chunk(frames, fps=2.0)
answer, _ = engine.ask("What happened?")
```

### 启用 Level 1 淘汰 (推荐: 全部自动)

```python
from temporal_encoding.model import StreamQwenModel, VideoStreamingInference, EvictionConfig

# sink_size=0 + window_size=0 → 首 chunk 自动检测 sink, window 自动计算
config = EvictionConfig(
    max_cache_tokens=130_000,  # 建议通过实验调优; 范围: 100K(保守) ~ 150K(激进)
)
engine = VideoStreamingInference(model, processor, eviction_config=config)

for chunk in video_chunks:
    engine.append_video_chunk(chunk, fps=2.0)
    # 首 chunk: 自动检测 sink_size
    # 后续: 自动检查并执行淘汰
    print(engine.get_cache_info())

answer, _ = engine.ask("What happened in the video?")
```

### 启用 Level 2 均匀时序采样

```python
config = EvictionConfig(
    max_cache_tokens=100_000,
    # sink_size / window_size 保持 0 = 自动
    enable_temporal_sampling=True,  # 启用 Level 2
    mid_retention_ratio=0.3,       # 中间区域保留 30% chunk
)
engine = VideoStreamingInference(model, processor, eviction_config=config)
```

---

## 📈 容量预期 (RTX 4090 24GB, 1920×1080, 4帧/chunk)

| 场景 | 无淘汰 | Level 1 (Sink+Window) |
|------|--------|----------------------|
| 120 帧 (30 chunks, ~60s) | ✅ 极限 (22.89 GB reserved) | ✅ 无压力 |
| 160 帧 (40 chunks, ~80s) | ❌ **OOM** | ✅ cache 稳定在 100K tokens |
| 600 帧 (150 chunks, ~5min) | ❌ OOM | ✅ cache 稳定 |
| 3600 帧 (900 chunks, ~30min) | ❌ OOM | ✅ cache 稳定 |
| ∞ 帧 | ❌ OOM | ✅ cache 恒定 ≤ max_cache_tokens |

> 无淘汰 120 帧即为极限, 不是 600 帧。此表基于 test_step10 实测。

---

## 🧪 实验验证

详见 [EVICTION_EXPERIMENT_PROMPT.md](EVICTION_EXPERIMENT_PROMPT.md)。

三个实验分别验证:
1. **实验 A — sink_size 自动检测**: 验证首 chunk 自动检测的正确性, 不同分辨率/帧数下 sink 值是否合理
2. **实验 B — OOM-Free 长视频处理**: 处理完整 1.mp4, 确认淘汰后显存不超限
3. **实验 C — 滑窗逐段 + 周期性提问**: 编码 + 周期性 ask(), 验证淘汰不影响问答质量
