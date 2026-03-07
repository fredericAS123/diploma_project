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

四个实验分别验证:
1. **实验 A — sink_size 自动检测**: 验证首 chunk 自动检测的正确性, 不同分辨率/帧数下 sink 值是否合理。
2. **实验 B — OOM-Free 长视频处理**: 处理完整 `202208312002.mp4`, 确认淘汰后显存不超限。
3. **实验 C — 滑窗逐段 + 周期性提问**: 编码 + 周期性 ask(), 验证淘汰不影响问答稳定性。
4. **实验 D — 全视频累计歌词抽取**: 按“截至当前时刻累计可见文本”持续提问, 输出去重后的全视频歌词集合。


---

## 🧩 2026-02-24 项目改进更新（v2.1）

### 代码层关键修复

1. **DynamicCache 兼容修复（关键）**
   - 问题: 仅按旧结构 `key_cache/value_cache` 做淘汰会在新 `transformers` 结构下“统计显示已淘汰，但实际序列长度未下降”。
   - 改进: 在淘汰逻辑中兼容 `cache.layers[i].keys/values` 结构，确保 `torch.index_select` 真正裁剪 KV。
   - 结果: `cache_len` 在超限后稳定回落到 `max_cache_tokens`，OOM 风险显著下降。

2. **窗口自动计算 off-by-one 修复**
   - 问题: `sink + window == max` 时被额外减 1，导致窗口值与设计不一致。
   - 改进: 仅在 `sink + window > max` 时才回退窗口。
   - 结果: 与实验 A 判定严格一致（`window = max - sink`）。

3. **chunk 平均 token 统计修复**
   - 问题: 旧统计口径与测试脚本断言不一致，出现平均值偏差。
   - 改进: 统一为“首 chunk 后的简单平均”，并与 `EvictionConfig` 自动检测流程对齐。
   - 结果: 实验 A 的 `avg_chunk_tokens` 与实际增量稳定匹配。

4. **缓存长度读取与掩码构造一致性修复**
   - 问题: 淘汰后若仅读取内部计数而非实际 tensor shape，可能出现注意力掩码长度滞后。
   - 改进: `cache_manager.py` 优先读取真实 KV tensor 的 `seq` 维长度。
   - 结果: 淘汰后 attention mask 与真实 cache 长度严格一致。

### 新增/强化测试脚本与作用

| 文件 | 作用 | 关键价值 |
|------|------|----------|
| `test_eviction_exp_a.py` | sink/window 自动检测正确性 | 保证淘汰基线参数正确 |
| `test_eviction_exp_b.py` | 长视频 OOM-Free | 验证显存可控与 `ask()` 可用性 |
| `test_eviction_exp_c.py` | 周期问答稳定性 | 验证 snapshot/restore 在频繁淘汰下可用 |
| `test_eviction_exp_c_window_compare_report.txt` | 无窗口 vs 有窗口回答质量对比结果 | 证明滑动窗口对歌词抽取有提升 |
| `test_eviction_exp_d_cumulative_lyrics.py` | **新增**：全视频累计歌词抽取实验 | 满足“截至当前累计字幕”业务目标 |
| `test_eviction_exp_d_cumulative_lyrics_report_v2.txt` | **新增**：迭代后精炼报告 | 形成可复现的最终结果基线 |

### 实验 D 设计思路（新增）

- **核心问题**: 传统“当前帧 OCR 提问”容易漏掉早期字幕，不满足“迄今为止累计输出”的需求。
- **策略**:
  1) 维持流式 append + eviction；
  2) 每固定 chunk 周期提问“截至当前累计歌词”；
  3) 同时补问“当前帧歌词”作为兜底；
  4) 合并、清洗、去重、按首次时间排序输出最终歌词清单。
- **迭代优化**:
  - v1: 高召回但噪声多（品牌词、歌手名、制作信息混入）；
  - v2: 加强 prompt + 规则过滤，保留歌词正文，显著提升可读性。

### 当前推荐默认配置

- `VIDEO_PATH=/root/autodl-tmp/diploma_project/temporal_encoding/202208312002.mp4`
- `MAX_CACHE_TOKENS=150000`（4090 24GB 下稳定）
- `CHUNK_FRAMES=4`
- `SAMPLE_FPS=2.0`
- 累计歌词任务建议 `ASK_INTERVAL=10`

---

## ✅ 面向“原生 Qwen2.5-VL 不微调”可行性结论

- **工程可行性**: 高。通过上述修复，系统已能稳定长时运行、稳定淘汰、稳定问答恢复。
- **任务可行性（歌词字幕抽取）**: 中高。实测可提取多条歌词正文，且滑动窗口优于无窗口。
- **上限判断**: 不微调条件下，模型仍可能在以下场景退化：
  1) 低对比度/动态模糊字幕；
  2) 字幕样式极端复杂；
  3) 长时段后重复/广告词污染。

**结论**: 在“不做额外微调”的约束下，依靠当前改进（动态淘汰修复 + 累计提问策略 + 文本清洗）可以**成功达到可用级别**，并能满足课程项目的系统目标；若追求“接近人工字幕”的工业级准确率，后续仍建议加入轻量适配（如 OCR 辅助或任务微调）。
