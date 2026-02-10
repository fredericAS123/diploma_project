# 🎓 Diploma Project: Streaming VLM Temporal Encoding

基于 Qwen2.5-VL 的流式视频大语言模型（Streaming VLM）推理系统。

实现**持续前向传播保存 KV Cache，在收到用户问题时快速使用已有 KV Cache 进行回答**的核心能力。

---

## 📁 项目结构

```
diploma_project/
│
├── README.md                              # 项目根说明
│
├── temporal_encoding/                     # ⭐ 核心模块：流式推理引擎
│   ├── model/                             # 流式推理核心代码
│   │   ├── __init__.py                    # 模块导出 (StreamQwenModel, VideoStreamingInference, KVCacheManager)
│   │   ├── stream_qwen_model.py           # 流式 M-RoPE 位置追踪模型
│   │   ├── video_stream_inference.py      # 高层流式推理引擎
│   │   └── cache_manager.py              # KV Cache 生命周期管理器
│   │
│   ├── test_step1_cache.py               # GPU 测试：KV Cache + Stream State 快照恢复
│   ├── test_step2_cache_logic.py          # CPU 测试：KVCacheManager 纯逻辑
│   ├── test_step3_prompt.py              # CPU 测试：Prompt 裁剪逻辑
│   ├── test_step4_choice_cache.py         # GPU 测试：ask_choice() 缓存隔离
│   ├── test_step5_e2e.py                 # GPU 测试：端到端多帧时序理解
│   ├── test_step6_stream_vs_native.py     # 🔥 GPU 测试：流式 vs 原生离线对比
│   ├── test_step7_multi_chunk.py          # GPU 测试：多帧 Chunk 规模性能
│   └── TESTING_PROMPT.md                  # 测试文档与运行指南
│
├── qwen2_5_vl/                            # 参考代码与分析脚本
│   ├── configuration_qwen2_5_vl.py        # Qwen2.5-VL 模型配置源码
│   ├── modeling_qwen2_5_vl.py             # Qwen2.5-VL 模型实现源码
│   ├── modular_qwen2_5_vl.py             # 模块化模型实现
│   ├── processing_qwen2_5_vl.py           # 处理器实现
│   ├── task1_inference_verify.py          # Task 1: 基础推理验证
│   ├── task2_mrope_analysis.py            # Task 2: M-RoPE 分析脚本
│   ├── task2_mrope_analysis_report.txt    # M-RoPE 分析报告
│   ├── task3_stream_mrope_analysis.py     # Task 3: 流式 M-RoPE 分析
│   ├── task3_mrope_report.txt             # 流式 M-RoPE 报告
│   ├── task4_video_native_analysis.py     # Task 4: 原生视频分析
│   ├── task4_mrope_report.txt             # 视频 M-RoPE 报告
│   ├── task5_stream_absolute_time_experiment.py  # Task 5: 绝对时间实验
│   └── task5_stream_absolute_time_report.txt     # 绝对时间实验报告
│
└── web_demo/                              # Web 演示界面
    ├── main.py                            # FastAPI 入口
    ├── Qwen_inference.py                  # 推理封装
    ├── RoPE_learning.py                   # RoPE 学习脚本
    ├── test_Qwen.py                       # 快速测试
    ├── webui_gradio.py                    # Gradio Web UI
    └── webui_Qwen2_5_3B.py               # 3B 模型 Web UI
```

---

## 🏗️ 核心架构

### 系统设计

```
[视频流]
   │
   ├─ Frame 1,2 ──> append_video_chunk() ──> ViT (Conv3D) ──> LLM Prefill ──> KV Cache
   ├─ Frame 3,4 ──> append_video_chunk() ──> ViT (Conv3D) ──> LLM Chunk Prefill ──> KV Cache (累积)
   ├─ Frame 5,6 ──> ...
   │
   └─ 用户提问 ──> ask() ──> Snapshot Cache ──> QA Prefill ──> Decode ──> 答案
                                                                              │
                                                                   Restore Cache (保护视频流状态)
```

### 三层架构

| 层级 | 文件 | 职责 |
|------|------|------|
| **应用层** | `video_stream_inference.py` | 高层 API：`append_frame()`, `ask()`, `ask_choice()`, `reset()` |
| **模型层** | `stream_qwen_model.py` | 3 分支 M-RoPE 位置追踪 + `stream_state` 管理 |
| **缓存层** | `cache_manager.py` | KV Cache 生命周期：snapshot/restore/clone/clear |

---

## 🔑 关键技术点

### 1. 3 分支 M-RoPE 位置追踪

基于 [StreamingVLM](https://github.com/mit-han-lab/streaming-vlm) 的 Append 模式：

| 分支 | 条件 | 行为 |
|------|------|------|
| **Branch 1** (首次 Prefill) | 无 KV Cache | 标准 `get_rope_index` 计算 3D (T,H,W) 位置 |
| **Branch 2** (Chunk Prefill) | 有 Cache + `seq_len > 1` | 局部 `get_rope_index` + 全局偏移 `offset = last_cache_position + 1` |
| **Branch 3** (Decode) | 有 Cache + `seq_len == 1` | `position = last_cache_position + 1`（3 维统一） |

**位置追踪：**
```python
# 取 3 维的跨维度最大值（与 get_rope_index 中 st_idx = .max()+1 语义一致）
self._last_cache_position = int(position_ids[:, 0, -1].max().item())
```

### 2. KV Cache 快照/恢复

```python
# ask() 前：保护视频缓存 + 模型流式状态
cache_manager.snapshot(model)   # 深拷贝 cache + model.stream_state

# QA 完成后：恢复到问答前的状态
cache_manager.restore(model)    # cache + stream_state 一并恢复

# 继续追加新帧：位置计算自动从正确位置继续
engine.append_video_chunk(new_frames)
```

### 3. Qwen2.5-VL 双 RoPE 系统

| 组件 | RoPE 类型 | 维度 | 作用域 |
|------|-----------|------|--------|
| **ViT** | 2D (H, W) | 空间位置 | Chunk 内注意力（零跨 chunk 交互） |
| **LLM** | 3D M-RoPE (T, H, W) | 时空位置 | 全局序列，`mrope_section` 通道分割 |

**关键发现：** ViT 对不同 temporal chunk 使用完全相同的位置编码（.repeat(t,1)），时序建模完全由 LLM 的 M-RoPE 负责。

---

## 🔧 核心 API

### VideoStreamingInference

```python
from temporal_encoding.model import StreamQwenModel, VideoStreamingInference
from transformers import AutoProcessor

# 初始化
processor = AutoProcessor.from_pretrained(model_path)
model = StreamQwenModel.from_pretrained(model_path, torch_dtype=torch.bfloat16).to("cuda")
engine = VideoStreamingInference(model, processor, "cuda")

# 流式编码（推荐 4 帧 chunk）
engine.append_video_chunk([frame0, frame1, frame2, frame3], fps=4.0)
engine.append_video_chunk([frame4, frame5, frame6, frame7], fps=4.0)

# 回答问题（不污染视频缓存）
answer, metrics = engine.ask("What happened?", max_new_tokens=128, update_state=False)
print(f"Answer: {answer}")
print(f"TTFT: {metrics['ttft']:.3f}s")

# 继续追加帧
engine.append_video_chunk([frame8, frame9], fps=2.0)

# 多选题
choice = engine.ask_choice("What color?", choices=["Red", "Blue", "Green"])

# 监控
info = engine.get_cache_info()

# 重置（新视频）
engine.reset()
```

### StreamQwenModel

```python
# stream_state 导出/恢复（用于自定义缓存管理）
state = model.stream_state                # 导出
model.stream_state = saved_state           # 恢复
model.reset_stream_state()                 # 重置

# forward 时自动计算 position_ids（外部无需传入）
outputs = model(input_ids=ids, attention_mask=mask, past_key_values=cache, use_cache=True)
```

### KVCacheManager

```python
manager = KVCacheManager()
manager.cache = outputs.past_key_values    # 保存缓存

manager.snapshot(model)                    # 快照（含 stream_state）
# ... 做 QA ...
manager.restore(model)                     # 恢复

cloned = manager.clone(manager.cache)      # 独立副本
full_mask = manager.build_full_attention_mask(new_mask)
manager.clear()                            # 释放内存
```

---

## 🐛 已修复的 Bug

### Bug 1 (Critical): `_last_cache_position` 维度错误
- **问题：** 只取了 T 维 `position_ids[0, 0, -1]`，忽略了 H/W 可能更大
- **修复：** `position_ids[:, 0, -1].max().item()` 取跨维度最大值
- **影响：** 后续 chunk 偏移错误导致位置冲突

### Bug 2 (Critical): `rope_deltas` 使用过期值
- **问题：** `StreamQwenModelOutput.rope_deltas` 使用了父类 `outputs.rope_deltas`
- **修复：** 使用我们计算的 `rope_deltas` 值
- **影响：** 后续 decode 位置计算错误

### Bug 3 (Medium): TTFT 测量点错误
- **问题：** TTFT 在第一个 decode step 之后才记录
- **修复：** 移动到 prefill 完成后立即记录
- **影响：** TTFT 指标不准确（包含了一次 decode 延迟）

---

## 📊 推荐 Chunk 配置

| Chunk 大小 | temporal_patch_size 对齐 | T 值 | 特点 | 推荐场景 |
|------------|-------------------------|------|------|----------|
| 2 帧 | ✅ | 1 | 最低延迟 | 实时交互 |
| 4 帧 | ✅ | 2 | 延迟/质量均衡 | **通用推荐** |
| 6 帧 | ✅ | 3 | 较高吞吐 | 准实时 |
| 8 帧 | ✅ | 4 | 最高吞吐 | 批处理 |
| 3 帧 | ❌（填充至 4） | 2 | 浪费计算 | 不推荐 |

---

## 🚀 快速开始

```bash
# 1. 环境准备
pip install torch transformers accelerate Pillow opencv-python

# 2. 运行 CPU 测试验证逻辑
cd temporal_encoding
python test_step2_cache_logic.py
python test_step3_prompt.py

# 3. 运行 GPU 测试验证功能
python test_step1_cache.py
python test_step5_e2e.py

# 4. 🔥 运行核心对比测试
python test_step6_stream_vs_native.py
```

---

**Last Updated:** 2026-02-10
