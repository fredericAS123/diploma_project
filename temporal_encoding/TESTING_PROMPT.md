# Temporal Encoding Streaming VLM Test Suite

完整的测试文档，用于验证 Qwen2.5-VL 流式推理系统的正确性与性能。

---

## 📁 测试文件概览

### 核心逻辑测试（无需 GPU）
- **test_step2_cache_logic.py** - KVCacheManager 纯逻辑测试
- **test_step3_prompt.py** - Prompt 裁剪逻辑测试

### 功能测试（需要 GPU + 模型）
- **test_step1_cache.py** - KV Cache + Stream State 快照/恢复隔离测试
- **test_step4_choice_cache.py** - ask_choice() 多选项缓存隔离测试
- **test_step5_e2e.py** - 端到端多帧时序理解测试

### 性能与对比测试（需要 GPU + 模型 + 视频）
- **test_step6_stream_vs_native.py** - 🔥 **核心测试**：流式 vs 原生离线推理全面对比
- **test_step7_multi_chunk.py** - 多帧 Chunk 规模性能测试

---

## 🎯 测试目标与范围

### 1. 缓存隔离验证
确保 `ask(update_state=False)` 和 `ask_choice()` 不污染视频流缓存。

**测试点：**
- KV Cache 快照/恢复前后签名一致性
- 模型 `stream_state` (last_cache_position, rope_deltas) 正确保存/恢复
- QA 后能继续追加新帧

**相关测试：** test_step1, test_step4

---

### 2. KVCacheManager 逻辑完整性
验证缓存管理器的所有方法正确性（不依赖实际模型）。

**测试点：**
- `snapshot()`/`restore()` - 深拷贝 + 状态保护
- `clone()` - 独立缓存副本
- `discard_snapshot()` - 快照丢弃
- `build_full_attention_mask()` - Attention mask 拼接
- `clear()` - 内存释放
- `get_seq_length()` - 序列长度查询

**相关测试：** test_step2

---

### 3. Prompt 处理鲁棒性
验证 `_extract_vision_segment()` 对不同 chat template 结构的处理。

**测试点：**
- 正常 vision_start/end token 包裹
- 缺失 vision token 的 fallback
- 多段 vision 片段
- 空 prompt

**相关测试：** test_step3

---

### 4. 端到端时序理解
验证模型能正确理解跨帧时序关系。

**测试点：**
- 单帧 image 模式 + 多帧 video chunk 模式
- 不同颜色/形状的帧序列
- "最后出现的是什么" 类型的时序问答

**相关测试：** test_step5

---

### 5. 🔥 流式 vs 原生离线推理对比（核心需求）

**测试场景：**
使用真实视频 `/root/autodl-tmp/temporal_encoding/1.mp4` (~3s, 30fps)

**流式模式：**
1. 按 4 帧 chunk 逐步编码至 2 秒
2. 暂停后回答问题
3. 记录：编码时间、TTFT、总 QA 延迟、VRAM 使用、Cache 大小

**原生模式：**
1. 一次性加载完整视频（同样前 2 秒）+ 问题
2. Prefill + Decode 生成答案
3. 记录：Prefill 时间（TTFT）、总延迟、VRAM 使用

**对比指标：**
- **响应时间**：TTFT、总延迟
- **内存效率**：VRAM allocated/reserved、Cache memory
- **答案质量**：流式 vs 离线答案一致性
- **适用场景分析**

**相关测试：** test_step6 ⭐

---

### 6. Chunk 规模性能测试
对比不同帧数 chunk 的编码性能。

**测试 Chunk 大小：**
- 2 帧 (T=1): 最低延迟，最小 cache 增长
- 4 帧 (T=2): 推荐配置，延迟/质量均衡
- 6 帧 (T=3): 更高吞吐
- 3 帧: 非 temporal_patch_size 倍数，触发帧填充

**测量指标：**
- 编码延迟
- Cache 序列长度增长
- Cache 内存占用

**相关测试：** test_step7

---

## 🚀 运行指南

### 环境要求

**基础环境：**
```bash
Python >= 3.8
torch >= 2.0
transformers >= 4.37.0
Pillow
opencv-python (test_step6 需要)
```

**模型与数据：**
- 模型路径：`/root/autodl-tmp/Qwen/Qwen2___5-VL-3B-Instruct`
- 测试视频：`/root/autodl-tmp/temporal_encoding/1.mp4`
- GPU：推荐 >= 8GB VRAM

**环境变量（可选）：**
```bash
export QWEN_MODEL_PATH="/your/model/path"
export VIDEO_PATH="/your/video/path"
```

---

### 测试分类运行

#### 1️⃣ CPU 逻辑测试（无需 GPU）
```bash
cd temporal_encoding
python test_step2_cache_logic.py
python test_step3_prompt.py
```

#### 2️⃣ GPU 功能测试（需要模型）
```bash
python test_step1_cache.py
python test_step4_choice_cache.py
python test_step5_e2e.py
```

#### 3️⃣ 🔥 核心对比测试（需要模型 + 视频）
```bash
python test_step6_stream_vs_native.py
```

#### 4️⃣ 性能测试
```bash
python test_step7_multi_chunk.py
```

#### 5️⃣ 完整测试套件
```bash
# CPU 测试
python test_step2_cache_logic.py && python test_step3_prompt.py

# GPU 测试
python test_step1_cache.py && \
python test_step4_choice_cache.py && \
python test_step5_e2e.py && \
python test_step6_stream_vs_native.py && \
python test_step7_multi_chunk.py
```

---

## 📊 预期输出示例

### test_step6_stream_vs_native.py 输出结构

```
======================================================================
📹 STREAMING MODE TEST
======================================================================

[1] Loading video frames (first 2.0s)...
    ✅ Loaded 60 frames (fps=30.00, total=3.00s)

[2] Initializing streaming engine...
    ✅ VRAM after model load: {'allocated': 2845.12, 'reserved': 3072.00}

[3] Streaming encoding (4-frame chunks)...
    Chunk 1/15: Chunk 0 encoded (4 frame(s), cache_len=1234)
    ...
    ✅ Encoding completed in 2.456s

[4] Asking question: 'Describe what is happening in this video.'
    ✅ Answer: A person is walking in a park...
    TTFT: 0.123s
    Total QA latency: 1.234s

======================================================================
🎬 NATIVE OFFLINE MODE TEST
======================================================================
...

======================================================================
📊 COMPARISON REPORT
======================================================================

[Encoding Performance]
  Streaming encoding time: 2.456s
  Native prefill time:     1.234s

[QA Performance]
  Streaming TTFT:          0.123s
  Native TTFT:             1.234s
  ...
```

---

## ✅ 通过标准

### 所有测试
- 无 Python 语法错误
- 无运行时异常（除预期的 skip）
- 关键断言通过

### test_step1 & test_step4（缓存隔离）
- QA 前后 cache 签名一致
- `stream_state` 正确恢复
- QA 后能继续追加帧

### test_step6（核心对比）
- 流式与原生都能生成合理答案
- VRAM 记录完整
- TTFT 和总延迟数值合理
- 对比报告清晰展示性能差异

---

## 🐛 故障排查

### 1. 模型路径错误
**现象：** `Model not found: /root/autodl-tmp/...`

**解决：**
```bash
export QWEN_MODEL_PATH="/your/actual/model/path"
```

### 2. 视频加载失败（test_step6）
**现象：** `Cannot open video: ...`

**解决：**
```bash
export VIDEO_PATH="/your/video/path"
# 或修改脚本中的 VIDEO_PATH 常量
```

### 3. CUDA OOM
**现象：** `CUDA out of memory`

**解决：**
- 减小 test_step6/7 中的 CHUNK_SIZE
- 减小 max_new_tokens
- 使用更小的模型或 INT8 量化

### 4. transformers 版本不兼容
**现象：** `TypeError: get_rope_index() got an unexpected keyword argument`

**解决：**
```bash
pip install transformers>=4.37.0 --upgrade
```

---

## 📝 测试报告模板

运行完整测试后，可生成报告：

```markdown
# Streaming VLM Test Report

## Test Environment
- GPU: NVIDIA RTX 4090
- VRAM: 24GB
- Model: Qwen2.5-VL-3B-Instruct
- Video: 1.mp4 (3s, 30fps)

## Test Results

### Cache Isolation (Step 1, 4)
✅ PASSED - Cache and stream_state correctly protected

### E2E Understanding (Step 5)
✅ PASSED - Model correctly identifies temporal sequence

### Streaming vs Native (Step 6)
✅ PASSED
- Streaming TTFT: 0.123s (vs Native: 1.234s) → **10x faster**
- Streaming VRAM: 4.2GB (vs Native: 5.8GB) → **28% less**
- Answer quality: Comparable

### Chunk Size Comparison (Step 7)
✅ PASSED
- 2 frames: 0.045s encode, cache +256
- 4 frames: 0.078s encode, cache +512 (recommended)
- 6 frames: 0.112s encode, cache +768

## Conclusion
流式推理在低延迟场景下优势明显，适合实时交互应用。
```

---

## 🔗 相关文档

- [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) - 完整项目结构说明
- [model/stream_qwen_model.py](model/stream_qwen_model.py) - 核心流式模型实现
- [model/video_stream_inference.py](model/video_stream_inference.py) - 高层推理引擎

---

**Last Updated:** 2026-02-10  
**Test Coverage:** 7 test files, 6 major test scenarios  
**Status:** ✅ All tests implemented and documented
