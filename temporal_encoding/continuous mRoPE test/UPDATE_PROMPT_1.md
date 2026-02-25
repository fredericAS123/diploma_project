# 🔧 Update Prompt 1: 短期优化四项改进

本文档指导测试机上的 Copilot 运行和验证 Update 1 的全部修改。

---

## 📋 修改摘要

### 修改 1: 修复 `cache_memory_gb` 报告为 0

**问题:** `get_cache_info()` 返回的 `cache_memory_gb` 始终为 0.0。  
**根因:** transformers ≥ 4.50 的 `DynamicCache` 不再有 `key_cache`/`value_cache` 属性，改用 `cache.layers[i].key_state`/`.value_state`。旧代码访问不存在的属性被 `except Exception: pass` 静默吞掉。  
**修复:** 在 `video_stream_inference.py` 中新增 `_measure_cache_bytes()` 静态方法，按优先级尝试 3 种策略：
1. 新版 `cache.layers[].key_state / value_state`
2. 旧版 `cache.key_cache / value_cache`
3. 通用回退 `cache[i]` 逐层提取

**涉及文件:** `temporal_encoding/model/video_stream_inference.py`

---

### 修改 2: 后续 chunk prompt 结构优化

**问题:** 后续帧只追加裸 `<|vision_start|>...<|vision_end|>` token，缺少对话结构标记，导致 token 分布与训练时不同（OOD）。  
**优化:** 新增 `_extract_user_vision_turn()` 方法，后续 chunk 现在包裹为：
```
<|im_start|>user\n<|vision_start|>...<|vision_end|><|im_end|>\n
```
保留对话结构但不重复 system prompt 和文本内容。

**涉及文件:** `temporal_encoding/model/video_stream_inference.py`, `temporal_encoding/test_step3_prompt.py`

---

### 修改 3: 多轮 QA 测试

**新增文件:** `temporal_encoding/test_step8_multi_round_qa.py`  
**测试场景:** 
- Phase 1: 编码 2 帧红色圆形 → 问颜色
- Phase 2: 继续编码 2 帧蓝色方块 → 问所有形状
- Phase 3: 继续编码 2 帧绿色三角 → 问最后出现的形状

**验证点:**
- 每轮 QA 后 cache 正确恢复（`update_state=False`）
- 继续编码后 cache 正确增长
- `cache_memory_gb` 单调递增（验证修复 1）
- 答案语义正确性

---

### 修改 4: Web Demo Gradio 集成

**重写文件:**
- `web_demo/Qwen_inference.py` — 适配新 API（移除 `manual_time`，新增 `process_video_chunk` / `ask_choice` / `get_cache_info`，`**kwargs` 兼容旧接口）
- `web_demo/webui_gradio.py` — 适配新引擎（chunk 编码模式、KV Cache 状态显示、fps 计算修复）
- `web_demo/main.py` — 入口更新

**新增测试:** `temporal_encoding/test_step9_web_demo.py` — 验证 Web Demo 后端 API 完整性

---

## 🧪 测试运行指南

### 环境准备

```bash
# 确保在正确的 conda 环境中
conda activate videollm  # 或你的环境名

# 确保依赖已安装
pip install torch transformers accelerate Pillow opencv-python
```

### Step 1: CPU 测试（无需 GPU）

```bash
cd /root/autodl-tmp/diploma_project/temporal_encoding

# 测试 prompt 裁剪逻辑（含新增的 _extract_user_vision_turn 测试）
python test_step3_prompt.py

# 测试 KVCacheManager 纯逻辑
python test_step2_cache_logic.py
```

**期望结果:** 
- Step 3: 8/8 cases passed（原 4 个 + 新增 4 个 user vision turn 测试）
- Step 2: 6/6 functions verified

### Step 2: GPU 功能测试

```bash
# 多轮 QA 测试（新增）
python test_step8_multi_round_qa.py

# 验证点：
#   - cache_memory_gb > 0（不再为 0.0）
#   - 3 轮 QA 后 cache 均正确恢复
#   - cache_memory_gb 单调递增
#   - Phase 1 提到 "red"
#   - Phase 3 提到 "green" 或 "triangle"
```

### Step 3: Web Demo 后端集成测试（新增）

```bash
# 测试 QwenInferenceWrapper 全部 API
python test_step9_web_demo.py

# 验证点：
#   - process_frame() / process_video_chunk() 正常
#   - ask_question() 返回有效答案和 metrics
#   - cache_memory_gb > 0
#   - 旧参数 manual_time 被 **kwargs 静默忽略
#   - reset() 完全清理状态
#   - chunk→ask→chunk→ask 完整流程
```

### Step 4: 回归测试

```bash
# 运行原有测试确认没有回归
python test_step1_cache.py
python test_step4_choice_cache.py
python test_step5_e2e.py
python test_step7_multi_chunk.py

# 可选：核心对比测试（需要视频文件）
python test_step6_stream_vs_native.py
```

### Step 5: Web Demo 启动测试（可选，需要端口转发）

```bash
cd /root/autodl-tmp/diploma_project/web_demo
python main.py

# 在浏览器访问 http://localhost:6006
# 功能验证：
#   1. 上传视频
#   2. 设置 chunk_size=4
#   3. 点击 Start → 观察 KV Cache Status 面板
#   4. 输入问题 → 观察自动暂停 + 回答
```

---

## 🔍 已知注意事项

### 1. `_measure_cache_bytes()` 的 3 策略兼容

如果 transformers 版本较旧（< 4.50），策略 1（`cache.layers`）不会匹配，会自动降级到策略 2（`key_cache/value_cache`）。如果你的环境既不是新版也不是旧版，策略 3（`__getitem__`）作为最终回退。

**调试方法：** 如果 `cache_memory_gb` 仍为 0，在 `_measure_cache_bytes` 方法的 `except Exception:` 后添加 `traceback.print_exc()` 查看具体错误。

### 2. prompt 结构优化的影响

后续 chunk 从裸 vision token 变为 user turn 包裹，**会额外增加约 4 个 token/chunk**（`<|im_start|>user\n` + `<|im_end|>\n`）。这是微小的开销，但改善了 token 分布的一致性。

如果测试发现答案质量变化，可对比新旧版本：
```python
# 旧版行为（如需临时回退）
# 将 _build_frame_prompt 中的 _extract_user_vision_turn 换回 _extract_vision_segment
```

### 3. Web Demo 的 chunk_size 参数

- `chunk_size=1`: 逐帧 image 模式（兼容，但效率较低）
- `chunk_size=2`: T=1，最低延迟
- `chunk_size=4`: T=2，**推荐平衡点**
- `chunk_size=6/8`: T=3/4，高吞吐

建议在 Gradio 界面中使用 chunk_size=4。

---

## 📝 迭代指南

如果某个测试失败：

1. **Step 3 失败（prompt 裁剪）:** 检查 `_extract_user_vision_turn()` 的返回格式，确认 `<|im_start|>user\n` 和 `<|im_end|>\n` 的拼接顺序
2. **Step 8 cache_memory_gb=0:** 在 `_measure_cache_bytes()` 中添加调试打印，检查 `DynamicCache` 的实际属性：
   ```python
   print(f"DEBUG: cache type={type(cache)}, dir={[a for a in dir(cache) if not a.startswith('_')]}")
   ```
3. **Step 9 import 失败:** 确认 `sys.path` 包含 `temporal_encoding` 目录和 `web_demo` 目录
4. **Step 8/9 语义不匹配:** 合成图像的形状识别依赖模型能力，`⚠️` 警告不代表测试失败

---

## 📊 期望的完整测试输出

```
Step 2:  ✅ 6/6 cache logic tests
Step 3:  ✅ 8/8 prompt trimming tests (4 original + 4 new)
Step 1:  ✅ cache snapshot/restore
Step 4:  ✅ ask_choice cache isolation
Step 5:  ✅ E2E temporal understanding
Step 7:  ✅ multi-chunk performance
Step 8:  ✅ multi-round QA (cache_memory_gb > 0, cache growth, semantic correctness)
Step 9:  ✅ web demo backend (all 7 API tests)
Step 6:  ✅ streaming vs native (optional, needs video)
Step 10: ✅ max frame capacity (native resolution, real video, OOM/EOF detection)
```

---

## 🚀 Step 10: 最大编码帧数容量测试（原生分辨率/真实视频）

**测试文件:** `temporal_encoding/test_step10_max_frames.py`

### 实验目标

使用真实视频在**原生分辨率**下测试流式引擎的**最大编码帧数容量**，提供 RTX 4090（24GB VRAM）上的实际上限基准。

### 测试分辨率

- **原生分辨率** — 以视频原始尺寸进行编码

### 测试策略

1. **渐进式编码:** 从 10 chunks 开始，每次增加 10 chunks，直到 OOM 或视频帧耗尽（EOF）
2. **固定 chunk size:** 使用 4 帧/chunk（T=2，推荐配置）
3. **真实视频帧:** 使用 `cv2` 流式读帧，末尾不足 4 帧时自动填充
4. **记录指标:**
   - 最大成功编码帧数（真实帧数）
   - 总编码时间
   - KV Cache 序列长度
   - KV Cache 内存占用 (GB)
   - VRAM 峰值（allocated / reserved）
   - EOF / 填充帧统计

### 运行方法

```bash
cd /root/autodl-tmp/diploma_project/temporal_encoding

# 运行最大容量测试（需要 GPU + 模型权重 + 视频）
python test_step10_max_frames.py

# 可选：指定视频路径或抽帧步长
export VIDEO_PATH="/root/autodl-tmp/diploma_project/temporal_encoding/202208312002.mp4"
export FRAME_STRIDE=1

# 查看报告
cat test_step10_max_frames_report.txt
```

### 期望输出

**报告示例:**
```
═══════════════════════════════════════════════════════════
📊 MAXIMUM FRAME CAPACITY SUMMARY
═══════════════════════════════════════════════════════════

Resolution           Max Frames   Encode Time     Cache Len    Cache Mem    VRAM Peak (A/R)
──────────────────────────────────────────────────────────────────────────────────────────
1920×1080 (native)   480          38.452s         28500        1.8432 GB    14.20/15.30 GB

[Key Findings]
  • Highest capacity: 1920×1080 (native) with 480 frames
  • 1920×1080 (native): 12.49 frames/sec encoding throughput

[Memory Efficiency]
  • 1920×1080 (native): 0.7421 GB per megapixel

[Recommendations]
  • For real-time streaming (24 fps target):
    - 1920×1080 (native): Can handle up to 20.0s video at 24fps

  • Model baseline VRAM: 6.42 GB
  • Chunk size used: 4 frames
  • Frame stride: 1
  • Recommendation: Use smaller resolution for longer videos if needed
```

### 验证点

1. ✅ **渐进式 OOM/EOF 检测:** 能找到最后成功的编码帧数（OOM 或 EOF 终止）
2. ✅ **显存单调递增:** 帧数越多，VRAM 占用越高
3. ✅ **cache_memory_gb > 0:** 验证缓存计量有效
4. ✅ **VRAM 峰值记录:** 同时给出 allocated / reserved 峰值
5. ✅ **编码吞吐量:** 计算 frames/sec 提供性能参考

### 生产建议

- **原生分辨率上限:** 以本测试结果为上限基准
- **更长视频:** 建议降低分辨率或提高 stride
- **实时流:** 监控 VRAM，必要时做分辨率/帧率自适应

---

**Created:** 2026-02-11  
**Updated:** 2026-02-11 (added Step 10 max capacity test)  
**Covers:** Short-term optimizations 1-4 + Step 10 capacity benchmark
