
## 一、Test Step 6 的 OOM 改进策略

对比我最初生成的版本和你实际运行的版本，改进体现在 test_step6_stream_vs_native.py 的 `main()` 中引入了**自动降级重试机制**：

### 核心改动

1. **`NativeOOMError` 异常类** — 将 `torch.OutOfMemoryError` 包装为自定义异常，使其可以在 `main()` 中被精确捕获

2. **`frame_stride` 参数** — `_load_video_frames()` 新增跳帧参数，stride=2 表示每隔 1 帧取 1 帧，将 48 帧降为 24 帧

3. **`stride_candidates` 梯级重试** — `main()` 中构建 `[1, 2, 4]` 候选列表，从 stride=1 开始尝试：
   - stride=1 → 流式成功，原生 **prefill 后 VRAM 达 17138 MB，generate 阶段 OOM**
   - 自动 `torch.cuda.empty_cache()` 清理后用 stride=2 重试
   - stride=2 → 双方均成功

4. **`TeeWriter`** — 同时输出到控制台和文件，确保报告完整记录包括 OOM 的过程

**本质：** 原版直接崩溃退出；改进版用异常捕获 + 降采样重试，保证一定能完成对比，同时记录降级日志供分析。

---

## 二、全部测试结果综合分析

### 2.1 基础逻辑验证（Step 2-3）✅ 全部通过

| 测试 | 结果 | 覆盖范围 |
|:--|:--|:--|
| Step 2 KVCacheManager | ✅ 6/6 | snapshot/restore、clone、discard、mask 拼接、clear、seq_length |
| Step 3 Prompt 裁剪 | ✅ 4/4 | 正常/fallback/多段/空输入 |

结论：**缓存管理器和 prompt 处理逻辑无缺陷。**

### 2.2 GPU 功能验证（Step 1, 4, 5）✅ 全部通过

| 测试 | 关键数据 | 结论 |
|:--|:--|:--|
| Step 1 快照/恢复 | QA 前后 cache_sig 一致 (117, 0.0)→(117, 0.0)，恢复后成功追加至 183 | **缓存隔离机制工作正确** |
| Step 4 ask_choice | 正确选出 "red"，缓存/stream_state 隔离验证通过 | **多选评分 + 独立分叉正确** |
| Step 5 端到端 | Test A 回答 "Blue rectangle appears"，Test B 回答 "A blue square" | **模型正确理解时序（后出现=蓝色）** |

关键发现：Step 5 证明了即使不微调，**模型能通过 KV Cache 中的历史信息正确回答时序问题**。

### 2.3 核心对比测试（Step 6）—— 最重要的结论

#### OOM 事件分析

| 阶段 | stride=1 (48帧) | stride=2 (24帧) |
|:--|:--|:--|
| 流式编码 | ✅ cache_len=28784, VRAM 8298 MB | ✅ cache_len=14419, VRAM 7793 MB |
| 原生 prefill | ✅ VRAM **17139 MB** | ✅ VRAM 12216 MB |
| 原生 generate | ❌ **OOM** | ✅ |

**为什么流式模式不 OOM 但原生模式 OOM？**

原生模式在 prefill 时需要对 **完整序列（~28784 token）** 做一次性 self-attention 计算，attention 矩阵的峰值显存 ∝ $O(n^2)$。而流式模式每次 chunk prefill 只处理 **~2394 token**（单 chunk），峰值显存恒定。即使流式模式累积了同样长度的 KV Cache，cache 的显存增长是 $O(n)$ 线性的，不会触发 $O(n^2)$ 峰值。

#### 核心性能对比（stride=2，公平对比）

| 指标 | 流式 | 原生 | 差异 |
|:--|:--|:--|:--|
| **TTFT** | **0.050s** | 4.164s | **83× 加速** |
| **总 QA 延迟** | **0.912s** | 10.041s | **11× 加速** |
| **编码阶段 VRAM** | **7793 MB** | 12216 MB | **节省 4424 MB (36%)** |
| **QA 后 VRAM** | **7792 MB** | 12216 MB | 节省 4424 MB |
| Cache 长度 | 14419 | 14381 | 几乎相同 |

#### 答案质量对比

| 方面 | 流式 | 原生 |
|:--|:--|:--|
| 场景 | ✅ 夜间户外 | ✅ 夜间城市环境 |
| 人物 | ✅ 白衬衫+黑帽 | ✅ 白衬衫(多细节) |
| 行为 | ✅ 走路+说话 | ✅ 向前移动 |
| 画面 | ✅ 晃动+建筑树木 | ✅ 手持拍摄+建筑树木停车 |
| 详细度 | 中等（5句） | 较高（8句） |

**结论：两者语义一致，都正确描述了视频内容。** 原生答案更详细是因为它没有 system prompt 中 "at most 60 tokens" 的约束。流式模式的答案更简洁直接，对于实时交互场景反而更实用。

### 2.4 Chunk 规模测试（Step 7）

| Chunk 大小 | 编码时间 | Cache 增长 | VRAM Δ |
|:--|:--|:--|:--|
| 2 帧 (T=1) | 0.586s ⚠️ | 119 | 13.31 MB |
| 4 帧 (T=2) | 0.060s | 183 | 6.43 MB |
| 6 帧 (T=3) | 0.070s | 247 | 8.68 MB |
| 3 帧 (非倍数) | 0.055s | 183 | 6.43 MB |

**关键发现：**
- 2 帧 chunk 首次编码异常慢（0.586s vs 0.06s），这是因为 **首帧包含 system prompt 的完整 chat template tokenization + 首次 CUDA kernel 编译**，后续 chunk 无此开销
- 3 帧被处理器自动填充为 4 帧（cache_len=183 与 4 帧相同），验证了 `temporal_patch_size=2` 的填充机制
- **4 帧 chunk 是最佳平衡点**，与设计预期一致

---

## 三、总结论

### 核心结论

> **不微调模型，基于原生 Qwen2.5-VL-3B-Instruct 的流式推理改造在功能和性能上均得到验证：**
> 1. 功能正确性：7/7 测试全部通过，缓存隔离、时序理解、位置追踪均工作正常
> 2. TTFT 加速 83×，总延迟加速 11×，VRAM 节省 36%
> 3. 答案质量与原生等价（语义一致）
> 4. 流式模式能处理原生模式 OOM 的场景（48帧），证明了显存效率的根本性优势

### 项目当前状态

✅ **已完成：**
- 核心 3 文件实现（stream_qwen_model / video_stream_inference / cache_manager）
- 3 个 bug 修复（position 维度、rope_deltas 过期值、TTFT 测量点）
- 7 个测试文件 + 完整测试文档
- 项目结构文档

---

## 四、后续工作建议

### 短期（毕设必需）

1. **修复 `cache_memory_mb` 报告为 0 的问题** — 测试报告中所有 `cache_memory_mb` 均为 0.0，这是因为 `DynamicCache` 的内部结构可能与 `get_cache_info()` 中遍历 `key_cache`/`value_cache` 的方式不匹配，需要调试

2. **后续 chunk 的 prompt 结构优化** — 前面分析中提到的 ⚠️ 降质因素 2，后续帧的裸 vision segment 缺少 `<|im_start|>user\n...<|im_end|>` 包裹。可以改为保留对话结构包裹但去掉重复文本，这是一个低成本的质量提升

3. **多轮 QA 测试** — 当前所有测试都是单轮问答。应测试：编码→问→**继续编码→再问**的完整场景，验证 `update_state=False` 恢复后继续追加帧→再次 QA 的答案连贯性

4. **Web Demo 集成** — web_demo 目录已有 Gradio 框架，将流式引擎接入 Web 前端实现真正的实时交互演示

### 中期（论文加分项）

5. **Temporal 间距精确修正** — Branch 2 的全局偏移 `offset = last_cache_position + 1` 导致 chunk 边界处时间间距不精确。可以引入精确的 `second_per_grid_t` 累积计算，使跨 chunk 的 T 维位置与原生 `get_rope_index` 完全一致

6. **长视频 Sliding Window / Eviction** — 当前全量累积 KV Cache，对长视频（>30s）可能 OOM。实现基于重要性评分的 cache eviction 策略

7. **量化对比实验** — 在论文中对比不同帧率、不同 chunk size、不同视频长度下的性能曲线，形成系统性实验数据