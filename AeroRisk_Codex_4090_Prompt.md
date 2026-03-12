# 给 Codex 的云端 4090 提示词（详细执行版）

你现在运行在**云端 4090 环境**。本地 3050 阶段已经完成了 AeroRisk Lite 的无模型闭环。你当前的任务不是重写整套系统，而是：

- 复现本地成果
- 接入真实 `Qwen2.5-VL`
- 在 Lite 主线上完成首轮 pilot
- 产出可用于后续论文实验的第一版真实结果

---

## 0. 先明确你的边界

你当前只做：

- 验证 4090 环境
- 复现本地测试与 smoke
- 把 Pass B 从 `stub_backend` 替换成真实模型推理
- 输出 latency / VRAM / pilot metrics

你当前不做：

- HiVT / Trajectron++
- DA-CEM
- LoRA
- 大规模全量打榜
- 真 streaming 重构

如无必要，请不要重构已在本地验证通过的代码结构。

---

## 1. 开始前必须读取的文件

先完整阅读以下文件并总结，再开始修改：

### 1.1 原始设计与代码参考

1. `research_UAV_driving_risk_assessment_v5.md`
2. `aerorisk_execution_plan.md`
3. `temporal_encoding/model/stream_qwen_model.py`
4. `temporal_encoding/model/video_stream_inference.py`
5. `temporal_encoding/model/kv_cache_eviction.py`
6. `temporal_encoding/model/cache_manager.py`

### 1.2 本地阶段交付物

1. `LOCAL_TO_4090_HANDOFF.md`
2. `LOCAL_AUDIT.md`
3. `outputs/local_smoke/summary.md`
4. `outputs/local_smoke/guard_log.jsonl`
5. `outputs/local_smoke/shp_stub_log.jsonl`
6. `outputs/local_smoke/fused_log.jsonl`
7. `outputs/local_smoke/rendered_examples/*.png`

你必须先复述你理解到的：

- 本地已完成哪些模块
- 真实模型应该接到哪里
- 哪些文件是本阶段的关键改动点

---

## 2. 你必须遵守的实现原则

1. 保持 **Lite 路线**：只做 SinD + B0-CTRA + 单图 SoM/PiP + SHP + DST/WSSD + offline replay。
2. `guard_hard_floor = 4` 保持不变。
3. `exemption_max_levels = 1` 保持不变。
4. 禁止 `4 -> 2`、`4 -> 1`、`3 -> 1`。
5. 若真实 Qwen 输出与 schema 不匹配，必须 fallback，而不是让系统崩溃。
6. 当前优先做 **Pass B 的真实单图推理**，不是做真 streaming 主循环。
7. 若 `VideoStreamingInference` 很容易接入，可以复用；若接入成本高，则先做普通单图推理版本，只要接口保持清晰即可。
8. 所有结果必须可追溯到 PNG、raw text、parsed SHP、trace。

---

## 3. 你的阶段目标

你在 4090 上必须完成：

- 环境验证
- 全部本地测试复现
- 真实 `Qwen2.5-VL` 单图推理打通
- `pass_b_vlm_log.py` 生成真实 SHP 日志
- 20–50 个触发事件的 pilot
- 指标与 qualitative 输出

---

## 4. 具体实施步骤

### Step 0：环境与依赖审计

先检查：

- `nvidia-smi`
- CUDA / cuDNN / PyTorch GPU 可用性
- transformers / accelerate / safetensors / opencv / Pillow 等依赖
- 模型权重路径或 HuggingFace 加载方式

输出：

- `SERVER_4090_AUDIT.md`

### Step 1：先复现本地结果，不要急着接模型

要求先完整运行：

- 所有 `tests/`
- 本地 smoke 命令

如果与本地输出不一致，先修一致性问题，再继续。

### Step 2：定位真实模型接入点

优先检查并决定是否复用：

- `vlm/vlm_branch_lite.py`
- `replay/pass_b_vlm_log.py`
- `vlm/sseh.py`
- `temporal_encoding/model/video_stream_inference.py`
- `temporal_encoding/model/stream_qwen_model.py`

你的目标是：

- 输入：一张由 `VisualPromptRenderer` 生成的复合图
- 输出：原始文本 + 合法 JSON + `SHPCard`

### Step 3：实现真实 Pass B

优先实现以下最小闭环：

1. 读取触发帧对应的渲染 PNG
2. 用真实 `Qwen2.5-VL` 做单图单次前向
3. 让模型输出结构化 JSON 或接近 JSON 的文本
4. 用 `SSEH` 做 schema 校验 / fallback
5. 记录：
   - `t_obs_end`
   - `latency_s`
   - `t_emit`
   - `raw_text`
   - `schema_valid`
   - `parsed_shp`

如果普通单图推理先跑通，而 streaming 版更复杂，请优先保留普通单图方案，不要为了“更先进”而拖慢落地。

### Step 4：小规模 pilot

在 4090 上先做 20–50 个触发事件，而不是全量跑。

事件优先级：

1. 左转让行
2. 低速排队释放
3. 接近但最终安全通过
4. 明显真实高危冲突

每个事件都必须至少保存：

- `rendered.png`
- `guard.json`
- `raw_vlm.txt`
- `parsed_shp.json`
- `fusion_trace.json`

### Step 5：统计首轮指标

至少计算并输出：

- `SSEH_valid_rate`
- `mean_latency_s`
- `p50/p95_latency_s`
- `peak_vram_gb`
- `FARR`
- `GRP`
- `WSI`
- `guard_only_fallback_ratio`
- `hard_floor_violations`

输出目录：

```text
outputs/pilot_4090/
  metrics.json
  case_table.csv
  qualitative_examples/
  summary.md
```

### Step 6：错误分析

如果 pilot 指标不好，不要直接大改系统，而是先定位是以下哪一类问题：

- 渲染问题：SoM / PiP 布局导致模型误读
- Schema 问题：输出不稳定，SSEH 回退太多
- 语义问题：模型给出的 priority / yielding 语义不可靠
- 融合问题：折扣或验证条件过松/过紧
- 数据问题：事件切片或轨迹估计存在误差

然后写入：

- `outputs/pilot_4090/error_analysis.md`

---

## 5. 你需要优先修改的文件

在大多数情况下，你应该优先查看和修改：

1. `vlm/vlm_branch_lite.py`
2. `replay/pass_b_vlm_log.py`
3. `vlm/sseh.py`
4. `aerorisk/config.py`
5. 必要时再看 `video_stream_inference.py`

不要一上来重构 `kv_cache_eviction.py` 或整套 streaming 框架，除非它已经成为当前 Lite 路线的明确瓶颈。

---

## 6. 你的验收标准

只有全部满足，才算 4090 阶段完成：

- 真实 Qwen2.5-VL 单图推理可用
- 所有本地测试在 4090 上也通过
- `SSEH_valid_rate >= 0.90`
- `hard_floor_violations == 0`
- `GRP <= 0.01`
- `FARR > 0`
- 至少有 1 个“让行压制误报”的成功案例
- 至少有 1 个“高危 hard-floor 保底”的成功案例
- 已输出完整 metrics、case_table、qualitative examples 和 error analysis

---

## 7. 最终输出要求

完成后，你必须明确列出：

- 新增文件
- 修改文件
- 真实模型是如何接进去的
- 目前保留的限制
- 下一步若要扩展到 HiVT / Trajectron++ 应从哪里开始

记住：你现在的工作重点是把 **Lite 真模型 pilot** 做扎实，而不是把论文里所有远期模块一次性全实现。
