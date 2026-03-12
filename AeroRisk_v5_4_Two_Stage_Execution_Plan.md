# AeroRisk v5.4 两阶段执行方案（先本地 3050，再云端 4090）

> 用途：把当前 AeroRisk Lite 路线改造成真正可执行的两阶段工程方案。  
> 核心原则：**先在本地 3050 完成“无大模型依赖的完整闭环”**，再把已验证代码迁移到 **4090** 上接入真实 Qwen2.5-VL 并开展 pilot 实验。  
> 本文件优先级：当它与旧的 `aerorisk_execution_plan.md` 或 `research_UAV_driving_risk_assessment_v5.md` 存在实现层冲突时，**以本文件为准**；研究叙事仍以原研究文档为准。

---

## 0. 为什么必须拆成两阶段

你的原始研究文档和执行计划已经明确区分了硬件角色：

- **本地 3050 (8GB)**：代码编写、单元测试、CPU 逻辑验证
- **云 4090 (24GB)**：主实验、流式推理、参数消融

这与 AeroRisk 的架构本身也是一致的：系统强调**时间戳严格对齐的解耦仿真**、`SyncAligner` 的严格因果对齐、以及 Guard/Base 与 VLM 旁路的异步解耦。因此，把实现过程拆成“本地先完成 deterministic pipeline，再上 4090 接入真实 VLM”不仅现实，而且最符合你的论文叙事。  

---

## 1. 本次实现只做什么，不做什么

### 1.1 当前只做 Lite 主线

当前只实现下面这一条最短闭环：

- 数据：**SinD**
- 物理底座：**B0-CTRA / Guard-only**
- 视觉分支：**SoM + PiP 单帧复合图**
- 结构化输出：**SSEH / SHPCard**
- 融合：**WSSD + Shafer 折扣 + DST + Physical Verifier + TransparentFuserLite**
- 调度方式：**offline 3-pass replay**

### 1.2 当前明确不做

以下内容全部推迟到 Lite 在 4090 跑通后：

- HiVT
- Trajectron++
- DA-CEM / 复杂 KV eviction 扩展
- 真 streaming 问答
- 长视频因果问答
- DRIFT / VisDrone 全流程
- LoRA 微调
- 大规模打榜与跨城市泛化

---

## 2. 你需要让 Codex 遵守的顶层规则

1. **严格因果**：`SyncAligner` 只能使用 `<= T` 的观测，默认 `zoh`，禁止双边插值。
2. **先 stub 再真模型**：本地 3050 阶段只允许 stub / mock SHP，不要求真实 Qwen 推理。
3. **4090 阶段才接入真实 Qwen2.5-VL**。
4. **Guard hard floor 保留**：`guard_hard_floor = 4`。
5. **Lite 阶段豁免上限**：`exemption_max_levels = 1`。
6. **禁止 4→2、4→1、3→1**。
7. **VLM 不可直接把高危硬对冲成安全**；语义豁免必须先表现为对 Guard 证据的折扣与不确定性重分配。
8. **物理续验优先**：只有 `r_guard <= 3` 且让行/受控释放等语义被近期真实物理历史支持时，才允许 `3 -> 2`。
9. **先做离线 3-pass，不做真异步并发**。

---

## 3. 两阶段总览

## 阶段 A：本地 3050（目标：把“无真实 VLM 的完整工程闭环”做对）

### A.1 本阶段目标

在本地 3050 上完成以下事情：

- 环境审计与项目骨架
- `SyncAligner` + `SinDAdapter`
- `TrackState / FrameData / RiskMap / SHPCard` 等核心 dataclass
- `CTRAPredictor` + `SSMCalculator` + `GuardBranch` + `Sentinel`
- `VisualPromptRenderer`（SoM + PiP + Semantic Canvas）
- `SSEH` 的 schema、解析与 fallback 逻辑
- `WSSD / DST / ShaferDiscounting / PhysicalVerifier / TransparentFuserLite`
- `OfflineReplayEngine` 三遍回放
- **stub SHP** 驱动的端到端跑通
- synthetic + 小规模 SinD smoke test

### A.2 本阶段不要求

本地 3050 阶段**不要求**：

- 真实 Qwen2.5-VL 推理
- 高吞吐 benchmark
- 大规模实验
- HiVT / Trajectron++

### A.3 本地阶段必须读取的文件（Codex 审计顺序）

Codex 在本地开始编码前，必须先完整审计以下文件；若文件缺失，就在仓库内查找等价路径：

1. `research_UAV_driving_risk_assessment_v5.md`
2. `aerorisk_execution_plan.md`
3. `temporal_encoding/model/stream_qwen_model.py`
4. `temporal_encoding/model/video_stream_inference.py`
5. `temporal_encoding/model/kv_cache_eviction.py`
6. `temporal_encoding/model/cache_manager.py`
7. `requirements.txt` / `environment.yml` / `pyproject.toml` / `setup.py`（若存在）
8. 现有 `tests/`、`configs/`、`scripts/`、`README*`

### A.4 本地阶段建议执行顺序

#### A0. 仓库审计

- 列出目录树
- 识别已有环境文件与测试框架
- 识别是否已有 `aerorisk/` 包或等价目录
- 形成 `LOCAL_AUDIT.md`

#### A1. 项目骨架与配置

创建并补齐：

```text
aerorisk/
  config.py
  data/
  guard/
  vlm/
  fusion/
  replay/
  evaluation/
  tests/
```

要求先实现最小可运行 dataclass 与配置对象：

- `AeroRiskConfig`
- `TrackState`
- `FrameData`
- `RiskItem`
- `RiskMap`
- `SHPCard`
- `FusionResult`

#### A2. 数据层

实现：

- `data/base_adapter.py`
- `data/sync_aligner.py`
- `data/sind_adapter.py`

要求：

- 原生时间轴与工作时间轴同时保留
- `working_fps = 10.0`
- 默认 `interp_mode = "zoh"`
- `heading`、`yaw_rate`、`ax/ay` 都要能从轨迹计算出来
- 稀疏输入下仍不能读取未来点

#### A3. Guard 物理底座

实现：

- `guard/ctra_model.py`
- `guard/ssm_calculator.py`
- `guard/guard_branch.py`
- `guard/sentinel.py`

要求：

- 先平滑 `v` / `heading`，再求导得到 `a` / `omega`
- Guard 输出标准化 `RiskMap`
- 至少支持 TTC / DRAC
- `GuardBranch` 能对同一帧产出：
  - 总体 `r_guard`
  - 成对高危对象的 `RiskMap`
  - Sentinel 候选目标

#### A4. 视觉提示与结构化输出（本地只做无模型版本）

实现：

- `vlm/visual_prompt_renderer.py`
- `vlm/shp.py`
- `vlm/sseh.py`
- `vlm/vlm_branch_lite.py`

要求：

- `VisualPromptRenderer` 输出 **单张** 复合图，不允许多图
- PiP 必须绘制在底部 `Semantic Canvas` 中，不能覆盖主图
- `SSEH` 负责 schema、校验、fallback
- `VLMBranchLite` 在本地默认走 `stub_backend`
- stub 输入是渲染图或 metadata，输出固定格式 JSON，再转为 `SHPCard`

#### A5. 融合与续验

实现：

- `fusion/dst_combiner.py`
- `fusion/discounting.py`
- `fusion/wssd.py`
- `fusion/physical_verifier.py`
- `fusion/transparent_fuser_lite.py`

要求：

- `guard_hard_floor = 4`
- `exemption_max_levels = 1`
- `PhysicalVerifier` 先支持最简单的“让行成立”检查：
  - 检查目标车辆在 `t_obs_end -> current_time` 之间是否持续减速 / 低速等待
- 输出完整 trace：
  - `mode`
  - `K`
  - `alpha_guard`
  - `age`
  - `verification_passed`
  - `final_level`

#### A6. 三遍离线回放

实现：

- `replay/pass_a_guard_log.py`
- `replay/pass_b_vlm_log.py`
- `replay/pass_c_fuse_replay.py`
- `replay/offline_replay_engine.py`

Pass A：生成 Guard 日志  
Pass B：读取触发帧并生成 stub SHP 日志  
Pass C：按 `t_emit <= T` 的规则融合重放

#### A7. 测试与本地 smoke

至少创建并跑通：

- `tests/test_phase0_env.py`
- `tests/test_phase1_data.py`
- `tests/test_phase2_guard.py`
- `tests/test_phase3_render_and_sseh.py`
- `tests/test_phase4_fuser_lite.py`
- `tests/test_phase5_offline_replay.py`

本地 smoke 要求：

- 至少 1 段 synthetic case
- 至少 1 段 SinD 真实切片
- 输出：
  - 1 份 Guard 日志
  - 1 份 stub SHP 日志
  - 1 份融合日志
  - 若干渲染 PNG

### A.5 本地阶段的完成判据（Go / No-Go）

只有同时满足以下条件，才允许迁移到 4090：

1. 所有本地测试通过
2. `SyncAligner` 无未来泄漏
3. `RiskMap` 输出结构稳定
4. `VisualPromptRenderer` 的 PiP 不遮挡主图
5. `SSEH` 对合法 JSON 解析成功，对非法输出 fallback 正常
6. `OfflineReplayEngine` 能完成 Pass A / B(stub) / C
7. 至少 1 个真实 SinD 切片可以端到端跑完
8. 输出 `LOCAL_TO_4090_HANDOFF.md`

### A.6 本地阶段必须交付的文件

本地阶段结束时，Codex 必须额外交付：

- `LOCAL_AUDIT.md`
- `LOCAL_TO_4090_HANDOFF.md`
- `outputs/local_smoke/guard_log.jsonl`
- `outputs/local_smoke/shp_stub_log.jsonl`
- `outputs/local_smoke/fused_log.jsonl`
- `outputs/local_smoke/rendered_examples/*.png`
- `outputs/local_smoke/summary.md`

`LOCAL_TO_4090_HANDOFF.md` 至少包含：

- 当前目录结构
- 已实现模块
- 未实现模块
- 运行命令
- 已知 bug / TODO
- 需要 4090 才能做的事情

---

## 阶段 B：云端 4090（目标：接入真实 Qwen2.5-VL 并做 pilot）

### B.1 本阶段目标

在 4090 上完成以下事情：

- 复现本地全部测试
- 把 Pass B 的 `stub_backend` 替换为真实 `Qwen2.5-VL` 推理
- 优先做**单图单次前向**，不强求一开始就接到复杂 streaming cache
- 测量吞吐 / 延迟 / 显存
- 做小规模 pilot：20–50 个触发事件
- 输出初版 `FARR / GRP / WSI / SSEH_valid_rate / latency`

### B.2 云端阶段必须额外读取的文件

在本地阶段的所有文件之外，4090 上还必须优先读取：

1. `LOCAL_TO_4090_HANDOFF.md`
2. `outputs/local_smoke/summary.md`
3. `outputs/local_smoke/guard_log.jsonl`
4. `outputs/local_smoke/shp_stub_log.jsonl`
5. `outputs/local_smoke/fused_log.jsonl`
6. `outputs/local_smoke/rendered_examples/*.png`

### B.3 云端阶段建议执行顺序

#### B0. 云端环境验证

先验证：

- `nvidia-smi`
- CUDA 是否可用
- PyTorch GPU 可用
- transformers / tokenizers / accelerate / opencv 是否可用

形成：`SERVER_4090_AUDIT.md`

#### B1. 复现本地结果

要求先在 4090 上重新运行：

- 全部单元测试
- 本地 smoke 样例

只有确认 4090 上与本地 stub 结果一致，才开始动真模型。

#### B2. 接入真实 Qwen2.5-VL（只改 Pass B）

实现优先级：

1. **首选**：在 Pass B 中实现“读取一张复合图 -> 单次前向 -> 输出 JSON -> 校验为 SHPCard”
2. **次选**：若现有 `VideoStreamingInference` 很好接，就轻量复用
3. **禁止**：为了追求“真 streaming”而重构整条主线

也就是说，4090 阶段首先只需要把：

- `pass_b_vlm_log.py`
- `vlm/vlm_branch_lite.py`
- `vlm/sseh.py`

改到能跑真实单图推理即可。

#### B3. 真实延迟写回

真实 VLM 输出后，必须记录：

- `t_obs_end`
- `latency_s`
- `t_emit = t_obs_end + latency_s`
- `raw_text`
- `parsed_shp`
- `schema_valid`

#### B4. 20–50 事件 pilot

优先选择以下事件：

- 左转让行
- 低速排队释放
- 交叉口接近但最终安全通过
- 明显高危冲突（用于检查 hard-floor）

对每个事件都要导出：

- 渲染图
- Guard 日志
- 原始 VLM 文本
- SHP JSON
- 融合 trace

#### B5. 统计与报告

至少输出：

- `SSEH_valid_rate`
- `mean_latency_s`
- `peak_vram_gb`
- `FARR`
- `GRP`
- `WSI`
- `hard_floor_violations`
- `guard_only_fallback_ratio`

形成：

- `outputs/pilot_4090/metrics.json`
- `outputs/pilot_4090/case_table.csv`
- `outputs/pilot_4090/qualitative_examples/`
- `outputs/pilot_4090/summary.md`

### B.4 云端阶段完成判据

只有同时满足以下条件，才能认定 Lite 正式跑通：

1. 真实单图 Qwen2.5-VL 推理可用
2. `SSEH_valid_rate >= 0.90`
3. `hard_floor_violations == 0`
4. `GRP <= 0.01`
5. `FARR > 0`
6. 至少 1 个“让行误报压制”案例成立
7. 至少 1 个“真实高危仍被 hard-floor 拦住”的案例成立
8. 所有核心输出都能追溯到 PNG / JSON / trace

---

## 4. Codex 实施时的参考顺序

### 4.1 读文件顺序（必须遵守）

1. 先读研究文档：`research_UAV_driving_risk_assessment_v5.md`
2. 再读原执行方案：`aerorisk_execution_plan.md`
3. 再读现有流式代码：
   - `stream_qwen_model.py`
   - `video_stream_inference.py`
   - `kv_cache_eviction.py`
   - `cache_manager.py`
4. 再读当前 prompt / 本文件
5. 最后才开始编码

### 4.2 编码优先顺序

1. dataclass / config
2. data
3. guard
4. render + schema + stub
5. fusion
6. replay
7. tests
8. only then real VLM on 4090

### 4.3 发生冲突时的决策顺序

实现冲突时按以下优先级处理：

1. **本文件**
2. 你实际运行环境的硬件约束
3. `AeroRisk` 的严格因果与 hard-floor 原则
4. 原研究文档
5. 原执行方案

---

## 5. 最终建议给你的实际使用方式

你实际使用时，建议分三步：

### 第一步：在本地 3050 上把 Local Prompt 交给 Codex

目标：把无模型闭环完全跑通。

### 第二步：检查交付物

确认是否拿到了：

- 测试
- PNG
- jsonl 日志
- handoff 文档

### 第三步：把 Server Prompt 交给 4090 上的 Codex

目标：只接真模型、做 pilot、出第一版结果。

---

## 6. 一句总原则

**3050 阶段解决“代码是否正确、接口是否稳定、闭环是否成立”；4090 阶段解决“真实 VLM 是否可用、延迟是否可接受、Lite 指标是否涨点”。**
