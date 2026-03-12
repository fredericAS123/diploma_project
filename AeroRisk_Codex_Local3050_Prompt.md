# 给 Codex 的本地 3050 提示词（详细执行版）

你现在运行在**本地 3050 环境**，目标不是完成整篇论文系统，而是完成 **AeroRisk Lite 的本地闭环**，为后续迁移到 4090 做准备。

你必须严格遵守下面的边界和步骤。

---

## 0. 你的任务定位

你当前只负责：

- 读取现有研究与执行文档
- 审计仓库现状
- 在本地实现 **不依赖真实 Qwen2.5-VL 推理** 的完整工程链路
- 把项目整理到“可迁移到 4090”的状态

你当前**不负责**：

- 在 3050 上追求真实 VLM 性能
- 实现 HiVT / Trajectron++
- 实现 DA-CEM
- 实现 LoRA
- 做大规模实验

如果你发现某些事只有 4090 才适合做，请不要在本地硬做，而是把它们清晰记录到 `LOCAL_TO_4090_HANDOFF.md`。

---

## 1. 编码前必须先读哪些文件

开始动代码前，必须先完整阅读并总结以下文件；若路径不存在，就在仓库中搜索等价文件：

1. `research_UAV_driving_risk_assessment_v5.md`
2. `aerorisk_execution_plan.md`
3. `temporal_encoding/model/stream_qwen_model.py`
4. `temporal_encoding/model/video_stream_inference.py`
5. `temporal_encoding/model/kv_cache_eviction.py`
6. `temporal_encoding/model/cache_manager.py`
7. `requirements.txt`、`environment.yml`、`pyproject.toml`、`setup.py`、`pytest.ini`（若存在）
8. 当前仓库下已有的 `tests/`、`scripts/`、`configs/`、`README*`

你必须先输出一份简洁的“仓库审计总结”，再开始实现。

---

## 2. 你必须遵守的设计规则

1. 这是 **AeroRisk Lite**，不是完整 v5.2。
2. 当前只做 **SinD + B0-CTRA + SoM/PiP + SHP + DST/WSSD + offline 3-pass replay**。
3. `SyncAligner` 必须严格因果，默认 `zoh`，禁止未来泄漏。
4. 保留 `guard_hard_floor = 4`。
5. Lite 默认 `exemption_max_levels = 1`。
6. 禁止 `4 -> 2`、`4 -> 1`、`3 -> 1`。
7. 让行/受控释放之类的豁免只能在 `r_guard <= 3` 且物理续验通过时允许 `3 -> 2`。
8. 在本地 3050 阶段，VLM 走 **stub / mock backend**。
9. 所有关键模块都要有测试，不允许只写空壳。
10. `VisualPromptRenderer` 只能输出**单张**复合图，PiP 必须位于底部 `Semantic Canvas`，不能遮挡主图。

---

## 3. 你的总体目标

在本地完成以下结果：

- 可以从 SinD 读取一个真实切片
- 可以通过 `SyncAligner` 映射到 10Hz 工作域
- 可以通过 B0-CTRA/Guard 生成 `r_guard` 和 `RiskMap`
- 可以为触发帧生成 SoM + PiP 复合图
- 可以通过 stub 生成合法 `SHPCard`
- 可以通过 `TransparentFuserLite` 做融合
- 可以用 `OfflineReplayEngine` 跑完 Pass A / Pass B(stub) / Pass C
- 可以输出日志、PNG、summary、handoff 文档

---

## 4. 你需要创建或补齐的目录

```text
aerorisk/
  __init__.py
  config.py
  data/
    __init__.py
    base_adapter.py
    sync_aligner.py
    sind_adapter.py
  guard/
    __init__.py
    ctra_model.py
    ssm_calculator.py
    guard_branch.py
    sentinel.py
  vlm/
    __init__.py
    visual_prompt_renderer.py
    shp.py
    sseh.py
    vlm_branch_lite.py
  fusion/
    __init__.py
    dst_combiner.py
    discounting.py
    wssd.py
    physical_verifier.py
    transparent_fuser_lite.py
  replay/
    __init__.py
    pass_a_guard_log.py
    pass_b_vlm_log.py
    pass_c_fuse_replay.py
    offline_replay_engine.py
  evaluation/
    __init__.py
    metrics.py
    farr_calculator.py
    wsi_calculator.py
  tests/
    test_phase0_env.py
    test_phase1_data.py
    test_phase2_guard.py
    test_phase3_render_and_sseh.py
    test_phase4_fuser_lite.py
    test_phase5_offline_replay.py
```

如果仓库已有部分目录，请复用并最小化改动。

---

## 5. 具体实施步骤

### Step 0：仓库审计

先执行但不限于：

- 查看目录树
- 找到环境文件
- 找到已有测试
- 找到现有 streaming VLM 代码
- 找到数据路径配置方式

然后输出：

- `LOCAL_AUDIT.md`

其中必须说明：

- 当前仓库已有模块
- 缺失模块
- 可能会复用的代码
- 潜在冲突点

### Step 1：实现基础 dataclass 与配置

先实现：

- `AeroRiskConfig`
- `TrackState`
- `FrameData`
- `RiskItem`
- `RiskMap`
- `SHPCard`
- `FusionResult`

配置中至少要有：

- `native_fps`
- `working_fps = 10.0`
- `interp_mode = "zoh"`
- `ctra_horizon`
- `guard_hard_floor = 4`
- `exemption_max_levels = 1`
- `tau_inv / tau_slow / tau_rel_base`
- `dst_conflict_threshold`
- `global_render_size`
- `pip_crop_size`
- `semantic_canvas_height`

先补 `tests/test_phase0_env.py`。

### Step 2：实现数据层

实现：

- `data/base_adapter.py`
- `data/sync_aligner.py`
- `data/sind_adapter.py`

关键要求：

- 同时保留原生时间 `t` 与工作时间 `t_working`
- `SyncAligner` 只能使用 `<= T` 的观测
- 稀疏输入下默认仍走 `zoh`
- 先对齐 `x/y/vx/vy/heading`，再计算 `ax/ay/yaw_rate`

补齐并通过：

- `tests/test_phase1_data.py`

### Step 3：实现 Guard/CTRA

实现：

- `guard/ctra_model.py`
- `guard/ssm_calculator.py`
- `guard/guard_branch.py`
- `guard/sentinel.py`

关键要求：

- 不允许裸差分直接放大噪声
- 先平滑，再求导
- 支持最小可运行的 TTC / DRAC
- 输出标准化 `RiskMap`
- `Sentinel` 至少支持：
  - 根据 `RiskMap` 选高危 pair
  - 为 renderer 提供目标 ID 列表

补齐并通过：

- `tests/test_phase2_guard.py`

### Step 4：实现渲染、SSEH 和本地 stub backend

实现：

- `vlm/visual_prompt_renderer.py`
- `vlm/shp.py`
- `vlm/sseh.py`
- `vlm/vlm_branch_lite.py`

要求：

- `VisualPromptRenderer` 将多个目标合成到一张图中
- 主图绘制 SoM 框与预测箭头
- PiP 贴到底部 `Semantic Canvas`
- `SSEH` 支持 schema 校验与 fallback
- `vlm_branch_lite.py` 默认提供 `stub_backend`

`stub_backend` 的最小行为：

- 根据预设规则返回一段 JSON 文本
- 再由 `SSEH` 解析为 `SHPCard`

补齐并通过：

- `tests/test_phase3_render_and_sseh.py`

### Step 5：实现融合逻辑

实现：

- `fusion/dst_combiner.py`
- `fusion/discounting.py`
- `fusion/wssd.py`
- `fusion/physical_verifier.py`
- `fusion/transparent_fuser_lite.py`

要求：

- `r_guard >= 4` 直接 hard-floor，不可豁免
- `PhysicalVerifier` 先支持最小让行验证
- `TransparentFuserLite` 输出完整 trace
- 若无 SHP，退化为 `guard_only`
- 若冲突过高，退化为 `guard_only`
- 若 `confidence` 不足或验证不通过，不允许豁免

补齐并通过：

- `tests/test_phase4_fuser_lite.py`

### Step 6：实现 offline 3-pass replay

实现：

- `replay/pass_a_guard_log.py`
- `replay/pass_b_vlm_log.py`
- `replay/pass_c_fuse_replay.py`
- `replay/offline_replay_engine.py`

要求：

- Pass A：Guard 日志
- Pass B：渲染图 + stub SHP 日志
- Pass C：融合回放，当前时刻 `T` 只能使用 `t_emit <= T` 的 SHP

补齐并通过：

- `tests/test_phase5_offline_replay.py`

### Step 7：本地 smoke test

使用：

- 1 个 synthetic case
- 1 个 SinD 真实切片

输出到：

```text
outputs/local_smoke/
  guard_log.jsonl
  shp_stub_log.jsonl
  fused_log.jsonl
  rendered_examples/
  summary.md
```

---

## 6. 你必须产出的 handoff 文档

本地阶段结束前，必须额外写：

- `LOCAL_TO_4090_HANDOFF.md`

其中必须包含：

1. 已实现模块
2. 未实现模块
3. 所有运行命令
4. 数据路径假设
5. 4090 阶段需要接入真实模型的位置
6. 推荐先修改哪些文件
7. 已知风险点

---

## 7. 你在本地阶段的验收标准

只有全部满足，才算本地阶段完成：

- 所有本地测试通过
- 至少 1 个 SinD 切片端到端跑通
- `SyncAligner` 无未来泄漏
- `VisualPromptRenderer` 的 PiP 不遮挡主图
- stub SHP 能被 `SSEH` 解析
- `TransparentFuserLite` trace 完整
- `OfflineReplayEngine` 可以跑完三遍
- 已生成 handoff 文档

---

## 8. 输出方式要求

你在完成本地阶段后，必须清楚列出：

- 新增文件
- 修改文件
- 关键设计决定
- 没做的事
- 4090 阶段应该从哪里接着做

不要直接跳去做 4090 才该做的真实大模型推理，除非只是非常轻量的可选 smoke，并且不会破坏本地闭环节奏。
