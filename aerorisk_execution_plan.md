# AeroRisk v5.2 执行方案 — Codex Plan Mode

> **用途**: 本文档作为 Codex 计划模式的输入，指导逐阶段实现 AeroRisk v5.2  
> **原则**: 本地优先 → 云端 4090 → 云端 H800；每步测试通过后才进入下一步  
> **核心参考**: `research_UAV_driving_risk_assessment_v5.md`（系统设计）  
> **现有代码**: `temporal_encoding/model/` 下的 `stream_qwen_model.py`、`video_stream_inference.py`、`kv_cache_eviction.py`、`cache_manager.py`  
> **v5.2 核心变化**: `SyncAligner` 10Hz 频域对齐 + `BaseGuard`/`RiskMap` 统一协议 + B0/B1/B2 基座消费适配 + PiP/SoM 单帧渲染 + 折扣型 DST 融合 + Base→Plugin 增量打榜

---

## 全局约束

1. **Python 环境**: `conda activate videollm`，Python 3.10+，PyTorch 2.x，transformers
2. **模型**: Qwen2.5-VL-3B-Instruct（本地权重或 HuggingFace 加载）
3. **硬件梯度**:
   - 本地 3050 (8GB)：代码编写、单元测试、CPU 逻辑验证
   - 云 4090 (24GB)：主实验、流式推理、参数消融
   - 云 H800 (80GB)：大规模实验、离线 VLM 对照、可选 LoRA
4. **测试铁律**: 每个阶段产出一个 `test_phase_N.py`，所有测试绿灯才允许进入下一阶段
5. **数据集**: SinD（核心打榜）+ DRIFT（运动稳定性）+ VisDrone（定性分析）
6. **底座**: HiVT (CVPR 2022) + Trajectron++ (ECCV 2020) + Guard-only
7. **统一工作频率**: 轨迹工作域统一到 10Hz；视频仍保留原生 fps，仅通过时间戳与轨迹对齐

### 目录结构

```
aerorisk/
├── __init__.py
├── config.py                      # 全局配置与超参数 (含 CTRA / DST / PiP 参数)
├── data/
│   ├── __init__.py
│   ├── base_adapter.py            # 抽象数据适配器
│   ├── sync_aligner.py            # 原生轨迹 -> 10Hz 工作频率对齐
│   ├── sind_adapter.py            # SinD 数据集适配器
│   ├── drift_adapter.py           # DRIFT 数据集适配器
│   ├── visdrone_adapter.py        # VisDrone 适配器 (仅定性)
│   └── unified_loader.py          # 统一事件迭代器
├── guard/
│   ├── __init__.py
│   ├── ctra_model.py              # CTRA 运动模型实现
│   ├── ssm_calculator.py          # TTC/DRAC/PET 计算 (基于 CTRA)
│   ├── guard_branch.py            # Guard 风险评级
│   └── sentinel.py                # Sentinel 触发与 RIP 生成
├── base/
│   ├── __init__.py
│   ├── base_guard.py              # BaseGuard / RiskMap 协议
│   ├── base_predictor.py          # 可插拔底座抽象接口
│   ├── hivt_adapter.py            # HiVT 底座适配器
│   └── trajectron_adapter.py      # Trajectron++ 底座适配器
├── vlm/
│   ├── __init__.py
│   ├── visual_prompt_renderer.py  # SoM + PiP 单帧视觉提示渲染
│   ├── frame_scheduler.py         # 自适应跳帧调度器
│   ├── sseh.py                    # 结构化语义证据头
│   ├── shp.py                     # SHP 卡片数据结构
│   └── vlm_branch.py              # VLM Watch 分支封装
├── fusion/
│   ├── __init__.py
│   ├── dst_combiner.py            # Dempster-Shafer 证据合成
│   ├── discounting.py             # Shafer 证据折扣
│   ├── da_kgrf.py                 # 时延感知语义重投影 (含 DST)
│   ├── wssd.py                    # 语义特征分层衰减
│   ├── causal_slot_group.py       # 因果槽组融合
│   └── transparent_fuser.py       # 透明有序融合器
├── memory/
│   ├── __init__.py
│   ├── da_cem.py                  # 双锚因果事件记忆
│   ├── event_card.py              # 事件卡数据结构
│   └── anchor_manager.py          # 锚点管理 (物理+语义双锚)
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py                 # 客观风险检测指标
│   ├── farr_calculator.py         # FARR 误报压制率
│   ├── gt_builder.py              # Physical GT / Type-S GT 生成协议
│   ├── wsi_calculator.py          # WSI 预警稳定性
│   ├── vram_stability.py          # VRAM-Stability Score
│   ├── type_k_evaluator.py        # Type-K 事件评估
│   ├── type_s_evaluator.py        # Type-S 事件评估
│   └── sav_calculator.py          # SAV + McNemar's test
├── pipeline.py                    # 完整 AeroRisk 管线
└── tests/
    ├── test_phase0_env.py
    ├── test_phase1_data.py
    ├── test_phase2_guard.py
    ├── test_phase3_vlm.py
    ├── test_phase4_fusion.py
    ├── test_phase5_memory.py
    ├── test_phase6_fuser.py
    ├── test_phase7_pipeline.py
    └── test_phase8_experiments.py
```

---

## 阶段 0：环境准备与代码审计

**硬件**: 本地  
**预计耗时**: 1 天  
**前置条件**: 无

### 0.1 目标

确认现有代码可运行，理解接口，创建项目骨架。

### 0.2 步骤

1. **审计现有代码**：
   - `stream_qwen_model.py`（259行）：三分支 mRoPE 位置追踪
     - Branch 1: 首次 prefill，调用标准 `get_rope_index`
     - Branch 2: chunk prefill，局部 `get_rope_index` + `position_ids += offset`
     - Branch 3: decode，`_last_cache_position + 1`
   - `video_stream_inference.py`（866行）：
     - `append_frame(frame)` — ViT 编码 + KV 追加 (~330–500ms/帧 on 4090)
     - `ask(question, query_image)` — snapshot → prefill → decode → restore
     - `ask_stream()` — 流式 QA
     - `ask_choice()` — log-prob 多选
   - `kv_cache_eviction.py`（717行）：
     - L1 Sink + Window (`_evict_sink_window`)
     - L2 + Temporal Sampling (`_evict_temporal_sampling`)
     - L3 Frame Importance (`_evict_frame_importance`)
   - `cache_manager.py`（280行）：
     - snapshot/restore 保护视频 KV Cache
     - 集成 KVCacheEvictor

2. **创建项目骨架**：
   ```bash
   mkdir -p aerorisk/{data,guard,base,vlm,fusion,memory,evaluation,tests}
   touch aerorisk/__init__.py aerorisk/config.py
   ```

3. **创建 `aerorisk/config.py`**：

   包含全局配置，关键参数如下：

   | 参数组 | 关键字段 | 默认值 | 说明 |
   |--------|----------|--------|------|
   | 时间协议 | `native_fps` | 25.0 | SinD 原生帧率 |
    | | `working_fps` | 10.0 | 基座统一工作频率 |
   | | `semantic_fps` | 8.0 | VLM 名义采样率 |
   | | `chunk_size` | 4 | VLM 每次处理帧数 |
    | CTRA | `ctra_dt` | 0.04 | 预测步长 = 1/native_fps |
    | | `ctra_horizon` | 4.0 | 预测视野 (秒) |
    | | `accel_smooth_window` | 5 | 加速度/角速度稳健估计窗口 |
    | | `derivative_smoother` | ema | 求导前平滑器：ema 或 kalman |
    | | `ema_alpha_v / ema_alpha_heading` | 0.4 / 0.3 | 速度/航向 EMA 系数 |
    | | `kalman_q / kalman_r` | 1e-3 / 1e-2 | 1D 卡尔曼过程/观测噪声 |
   | Guard | `ttc_thresholds` | {1:4.0, 2:2.0, 3:1.5, 4:1.0, 5:0.0} | TTC 阈值 |
   | | `guard_hard_floor` | 4 | Guard 硬底座阈值 |
    | WSSD | `tau_inv / tau_slow / tau_rel_base` | 30 / 5 / 0.5 | 三类特征 TTL (秒) |
    | | `tau_rel_max` | 2.0 | 低速博弈时的关系特征最大续命时间 |
    | DST | `dst_conflict_threshold` | 0.7 | K > 此值退化 Guard-only |
    | | `discount_strength` | 0.4 | 语义豁免对 Guard 的折扣强度 |
   | DA-CEM | `max_cache_tokens` | 130000 | KV Cache 上限 |
   | | `eviction_l1/l2/l3_ratio` | 0.70/0.85/0.95 | 淘汰触发比例 |
    | PiP/SoM | `global_render_size` | 1024 | 单帧渲染主图边长 |
    | | `pip_crop_size` | 224 | PiP 特写尺寸 |
    | | `max_pip_targets` | 4 | 单帧最多拼接的高危目标数 |
    | | `semantic_canvas_height` | 448 | PiP 专用黑色语义画布高度 |
    | | `render_canvas_size` | 1024x1472 | 最终送入 VLM 的复合画布尺寸 |
    | 对齐 | `interp_mode` | zoh | 轨迹重采样方式，默认零阶保持 |
    | | `allow_future_leakage` | false | 强制禁止跨时刻双边插值 |
    | | `ctra_hist_len` | 5 | B0 估计 $a,\omega$ 的历史帧数 |
   | 融合器 | `causal_slot_groups` | 4 组 | 因果槽组定义 |
   | | `exemption_max_levels` | 2 | 豁免最多降级数 |

### 0.3 测试

```python
# test_phase0_env.py
def test_existing_code_imports():
    from temporal_encoding.model.stream_qwen_model import StreamQwenModel
    from temporal_encoding.model.video_stream_inference import VideoStreamingInference
    from temporal_encoding.model.kv_cache_eviction import KVCacheEvictor, EvictionConfig
    from temporal_encoding.model.cache_manager import KVCacheManager

def test_config_creation():
    from aerorisk.config import AeroRiskConfig
    cfg = AeroRiskConfig()
    assert cfg.native_fps == 25.0
    assert cfg.working_fps == 10.0
    assert cfg.interp_mode == "zoh"
    assert cfg.allow_future_leakage is False
    assert len(cfg.causal_slot_groups) == 4
    assert cfg.dst_conflict_threshold == 0.7
```

**通过标准**: 所有导入和配置测试通过。

---

## 阶段 1：数据管道（SinD / DRIFT / VisDrone）

**硬件**: 本地（代码编写）+ 4090（数据加载验证）  
**预计耗时**: 5 天  
**前置条件**: 阶段 0 通过

### 1.1 目标

为 SinD 和 DRIFT 创建统一数据接口，输出标准化的轨迹帧数据 + 视频帧，并通过 `SyncAligner` 将所有轨迹映射到统一的 10Hz 工作频率。

### 1.2 数据集特点

| 数据集 | 来源 | 类型 | 关键特性 | 用途 |
|--------|------|------|----------|------|
| **SinD** | 清华 ITSC 2022 | 信号灯交叉口 | 4城、7类交通参与者、信号灯状态、HD Map | 核心打榜 |
| **DRIFT** | 多无人机 | 真实 UAV 视频 | 动态视角变化、2D 轨迹 | 运动稳定性验证 |
| **VisDrone** | — | 无人机视频 | 小目标密集场景 | 仅定性分析 |

### 1.3 代码设计

**`aerorisk/data/base_adapter.py`**:

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Iterator
import numpy as np

@dataclass
class TrackState:
    track_id: int
    frame_id: int
    t: float                  # 物理时间 = frame_id / native_fps
    x: float                  # 世界坐标 x (米)
    y: float                  # 世界坐标 y (米)
    vx: float                 # 速度 x (m/s)
    vy: float                 # 速度 y (m/s)
    heading: float            # 航向角 (弧度) — CTRA 需要
    yaw_rate: float           # 偏航角速度 (rad/s) — CTRA 需要
    ax: float                 # 世界坐标加速度 x (m/s²) — CTRA 需要
    ay: float                 # 世界坐标加速度 y (m/s²) — CTRA 需要
    width: float
    height: float
    obj_class: str = "car"

@dataclass
class FrameData:
    frame_id: int
    t: float
    t_working: float
    tracks: List[TrackState]
    image: Optional[np.ndarray] = None
    signal_state: Optional[Dict] = None  # SinD 信号灯状态
    metadata: Optional[Dict] = None

class BaseDataAdapter(ABC):
    @abstractmethod
    def get_native_fps(self) -> float: ...
    @abstractmethod
    def get_total_frames(self) -> int: ...
    @abstractmethod
    def get_frame(self, frame_id: int) -> FrameData: ...
    @abstractmethod
    def iter_frames(self, start: int = 0, end: int = None) -> Iterator[FrameData]: ...
    @abstractmethod
    def get_video_frame(self, frame_id: int) -> Optional[np.ndarray]: ...
```

**`aerorisk/data/sync_aligner.py`**:

```python
@dataclass
class AlignedTrackWindow:
    track_id: int
    times_native: np.ndarray
    times_working: np.ndarray
    xy: np.ndarray
    velocity: np.ndarray
    heading: np.ndarray

class SyncAligner:
    """将高频原始轨迹重采样到 10Hz 工作域"""

    def __init__(self, native_fps: float, working_fps: float = 10.0, mode: str = "zoh"):
        ...

    def align_tracks(self, track_df) -> AlignedTrackWindow:
        """保留 native 时间轴，同时输出 working 时间轴上的严格因果对齐结果"""
        ...
```

**强因果约束**：

- `SyncAligner` 在任意工作时刻 $T$ 只能消费 $\leq T$ 的观测点；
- 默认使用 **Zero-Order Hold (ZOH)**；
- 若启用因果外推，也只能基于最近两个历史点外推到 $T$；
- 严禁调用任何会读取 $T+\Delta t$ 观测值的双边线性插值器。

**`aerorisk/data/sind_adapter.py`**: 

- 读取 SinD CSV 轨迹 + 视频帧 + 信号灯状态
- 先经 `SyncAligner` 重采样到 10Hz 工作频率，再构造 `FrameData`
- TrackState 需从 CSV 计算 `heading`（由 vx/vy 反算 atan2）、`yaw_rate`（由相邻帧 heading 差分）以及 `ax/ay`（由速度二阶差分）
- 信号灯状态映射到 `signal_state` 字典

**`aerorisk/data/drift_adapter.py`**: 

- 读取 DRIFT 2D 轨迹 + 视频帧
- 同样通过 `SyncAligner` 统一到 10Hz 工作频率
- 需处理 UAV 视角变化带来的坐标系漂移

**`aerorisk/data/visdrone_adapter.py`**: 仅加载视频帧，用于定性分析。

**关键**: `TrackState` 新增 `heading`、`yaw_rate`、`ax/ay` 字段，这是 CTRA 运动模型所必需的；`FrameData` 同时保留原生物理时间和工作时间，用于 Base/VLM 解耦对齐。

### 1.4 测试

```python
# test_phase1_data.py

def test_sind_load():
    adapter = SinDAdapter("data/SinD/", recording="Tianjin_01")
    assert adapter.get_native_fps() == 25.0
    frame = adapter.get_frame(0)
    assert len(frame.tracks) > 0
    assert frame.image is not None  # 必须有视频帧

def test_sind_signal_state():
    adapter = SinDAdapter("data/SinD/", recording="Tianjin_01")
    frame = adapter.get_frame(100)
    assert frame.signal_state is not None  # SinD 提供信号灯

def test_sync_aligner_resamples_to_10hz():
    aligner = SyncAligner(native_fps=25.0, working_fps=10.0)
    aligned = aligner.align_tracks(dummy_track_df)
    dt = np.diff(aligned.times_working)
    assert np.allclose(dt, 0.1, atol=1e-4)

def test_sync_aligner_is_causal():
    """在 T 时刻不得读取未来观测点"""
    aligner = SyncAligner(native_fps=25.0, working_fps=10.0, mode="zoh")
    aligned = aligner.align_tracks(dummy_track_df)
    assert aligned.times_working[-1] <= aligned.times_native[-1]

def test_sync_aligner_no_future_leakage_under_sparse_input():
    """稀疏轨迹下也只能做 ZOH 或历史外推，不能双边插值"""
    aligner = SyncAligner(native_fps=25.0, working_fps=10.0, mode="zoh")
    aligned = aligner.align_tracks(sparse_dummy_track_df)
    assert not uses_future_observation(aligned)

def test_frame_keeps_dual_timestamps():
    adapter = SinDAdapter("data/SinD/", recording="Tianjin_01")
    frame = adapter.get_frame(0)
    assert hasattr(frame, 't')
    assert hasattr(frame, 't_working')

def test_track_has_ctra_fields():
    adapter = SinDAdapter("data/SinD/", recording="Tianjin_01")
    frame = adapter.get_frame(0)
    for t in frame.tracks:
        assert hasattr(t, 'heading')
        assert hasattr(t, 'yaw_rate')
        assert hasattr(t, 'ax')
        assert hasattr(t, 'ay')
        assert isinstance(t.heading, float)

def test_drift_load():
    adapter = DRIFTAdapter("data/DRIFT/", recording_id=1)
    frame = adapter.get_frame(0)
    assert frame.image is not None
    assert len(frame.tracks) > 0

def test_unified_loader():
    loader = UnifiedLoader(dataset="sind", base_path="data/SinD/")
    count = 0
    for frame in loader.iter_events(max_events=5):
        assert isinstance(frame, FrameData)
        count += 1
    assert count == 5
```

**通过标准**: SinD 和 DRIFT 各至少一个 recording 可正常加载（含视频帧、CTRA 字段）。

---

## 阶段 2：Guard 物理底座 + CTRA 运动模型

**硬件**: 本地（纯 CPU 计算）  
**预计耗时**: 5 天  
**前置条件**: 阶段 1 通过

### 2.1 目标

实现 CTRA 运动模型 + SSM 计算 + Guard 风险评级 + Sentinel 触发 + `BaseGuard` 统一协议。

### 2.2 代码设计

**`aerorisk/guard/ctra_model.py`** — CTRA 运动模型核心实现:

```python
import numpy as np
from dataclasses import dataclass

class EMASmoother:
    def __init__(self, alpha: float): ...
    def update(self, value: float) -> float: ...

class Kalman1D:
    def __init__(self, q: float, r: float): ...
    def update(self, measurement: float) -> float: ...

@dataclass
class CTRAState:
    """CTRA 状态向量 [x, y, v, a, θ, ω]"""
    x: float
    y: float
    v: float
    a: float
    theta: float
    omega: float

class CTRAPredictor:
    """恒定转弯率与加速度 (Constant Turn Rate and Acceleration) 运动模型"""

    def preprocess_history(self, history_tracks):
        """先平滑 v / heading，再求导得到 a / omega，避免差分放大噪声"""
        ...

    def predict(self, state: CTRAState, dt: float) -> CTRAState:
        v_next = state.v + state.a * dt
        theta_next = state.theta + state.omega * dt
        if abs(state.omega) > 1e-6:
            num_x = lambda tau: (state.v + state.a * tau) * np.cos(state.theta + state.omega * tau)
            num_y = lambda tau: (state.v + state.a * tau) * np.sin(state.theta + state.omega * tau)
            xs = np.linspace(0.0, dt, 8)
            x_new = state.x + np.trapz([num_x(tau) for tau in xs], xs)
            y_new = state.y + np.trapz([num_y(tau) for tau in xs], xs)
        else:
            x_new = state.x + state.v * np.cos(state.theta) * dt + 0.5 * state.a * np.cos(state.theta) * dt**2
            y_new = state.y + state.v * np.sin(state.theta) * dt + 0.5 * state.a * np.sin(state.theta) * dt**2
        return CTRAState(x=x_new, y=y_new, v=v_next, a=state.a, theta=theta_next, omega=state.omega)

    @staticmethod
    def from_track_state(track) -> CTRAState:
        v = np.sqrt(track.vx**2 + track.vy**2)
        a = np.sqrt(track.ax**2 + track.ay**2)
        return CTRAState(x=track.x, y=track.y, v=v, a=a, theta=track.heading, omega=track.yaw_rate)
```

**工程硬约束**：禁止直接对原始抖动轨迹做裸二阶差分。必须先经过 `EMA` 或 `Kalman1D` 平滑，再计算 $a$ 与 $\omega$，否则 B0 与物理复核分支会因噪声放大触发海量伪高危。

**`aerorisk/guard/ssm_calculator.py`**: 基于 CTRA 预测轨迹计算 TTC / DRAC / PET:

- TTC: 在 CTRA 预测轨迹上逐步查找两车最小距离 < 碰撞阈值的时间点
- DRAC: 基于 CTRA 预测的速度差和距离变化率
- PET: 两车经过同一空间点的时间差

**`aerorisk/guard/guard_branch.py`**: 5 级风险评级:

| 等级 | 名称 | TTC 条件 | DRAC 条件 |
|------|------|----------|-----------|
| 1 | Free-flow | TTC > 4.0s | — |
| 2 | Normal | 2.0 < TTC ≤ 4.0 | — |
| 3 | Caution | 1.5 < TTC ≤ 2.0 | DRAC > 2.0 |
| 4 | Danger | 1.0 < TTC ≤ 1.5 | DRAC > 3.35 |
| 5 | Critical | TTC ≤ 1.0 | DRAC > 5.0 |

**`aerorisk/guard/sentinel.py`**: 

- Sentinel 触发 (Guard ≥ 3) → 生成 RIP (关注区域裁剪)
- 优先级队列 + NMS 去重

**`aerorisk/base/base_guard.py`**: `BaseGuard` / `RiskMap` 统一协议:

```python
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class RiskItem:
    track_ids: tuple[int, int]
    timestamp: float
    risk_score: float
    predict_coords: np.ndarray

@dataclass
class RiskMap:
    items: list[RiskItem]

class BaseGuard(ABC):
    @abstractmethod
    def emit_risk_map(self, frame_data) -> RiskMap:
        ...
```

**`aerorisk/base/base_predictor.py`**: 可插拔底座抽象接口:

```python
from abc import ABC, abstractmethod

class BasePredictor(ABC):
    @abstractmethod
    def predict_trajectories(self, history_tracks, horizon):
        """预测多个目标的未来轨迹"""
        ...
    
    @abstractmethod
    def get_risk_score(self, frame_data) -> int:
        """返回 1-5 风险等级"""
        ...
    
    @abstractmethod
    def name(self) -> str:
        """底座名称，用于实验记录"""
        ...
```

**基座消费模式约束**：

- **B0 / Guard-only**：读取过去 5 帧 $v, heading$，瞬时求解 $a, \omega$ 后直接生成 `RiskMap`；
- **B1 / HiVT**：按 `track_id` 聚合历史轨迹窗并编码成时空张量，随后将预测结果映射为 `RiskMap`；
- **B2 / Trajectron++**：同步消费历史坐标流与当前原始视频帧，再将概率预测结果映射为 `RiskMap`。

### 2.3 测试

```python
# test_phase2_guard.py

def test_ctra_straight_with_accel():
    """直线加速运动 (omega=0)"""
    state = CTRAState(x=0, y=0, v=10, a=2, theta=0, omega=0)
    pred = CTRAPredictor().predict(state, dt=1.0)
    assert abs(pred.x - 11.0) < 0.05
    assert abs(pred.y - 0.0) < 0.01

def test_ctra_turning_with_accel():
    """转弯+加速时应形成非线性轨迹"""
    state = CTRAState(x=0, y=0, v=8, a=1.5, theta=0, omega=np.pi/12)
    pred = CTRAPredictor().predict(state, dt=1.0)
    assert pred.x > 0 and pred.y > 0
    assert pred.v > state.v

def test_ctra_from_track_state():
    """从 TrackState 正确构造 CTRAState"""
    from aerorisk.data.base_adapter import TrackState
    track = TrackState(track_id=1, frame_id=0, t=0.0,
                       x=0, y=0, vx=3, vy=4, heading=0.927,
                       yaw_rate=0.1, ax=0.6, ay=0.8, width=2, height=5)
    ctra = CTRAPredictor.from_track_state(track)
    assert abs(ctra.v - 5.0) < 0.01
    assert abs(ctra.a - 1.0) < 0.01

def test_ttc_ctra_known():
    """两车对向行驶，CTRA 预测的 TTC 与解析解一致"""
    # 车A: (0,0) heading=0, v=15 m/s
    # 车B: (60,0) heading=π, v=15 m/s
    # 解析 TTC ≈ 60 / 30 = 2.0s
    ...

def test_guard_risk_levels():
    """验证 5 级风险等级分类边界"""
    from aerorisk.guard.guard_branch import GuardBranch
    # TTC = 3.0 → level 2
    # TTC = 0.5 → level 5
    ...

def test_sentinel_priority_queue():
    """优先级队列排序与 NMS"""
    ...

def test_pluggable_base_interface():
    """验证 Guard 实现 BasePredictor 接口"""
    from aerorisk.base.base_predictor import BasePredictor
    guard = GuardBranch(config)
    assert isinstance(guard, BasePredictor)

def test_guard_emits_risk_map():
    """Guard 输出标准化 RiskMap"""
    risk_map = GuardBranch(config).emit_risk_map(frame_data)
    assert hasattr(risk_map, 'items')
```

**通过标准**: CTRA 预测正确（直线加速 + 转弯）；SSM 计算一致；Guard 等级分类正确；在 SinD 上能检出已知冲突。

---

## 阶段 3：VLM 流式推理 + PiP/SoM 渲染 + SSEH + 帧调度

**硬件**: 4090（需 GPU）  
**预计耗时**: 7 天  
**前置条件**: 阶段 1 通过（需要帧数据）

### 3.1 目标

实现 VLM Watch 分支：单帧视觉提示渲染器 → 帧调度器 → SSEH 约束解码 → SHP 卡片输出。

### 3.2 代码设计

**`aerorisk/vlm/visual_prompt_renderer.py`** — SoM + PiP 单帧视觉提示渲染:

```python
class VisualPromptRenderer:
    """Guard 引导的单帧多目标视觉提示渲染器"""

    def render(self, frame_data, sentinel_targets, predicted_trajs):
        # 1. 全局降采样到 1024x1024
        # 2. 在主图绘制 SoM 编号框与 CTRA 轨迹箭头
    # 3. 创建 1024x1472 黑色扩边画布，底部 448px 作为 Semantic Canvas
    # 4. 选取若干高危目标，生成 224x224 PiP 特写并排版到底部画布区
    # 5. 使用引导线连接主图目标框与 Semantic Canvas 中的 PiP
    # 6. 返回单张复合图，供 VLM 单次前向使用
        ...
```

**`aerorisk/vlm/shp.py`** — SHP 卡片数据结构（与 v5.0 §3.3.1 对齐）:

```python
from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum

class SceneTopology(Enum):
    STRAIGHT = "straight"
    MERGE_ZONE = "merge_zone"
    DIVERGE_ZONE = "diverge_zone"
    INTERSECTION = "intersection"
    ROUNDABOUT = "roundabout"
    RAMP = "ramp"
    UNKNOWN = "unknown"

class SignalControlType(Enum):
    SIGNALIZED = "signalized"
    UNSIGNALIZED = "unsignalized"
    YIELD = "yield"
    STOP_SIGN = "stop_sign"
    UNKNOWN = "unknown"

# ... 更多枚举定义

@dataclass
class SHPCard:
    """Semantic Hazard Portrait — 单帧语义快照"""
    t_obs_end: float                       # 观测截止物理时间
    scene_topology: SceneTopology
    obstacle_on_drivable_area: bool
    signal_control_type: SignalControlType
    priority_relation: str                 # ego_has_priority | other_has_priority | ambiguous
    spillback: str                         # none | detected | propagating
    lane_blockage: str                     # none | partial | full
    visibility_level: str                  # clear | limited | severely_limited
    occlusion_pattern: str                 # none | static | dynamic
    semantic_risk_level: int               # 1-5, VLM 自评
    confidence: float                      # 0-1
    evidence_text: str                     # VLM 原始证据文本
    
    def to_json_schema(self) -> dict:
        """返回 JSON Schema，用于 SSEH 约束解码"""
        ...
```

**`aerorisk/vlm/frame_scheduler.py`** — 环形缓冲区 + Latest-Frame-Wins:

```python
class FrameScheduler:
    """自适应跳帧调度器
    
    策略: 环形缓冲区 + Sentinel 优先 + Latest-Frame-Wins
    - 普通帧按 semantic_fps 采样
    - Sentinel 触发帧立即入队（含 RIP 裁剪信息）
    - 如果 VLM 处理跟不上，跳过旧帧，总是取最新
    """
    
    def __init__(self, config):
        self.buffer_size = config.frame_buffer_size  # 16
        self.sentinel_weight = config.sentinel_priority_weight
        self.ring_buffer = deque(maxlen=self.buffer_size)
    
    def push(self, frame_data, is_sentinel=False, rip=None): ...
    def pop_next(self) -> Optional[ScheduledFrame]: ...
```

**`aerorisk/vlm/sseh.py`** — JSON Schema 驱动约束解码:

```python
class SSEH:
    """结构化语义证据头
    
    将 VLM 自由文本输出约束为 SHP JSON:
    1. 构造 System Prompt + JSON Schema
    2. 约束解码 (logits_processor)
    3. 校验 + 回退机制
    """
    
    def __init__(self, schema: dict):
        self.schema = schema
        self.validator = jsonschema.Draft7Validator(schema)
    
    def build_prompt(self, frame_data, sentinel_info=None) -> str: ...
    def parse_response(self, raw_text: str) -> Optional[SHPCard]: ...
    def fallback(self, raw_text: str) -> SHPCard: ...
```

**`aerorisk/vlm/vlm_branch.py`** — 封装现有 `VideoStreamingInference`:

```python
class VLMBranch:
    """VLM Watch 分支 — 非阻塞、异步感知"""
    
    def __init__(self, model, config):
        self.inference = VideoStreamingInference(model, ...)
        self.renderer = VisualPromptRenderer(config)
        self.scheduler = FrameScheduler(config)
        self.sseh = SSEH(SHPCard.to_json_schema())
        self._latest_shp: Optional[SHPCard] = None
    
    def feed_frame(self, frame_data, sentinel_info=None): ...
    def process_next(self) -> Optional[SHPCard]: ...
    def get_latest_shp(self) -> Optional[SHPCard]: ...
```

**渲染 SOP 约束**：

1. 严禁把同一帧的多个高危目标拆成多张碎片图分别送入 VLM；
2. 必须先在 CPU 侧完成全局底图、SoM 标记、CTRA 预测箭头、PiP 拼接与引导线绘制；
3. 同一帧只允许一次 `Single-Pass` 前向，所有高危交互体共享一张复合图。
4. PiP 必须放在扩边后的黑色 **Semantic Canvas** 内，严禁覆盖原始道路画面、信号灯、斑马线或远端交通参与者。

### 3.3 关键一致性验证

1. SHP 卡片的 JSON Schema 枚举值与 `SHPCard` dataclass 字段完全匹配
2. `t_obs_end` 由 `frame_id / native_fps` 计算，与 `append_frame()` 时间戳对齐
3. VLM 分支使用现有 `VideoStreamingInference` 的 `ask()` / `ask_stream()` 接口
4. 帧调度器的环形缓冲区大小与 `append_frame()` 吞吐匹配
5. SoM 框颜色、PiP 特写与主图目标编号必须一一对应，避免跨目标歧义

### 3.4 测试

```python
# test_phase3_vlm.py

def test_shp_schema_valid():
    """SHP JSON Schema 与 SHPCard dataclass 字段完全匹配"""
    card = SHPCard(t_obs_end=1.0, scene_topology=SceneTopology.INTERSECTION, ...)
    schema = card.to_json_schema()
    # 验证 schema 中所有必填字段
    assert "scene_topology" in schema["properties"]
    assert "signal_control_type" in schema["properties"]

def test_shp_enum_consistency():
    """枚举值与 v5 文档一致"""
    assert set(SceneTopology.__members__.keys()) == {
        "STRAIGHT", "MERGE_ZONE", "DIVERGE_ZONE", 
        "INTERSECTION", "ROUNDABOUT", "RAMP", "UNKNOWN"
    }

def test_sseh_parse_valid():
    """合法 JSON 解析为 SHPCard"""
    sseh = SSEH(schema)
    raw = '{"scene_topology": "intersection", ...}'
    card = sseh.parse_response(raw)
    assert isinstance(card, SHPCard)

def test_sseh_parse_invalid_fallback():
    """无效 JSON 触发回退，不抛异常"""
    sseh = SSEH(schema)
    card = sseh.fallback("this is not json")
    assert card.confidence < 0.3  # 低置信度回退

def test_frame_scheduler_latest_wins():
    """Latest-Frame-Wins 跳帧逻辑"""
    scheduler = FrameScheduler(config)
    for i in range(20):  # 推入 20 帧
        scheduler.push(FrameData(frame_id=i, ...))
    frame = scheduler.pop_next()
    assert frame.frame_id >= 16  # 旧帧应被跳过

def test_frame_scheduler_sentinel_priority():
    """Sentinel 帧优先处理"""
    scheduler = FrameScheduler(config)
    scheduler.push(FrameData(frame_id=10, ...), is_sentinel=False)
    scheduler.push(FrameData(frame_id=5, ...), is_sentinel=True, rip=...)
    frame = scheduler.pop_next()
    assert frame.frame_id == 5  # Sentinel 优先

def test_visual_prompt_renderer_single_pass():
    """多个高危目标被合并为一张复合图，而非多张裁切图"""
    renderer = VisualPromptRenderer(config)
    rendered = renderer.render(frame_data, sentinel_targets=[...], predicted_trajs={...})
    assert rendered is not None
    assert rendered.shape[0] == 1472
    assert rendered.shape[1] == 1024

def test_visual_prompt_renderer_uses_semantic_canvas():
    """PiP 只能出现在扩边语义画布中，不能遮挡主图"""
    renderer = VisualPromptRenderer(config)
    rendered = renderer.render(frame_data, sentinel_targets=[...], predicted_trajs={...})
    assert pip_content_only_in_bottom_canvas(rendered, canvas_height=448)

# === 4090 GPU 测试 ===
def test_append_frame_throughput():
    """实测 append_frame 吞吐 (需 4090)"""
    model = StreamQwenModel.from_pretrained(...)
    inference = VideoStreamingInference(model, ...)
    t0 = time.time()
    for _ in range(10):
        inference.append_frame(dummy_frame)
    elapsed = time.time() - t0
    fps = 10 / elapsed
    print(f"append_frame throughput: {fps:.1f} fps")
    assert fps > 1.0  # 最低要求

def test_vlm_branch_e2e():
    """VLM 分支端到端：feed 帧 → process → 获取 SHP (需 4090)"""
    vlm = VLMBranch(model, config)
    vlm.feed_frame(frame_data)
    shp = vlm.process_next()
    assert shp is None or isinstance(shp, SHPCard)
```

**通过标准**: SSEH 回退率 < 50%；单帧编码吞吐 > 1fps (4090)；SoM/PiP 单帧渲染可用；SHP Schema 一致性验证通过。

---

## 阶段 4：DA-KGRF 时延融合 + 折扣型 DST 证据理论

**硬件**: 本地（纯逻辑）+ 4090（集成测试）  
**预计耗时**: 5 天  
**前置条件**: 阶段 3 通过

### 4.1 目标

实现 WSSD 分层衰减 + Shafer 折扣 + DST 证据合成 + 时延倒逼的语义续命机制。

### 4.2 代码设计

**`aerorisk/fusion/dst_combiner.py`** — Dempster-Shafer 证据合成:

```python
import numpy as np
from typing import Dict, Tuple

class DSTCombiner:
    """Dempster-Shafer 证据合成器
    
    辨别框架 Θ = {R_low, R_mid, R_high}
    
    Dempster 合成规则:
        m₁₂(A) = [1/(1-K)] · Σ_{B∩C=A} m₁(B)·m₂(C)
    其中冲突系数:
        K = Σ_{B∩C=∅} m₁(B)·m₂(C)
    """
    
    def __init__(self, frame_of_discernment: list):
        self.theta = frame_of_discernment
    
    def combine(self, m1: Dict, m2: Dict) -> Tuple[Dict, float]:
        """Dempster 合成规则"""
        K = 0.0
        combined = {}
        
        for A, mass_a in m1.items():
            for B, mass_b in m2.items():
                intersection = A & B
                product = mass_a * mass_b
                if len(intersection) == 0:
                    K += product
                else:
                    key = frozenset(intersection)
                    combined[key] = combined.get(key, 0.0) + product
        
        if abs(1 - K) < 1e-10:
            return {frozenset(self.theta): 1.0}, 1.0
        
        normalized = {k: v / (1 - K) for k, v in combined.items()}
        return normalized, K
    
    def to_pignistic_probability(self, mass: Dict) -> Dict:
        """BetP 转换：将 mass 分配到单例命题"""
        betp = {e: 0.0 for e in self.theta}
        for subset, m in mass.items():
            for e in subset:
                betp[e] += m / len(subset)
        return betp
```

**`aerorisk/fusion/discounting.py`** — Shafer 折扣:

```python
class ShaferDiscounting:
    def apply(self, mass: Dict, alpha_guard: float, theta: frozenset) -> Dict:
        discounted = {}
        for subset, value in mass.items():
            if subset == theta:
                continue
            discounted[subset] = (1 - alpha_guard) * value
        discounted[theta] = alpha_guard + (1 - alpha_guard) * mass.get(theta, 0.0)
        return discounted
```

**`aerorisk/fusion/wssd.py`** — 语义特征分层衰减:

```python
class WSSD:
    """Weighted Semantic Staleness Decay — 分层衰减
    
    三类特征不同的衰减策略:
    - h_inv (不变特征): scene_topology, obstacle_on_drivable_area
      → 继承，TTL = τ_inv (30s)
    - h_slow (慢动态): signal_control_type, priority_relation, spillback
      → 指数衰减，TTL = τ_slow (5s)
        - h_rel (关系特征): lane_blockage, visibility_level, occlusion_pattern
            → 过期标 "unknown"，TTL = 动态 τ_rel (0.5~2.0s)
    """
    
    def decay(self, shp_card: SHPCard, age: float, config, speed: float = 0.0, rel_accel: float = 0.0) -> SHPCard: ...
```

**`aerorisk/fusion/da_kgrf.py`** — 整合 WSSD + DST + 非阻塞缓冲:

```python
class DAKGRF:
    """Delay-Aware Knowledge-Graph Reproject & Fuse
    
    核心逻辑:
    1. 从 VLM 非阻塞获取最新 SHP
    2. 计算 SHP 年龄 (当前物理时间 - t_obs_end)
    3. WSSD 衰减
    4. 构造 Guard 证据 m_guard 和 VLM 证据 m_vlm
    5. 由语义豁免计算折扣系数 alpha_guard，并先折扣 Guard
    6. DST 合成 → (m_combined, K)
    7. 若语义过期，执行物理意图复核决定是否“续命”
    8. 若 K > dst_conflict_threshold → Guard-only 退化
    9. 否则 → BetP → 风险等级
    """
    
    def __init__(self, config):
        self.dst = DSTCombiner(["R_low", "R_mid", "R_high"])
        self.wssd = WSSD(config)
        self.config = config
    
    def fuse(self, r_guard: int, shp: Optional[SHPCard], 
             current_time: float) -> Tuple[int, dict]:
        """返回 (fused_risk_level, debug_info)"""
        if shp is None:
            return r_guard, {"mode": "guard_only", "reason": "no_shp"}
        
        age = current_time - shp.t_obs_end
        decayed = self.wssd.decay(shp, age, self.config)
        
        m_guard = self._guard_to_mass(r_guard)
        m_vlm = self._shp_to_mass(decayed)

        alpha_guard = self._compute_discount(decayed)
        m_guard = self.discount.apply(m_guard, alpha_guard, frozenset(self.dst.theta))
        
        combined, K = self.dst.combine(m_guard, m_vlm)
        
        if K > self.config.dst_conflict_threshold:
            return r_guard, {"mode": "guard_only", "reason": f"high_conflict_K={K:.3f}"}
        
        betp = self.dst.to_pignistic_probability(combined)
        fused_level = self._betp_to_level(betp)
        return fused_level, {"mode": "dst_fused", "K": K, "betp": betp}
```

### 4.3 测试

```python
# test_phase4_fusion.py

def test_dst_basic_combination():
    """基本 DST 合成 — 两源一致时应强化"""
    dst = DSTCombiner(["low", "mid", "high"])
    m1 = {frozenset(["high"]): 0.7, frozenset(["low", "mid", "high"]): 0.3}
    m2 = {frozenset(["high"]): 0.8, frozenset(["low", "mid", "high"]): 0.2}
    combined, K = dst.combine(m1, m2)
    assert K < 0.3  # 低冲突
    assert combined[frozenset(["high"])] > 0.9  # 信念强化

def test_dst_high_conflict_fallback():
    """高冲突时 K > 阈值，触发 Guard-only 退化"""
    dst = DSTCombiner(["low", "high"])
    m1 = {frozenset(["high"]): 0.99, frozenset(["low"]): 0.01}
    m2 = {frozenset(["low"]): 0.99, frozenset(["high"]): 0.01}
    _, K = dst.combine(m1, m2)
    assert K > 0.9  # 高冲突

def test_shafer_discounting_moves_mass_to_unknown():
    theta = frozenset(["low", "mid", "high"])
    discount = ShaferDiscounting()
    mass = {frozenset(["high"]): 0.8, theta: 0.2}
    discounted = discount.apply(mass, alpha_guard=0.5, theta=theta)
    assert discounted[frozenset(["high"])] == 0.4
    assert discounted[theta] == 0.6

def test_dst_betp_conversion():
    """BetP 转换正确 — 单例 mass 不变，多集合平均分配"""
    dst = DSTCombiner(["a", "b", "c"])
    mass = {frozenset(["a"]): 0.6, frozenset(["a", "b", "c"]): 0.4}
    betp = dst.to_pignistic_probability(mass)
    assert abs(betp["a"] - (0.6 + 0.4/3)) < 0.001

def test_wssd_inv_no_decay():
    """不变特征 (h_inv) 在 τ_inv 内不衰减"""
    card = SHPCard(..., scene_topology=SceneTopology.INTERSECTION)
    decayed = WSSD(config).decay(card, age=10.0, config=config)
    assert decayed.scene_topology == SceneTopology.INTERSECTION

def test_wssd_rel_expires():
    """关系特征 (h_rel) 超过 τ_rel 后标记 unknown"""
    card = SHPCard(..., visibility_level="clear")
    decayed = WSSD(config).decay(card, age=1.0, config=config)
    assert decayed.visibility_level == "unknown"  # 0.5s TTL 已过期

def test_dakgrf_no_shp_returns_guard():
    """无 SHP 时不阻塞，直接返回 Guard"""
    kgrf = DAKGRF(config)
    level, info = kgrf.fuse(r_guard=3, shp=None, current_time=10.0)
    assert level == 3
    assert info["mode"] == "guard_only"

def test_dakgrf_conflict_fallback():
    """Guard 和 VLM 严重矛盾时退化"""
    kgrf = DAKGRF(config)
    # Guard 说 level 5 (critical)，VLM 说 level 1 (free-flow)
    shp = SHPCard(..., semantic_risk_level=1, confidence=0.9)
    level, info = kgrf.fuse(r_guard=5, shp=shp, current_time=shp.t_obs_end)
    assert info["mode"] == "guard_only"
    assert info["K"] > 0.7

def test_intent_continuation_resets_age():
    """若物理轨迹验证“让行”语义成立，则允许关系特征续命"""
    ...
```

**通过标准**: 折扣型 DST 数学正确；WSSD 动态衰减边界正确；语义续命机制触发；无 SHP 时 Guard-only 不阻塞。

---

## 阶段 5：DA-CEM 双锚缓存管理

**硬件**: 4090（需要实际 KV Cache 操作）  
**预计耗时**: 5 天  
**前置条件**: 阶段 3 通过（需要运行中的 VLM）

### 5.1 目标

扩展现有 `kv_cache_eviction.py`，实现双锚保留（物理锚点 + 语义锚点）+ 事件卡结构。

### 5.2 代码设计

**`aerorisk/memory/anchor_manager.py`** — 锚点管理:

```python
from dataclasses import dataclass
from typing import Set

@dataclass
class Anchor:
    """锚点 — 标记某个 KV chunk 不可淘汰"""
    chunk_id: int
    anchor_type: str    # "physics" | "semantic"
    timestamp: float
    reason: str         # e.g. "r_guard=4" or "spillback=propagating"
    expiry: float       # 锚点过期时间

class AnchorManager:
    """双锚管理器 — 物理锚点 + 语义锚点"""
    
    def mark_physics_anchor(self, chunk_id: int, guard_result: int, t: float):
        """Guard ≥ 3 时标记物理锚点"""
        ...
    
    def mark_semantic_anchor(self, chunk_id: int, shp_card, t: float):
        """SHP 中出现重要语义事件时标记语义锚点"""
        ...
    
    def is_anchored(self, chunk_id: int) -> bool:
        """检查 chunk 是否有活跃锚点"""
        ...
    
    def get_anchored_chunks(self) -> Set[int]:
        """获取所有有锚点的 chunk ID"""
        ...
    
    def expire_old_anchors(self, current_time: float):
        """清理过期锚点"""
        ...
```

**`aerorisk/memory/da_cem.py`** — 集成 AnchorManager + 扩展淘汰:

```python
class DACEM:
    """双锚因果事件记忆 — 扩展现有 KVCacheEvictor"""
    
    def __init__(self, config, evictor: KVCacheEvictor):
        self.anchor_mgr = AnchorManager()
        self.evictor = evictor
        self.config = config
    
    def on_guard_result(self, chunk_id, guard_level, t):
        if guard_level >= 3:
            self.anchor_mgr.mark_physics_anchor(chunk_id, guard_level, t)
    
    def on_shp_result(self, chunk_id, shp_card, t):
        if self._is_semantically_important(shp_card):
            self.anchor_mgr.mark_semantic_anchor(chunk_id, shp_card, t)
    
    def evict_if_needed(self, current_tokens, current_time):
        self.anchor_mgr.expire_old_anchors(current_time)
        anchored = self.anchor_mgr.get_anchored_chunks()
        # 淘汰时跳过 anchored chunks
        self.evictor.evict(exclude_chunks=anchored)
```

**`aerorisk/memory/event_card.py`** — 事件卡数据结构:

```python
@dataclass
class EventCard:
    """事件卡 — 记录一次完整的风险事件"""
    event_id: str
    start_time: float
    end_time: float
    peak_risk_level: int
    guard_levels: List[int]
    shp_cards: List[SHPCard]
    fused_levels: List[int]
    dst_conflict_history: List[float]
    anchored_chunks: Set[int]
```

### 5.3 测试

```python
# test_phase5_memory.py

def test_anchor_physics_marks():
    """物理锚点：Guard ≥ 3 时标记"""
    mgr = AnchorManager()
    mgr.mark_physics_anchor(chunk_id=5, guard_result=4, t=10.0)
    assert mgr.is_anchored(5)
    assert not mgr.is_anchored(4)

def test_anchor_semantic_marks():
    """语义锚点：重要 SHP 事件标记"""
    mgr = AnchorManager()
    shp = SHPCard(..., spillback="propagating")
    mgr.mark_semantic_anchor(chunk_id=8, shp_card=shp, t=15.0)
    assert mgr.is_anchored(8)

def test_anchor_expiry():
    """锚点过期后可被淘汰"""
    mgr = AnchorManager()
    mgr.mark_physics_anchor(chunk_id=5, guard_result=3, t=10.0)
    mgr.expire_old_anchors(current_time=200.0)  # 远超 TTL
    assert not mgr.is_anchored(5)

def test_eviction_skips_anchored():
    """淘汰跳过有锚 chunk"""
    dacem = DACEM(config, evictor)
    dacem.on_guard_result(chunk_id=5, guard_level=4, t=10.0)
    # 触发淘汰
    result = dacem.evict_if_needed(current_tokens=120000, current_time=10.0)
    assert 5 not in result.evicted_chunks

def test_no_oom_5min():
    """5 分钟连续视频不 OOM (需 4090, 集成测试)"""
    # 模拟 5 分钟 @ 25fps = 7500 帧
    for frame_id in range(7500):
        pipeline.process_frame(frame_data)
    assert get_vram_usage() < 23.0  # < 23GB on 4090
```

**通过标准**: 双锚标记/过期正确；淘汰跳过有锚 chunk；4090 上 5 分钟不 OOM。

---

## 阶段 6：透明融合器 + 因果槽组

**硬件**: 本地（纯逻辑）+ 4090（集成测试）  
**预计耗时**: 4 天  
**前置条件**: 阶段 4 通过

### 6.1 目标

实现因果槽组融合器 + DST 合成 + 双向融合（增强 + 豁免）+ Guard 硬底座约束。

### 6.2 代码设计

**`aerorisk/fusion/causal_slot_group.py`**:

```python
CAUSAL_SLOT_GROUPS = {
    "flow_chain": ["spillback", "lane_blockage", "visibility_level"],
    "control":    ["signal_control_type", "priority_relation"],
    "structure":  ["scene_topology", "obstacle_on_drivable_area"],
    "occlusion":  ["occlusion_pattern"],
}

class CausalSlotEvaluator:
    """因果槽组评估 — 消除 double counting"""
    
    def evaluate_group(self, shp_card, group_name) -> float:
        """对一个因果槽组评分 (0~1)"""
        ...
    
    def evaluate_all(self, shp_card) -> Dict[str, float]:
        """对所有槽组评分"""
        ...
```

**`aerorisk/fusion/transparent_fuser.py`**:

```python
class TransparentFuser:
    """透明有序融合器 — 完整融合逻辑"""
    
    def fuse(self, r_guard, r_base, shp, current_time, config) -> FusionResult:
        """
        融合流程:
        1. 获取底座风险 r_base (HiVT/Traj++/Guard)
        2. 获取 Guard 风险 r_guard
        3. 取底座和Guard的较大值作为物理基础
        4. 若无 SHP → 直接返回物理基础
        5. WSSD 衰减 SHP
        6. Guard → mass, VLM → mass
        7. DST 合成 → (combined, K)
        8. K > threshold → Guard-only
        9. 因果槽组评估 → 增强/豁免
        10. Guard ≥ hard_floor → 不可豁免
        11. 豁免最多降 exemption_max_levels 级
        12. 输出 FusionResult (含完整 trace)
        """
        ...
```

### 6.3 测试

```python
# test_phase6_fuser.py

def test_guard_hard_floor():
    """Guard ≥ 4 时不可豁免"""
    fuser = TransparentFuser(config)
    result = fuser.fuse(r_guard=4, r_base=4,
                        shp=low_risk_shp, current_time=t)
    assert result.fused_level >= 4

def test_dst_conflict_fallback():
    """DST K > 0.7 时退化为 Guard-only"""
    fuser = TransparentFuser(config)
    # Guard says 5, VLM says 1 → high conflict
    result = fuser.fuse(r_guard=5, r_base=5,
                        shp=very_low_risk_shp, current_time=t)
    assert result.fused_level == 5
    assert result.trace["mode"] == "guard_only"

def test_no_double_counting():
    """因果链槽位不重复计分"""
    evaluator = CausalSlotEvaluator()
    shp = SHPCard(..., spillback="propagating", lane_blockage="full",
                  visibility_level="severely_limited")
    scores = evaluator.evaluate_all(shp)
    # flow_chain 组只算一次，不是三个独立加分
    assert "flow_chain" in scores
    assert scores["flow_chain"] <= 1.0

def test_enhancement_works():
    """语义增强提高风险"""
    fuser = TransparentFuser(config)
    # Guard=2, VLM 发现 spillback
    shp = SHPCard(..., spillback="propagating", semantic_risk_level=4)
    result = fuser.fuse(r_guard=2, r_base=2, shp=shp, current_time=t)
    assert result.fused_level > 2

def test_exemption_works():
    """语义豁免降低风险"""
    fuser = TransparentFuser(config)
    # Guard=3, 但 VLM 确认场景安全
    shp = SHPCard(..., semantic_risk_level=1, confidence=0.9)
    result = fuser.fuse(r_guard=3, r_base=3, shp=shp, current_time=t)
    assert result.fused_level < 3

def test_exemption_limit():
    """豁免最多降 2 级"""
    fuser = TransparentFuser(config)
    shp = SHPCard(..., semantic_risk_level=1, confidence=1.0)
    result = fuser.fuse(r_guard=3, r_base=3, shp=shp, current_time=t)
    assert result.fused_level >= 1  # 3 - 2 = 1

def test_fusion_trace_complete():
    """FusionResult 包含完整 trace"""
    result = fuser.fuse(r_guard=3, r_base=3, shp=shp, current_time=t)
    assert "K" in result.trace
    assert "mode" in result.trace
    assert "slot_scores" in result.trace
```

**通过标准**: 因果槽组消除 double counting；DST 冲突退化正常；豁免受 hard_floor + max_levels 双约束；trace 完整。

---

## 阶段 7：可插拔底座集成 + 完整管线

**硬件**: 4090  
**预计耗时**: 5 天  
**前置条件**: 阶段 2, 3, 4, 5, 6 全部通过

### 7.1 目标

1. 实现 HiVT 和 Trajectron++ 的底座适配器
2. 将 B0/B1/B2 全部接到 `BaseGuard` / `RiskMap` 协议上
3. 将所有模块串联为完整管线
4. 在 SinD 上端到端运行

### 7.2 代码设计

**`aerorisk/base/hivt_adapter.py`**:

```python
class HiVTAdapter(BasePredictor):
    """HiVT (CVPR 2022) 底座适配器
    
    - 加载预训练 / SinD 微调 HiVT 模型
    - 输入: 历史轨迹序列
    - 输出: 多模态未来轨迹预测
    - 使用预测轨迹 + SSM 计算风险等级
    """
    
    def __init__(self, model_path: str, config):
        self.model = load_hivt(model_path)
        self.ssm = SSMCalculator(config)
    
    def predict_trajectories(self, history_tracks, horizon):
        return self.model.predict(history_tracks, horizon)
    
    def get_risk_score(self, frame_data) -> int:
        trajectories = self.predict_trajectories(...)
        ttc, drac = self.ssm.compute(trajectories)
        return self.ssm.to_risk_level(ttc, drac)
    
    def name(self) -> str:
        return "HiVT"
```

**`aerorisk/base/trajectron_adapter.py`**: 类似结构，封装 Trajectron++ CVAE 模型；额外要求同步消费原始视频帧和历史轨迹流。

**`aerorisk/base/guard_only_adapter.py`**: B0 物理基座包装器

```python
class GuardOnlyAdapter(BasePredictor, BaseGuard):
    """B0：纯 CTRA + SSM 基座，直接输出 RiskMap"""

    def __init__(self, config):
        self.guard = GuardBranch(config)

    def predict_trajectories(self, history_tracks, horizon):
        return self.guard.predict(history_tracks, horizon)

    def emit_risk_map(self, frame_data) -> RiskMap:
        return self.guard.emit_risk_map(frame_data)

    def get_risk_score(self, frame_data) -> int:
        return self.guard.evaluate(frame_data)

    def name(self) -> str:
        return "B0-CTRA"
```

**`aerorisk/pipeline.py`** — 完整管线:

```python
class AeroRiskPipeline:
    """完整 AeroRisk 管线
    
    架构: Base底座 + Guard物理 + VLM旁路 → DST融合 → 风险输出
    
    关键特性:
    - VLM 分支完全非阻塞
    - 支持三种底座: Guard-only / HiVT / Trajectron++
    - DST 冲突退化保证安全性
    - 双锚 KV Cache 管理
    """
    
    def __init__(self, base_predictor: BasePredictor, config: AeroRiskConfig):
        self.guard = GuardBranch(config)
        self.vlm = VLMBranch(model, config)
        self.fuser = TransparentFuser(config)
        self.memory = DACEM(config, evictor)
        self.base = base_predictor
        self.config = config
    
    def process_frame(self, frame_data: FrameData) -> PipelineResult:
        # 1. Guard 物理评估
        r_guard = self.guard.evaluate(frame_data)
        
        # 2. 底座评估 + 标准化 RiskMap
        r_base = self.base.get_risk_score(frame_data)
        risk_map = self.base.emit_risk_map(frame_data)
        
        # 3. 按需触发：仅高关注风险进入 VLM 漏斗
        if risk_map.items:
            rip = self.guard.generate_rip(frame_data, risk_map=risk_map)
            self.vlm.feed_frame(frame_data, is_sentinel=True, rip=rip)
        else:
            self.vlm.feed_frame(frame_data)
        
        # 4. VLM 非阻塞处理
        shp = self.vlm.get_latest_shp()
        
        # 5. 透明融合
        fusion_result = self.fuser.fuse(
            r_guard=r_guard, r_base=r_base,
            shp=shp, current_time=frame_data.t,
            config=self.config
        )
        
        # 6. 记忆更新
        self.memory.on_guard_result(chunk_id, r_guard, frame_data.t)
        if shp:
            self.memory.on_shp_result(chunk_id, shp, frame_data.t)
        self.memory.evict_if_needed(current_tokens, frame_data.t)
        
        return PipelineResult(
            frame_id=frame_data.frame_id,
            t=frame_data.t,
            r_guard=r_guard,
            r_base=r_base,
            r_fused=fusion_result.fused_level,
            shp=shp,
            risk_map=risk_map,
            trace=fusion_result.trace,
            base_name=self.base.name()
        )
```

**管线口径**：

- `Base` 可以是 B0 / B1 / B2 中任意一种；
- 插件永远只消费 `RiskMap` 与当前视频帧，不直接窥探底座内部网络结构；
- 主实验全部以 `Base → Base + AeroRisk` 的形式记录增量。

### 7.3 测试

```python
# test_phase7_pipeline.py

def test_pipeline_guard_only():
    """Guard-only 底座端到端"""
    pipe = AeroRiskPipeline(GuardOnlyAdapter(config), config)
    for frame in sind_adapter.iter_frames(end=100):
        result = pipe.process_frame(frame)
        assert 1 <= result.r_fused <= 5
        assert result.base_name == "B0-CTRA"

def test_pipeline_with_hivt():
    """HiVT 底座 + AeroRisk"""
    pipe = AeroRiskPipeline(HiVTAdapter(model_path, config), config)
    for frame in sind_adapter.iter_frames(end=100):
        result = pipe.process_frame(frame)
        assert 1 <= result.r_fused <= 5
        assert result.base_name == "HiVT"

def test_pipeline_with_trajectron():
    """Trajectron++ 底座 + AeroRisk"""
    pipe = AeroRiskPipeline(TrajectronAdapter(model_path, config), config)
    for frame in sind_adapter.iter_frames(end=100):
        result = pipe.process_frame(frame)
        assert result.base_name == "Trajectron++"

def test_pipeline_no_oom_300s():
    """5 分钟不 OOM (需 4090)"""
    pipe = AeroRiskPipeline(GuardOnlyPredictor(config), config)
    for frame in sind_adapter.iter_frames(end=7500):  # 5min @ 25fps
        pipe.process_frame(frame)
    assert get_vram_usage() < 23.0

def test_pipeline_vlm_failure_graceful():
    """VLM 故障时降级不崩溃"""
    pipe = AeroRiskPipeline(GuardOnlyPredictor(config), config)
    pipe.vlm = BrokenVLM()  # 模拟故障
    for frame in sind_adapter.iter_frames(end=50):
        result = pipe.process_frame(frame)
        assert 1 <= result.r_fused <= 5  # 降级到 Guard-only

def test_pipeline_result_has_trace():
    """PipelineResult 包含完整 trace (可审计)"""
    result = pipe.process_frame(frame)
    assert "mode" in result.trace
    assert "K" in result.trace or result.trace["mode"] == "guard_only"

def test_pipeline_emits_standard_risk_map():
    """任意底座均应输出标准化 RiskMap"""
    result = pipe.process_frame(frame)
    assert hasattr(result, 'risk_map')
```

**通过标准**: 三种底座端到端运行完成；无 OOM；VLM 故障降级正常；trace 完整。

---

## 阶段 8：实验与评估

**硬件**: 4090 (主) + H800 (补充)  
**预计耗时**: 10 天  
**前置条件**: 阶段 7 通过

### 8.1 实验矩阵总览

| 类别 | 编号 | 实验名称 | 硬件 | 优先级 |
|------|------|----------|------|--------|
| **A: 多基座打榜** | E1 | B0-Base (CTRA + SSM) | 4090 | P0 |
| | E2 | Ours-A (B0 + AeroRisk) | 4090 | P0 |
| | E3 | B1-Base (HiVT) | 4090 | P0 |
| | E4 | Ours-B (HiVT + AeroRisk) | 4090 | P0 |
| | E5 | B2-Base (Trajectron++) | 4090 | P0 |
| | E6 | Ours-C (Traj++ + AeroRisk) | 4090 | P0 |
| **B: 模块消融** | AB-1 | w/o DA-KGRF | 4090 | P1 |
| | AB-2 | w/o DA-CEM | 4090 | P1 |
| | AB-3 | w/o PiP+SoM | 4090 | P1 |
| | AB-4 | w/o SSEH | 4090 | P1 |
| | AB-5 | 融合策略对比 (加权/标准DST/折扣DST/折扣DST+因果) | 4090 | P1 |
| **C: 鲁棒性** | R-1 | Sentinel 触发频率敏感性 | 4090 | P2 |
| | R-2 | 帧丢失鲁棒性 (10%/30%/50%) | 4090 | P2 |
| | R-3 | SinD 跨城市泛化 | 4090 | P2 |
| | R-4 | DST 冲突阈值 $K_{th}$ 敏感性 | 4090 | P2 |
| | R-5 | 错误语义折扣鲁棒性 | 4090 | P2 |
| **D: 工程实验** | S-1 | 位置编码碎片化回归 | 4090 | P1 |
| | S-1b | Single-Pass PiP/SoM 显存收益 | 4090 | P1 |
| | S-2 | VRAM 长时稳定性 (60min) | H800 | P2 |
| | S-3 | 淘汰水位对性能影响 | 4090 | P2 |
| | S-4 | LoRA 微调 vs 原始 | H800 | P3 |

### 8.2 执行优先级

**第一波 (P0 — 必须完成)**:
1. E1 B0-Base (CTRA) — 纯物理基线
2. E2 Ours-A — 零参数物理基座 + 插件
3. **检查点 A**: E2 vs E1 验证“零参数的逆袭”是否成立
4. E3 B1-Base (HiVT) — 复现 HiVT
5. E4 Ours-B — HiVT + VLM 插件
6. E5 B2-Base (Trajectron++) — 复现 Trajectron++
7. E6 Ours-C — Traj++ + VLM 插件

**第二波 (P1 — 重要消融)**:
8. 先生成 Physical GT 与 Type-S GT
9. AB-1~5 消融实验
10. S-1 / S-1b 工程实验
11. 增量分析 (Base → Base + Plugin)

**第三波 (P2 — 鲁棒性)**:
12. R-1~5 鲁棒性实验
13. S-2~3 工程实验

**第四波 (P3 — 可选)**:
13. S-4 LoRA 微调 (仅在 VLM 增益确认后)

### 8.3 评估指标代码

**`aerorisk/evaluation/gt_builder.py`** — 双规 GT 生成协议:

```python
class GTBuilder:
    """生成 Physical GT 与 Type-S GT"""

    def build_physical_gt(self, trajectories):
        # 扫描未来 3s；以 PET<1.5s 或极小最短距离作为客观高危标准
        ...

    def mine_type_s_subset(self, baseline_alerts, future_outcomes):
        # 反向筛选“低TTC但未来5s绝对安全”的高虚警切片
        ...
```

**`aerorisk/evaluation/metrics.py`** — 通用指标:

| 指标 | 公式/含义 | Type-K/S |
|------|-----------|----------|
| F1 | 精确率+召回率调和 | 全局 |
| GRP | Global Recall Penalty，全局召回折损 | 全局 |
| Type-S F1 | 仅统计 Type-S 事件 | Type-S |
| AUROC | ROC 曲线下面积 | 全局 |
| SAV | Sign Agreement Vote + McNemar's test | 全局 |

**`aerorisk/evaluation/recall_guardrail.py`** — 全局召回约束:

```python
class GlobalRecallPenalty:
    """GRP = max(0, Recall_base - Recall_plugin)"""

    @staticmethod
    def compute(recall_base: float, recall_plugin: float) -> float:
        return max(0.0, recall_base - recall_plugin)
```

**`aerorisk/evaluation/farr_calculator.py`** — 误报压制率:

```python
class FARRCalculator:
    """False Alarm Reduction Rate
    
    FARR = 1 - N_FA(AeroRisk) / N_FA(Base)
    
    衡量 AeroRisk 插件在不增加漏报的前提下
    压制了多少底座的误报。
    """
    @staticmethod
    def compute(n_fa_base: int, n_fa_aerorisk: int) -> float:
        if n_fa_base == 0:
            return 0.0
        return 1 - n_fa_aerorisk / n_fa_base
```

**`aerorisk/evaluation/wsi_calculator.py`** — 预警稳定性:

```python
class WSICalculator:
    """Warning Stability Index
    
    WSI = 1 - (阈值穿越次数 / 窗口大小)
    
    衡量风险等级输出的稳定性，避免
    在阈值附近频繁翻转。
    """
    @staticmethod
    def compute(risk_scores: list, threshold: int, window_size: int) -> float:
        crossings = 0
        for i in range(1, len(risk_scores)):
            if (risk_scores[i-1] < threshold) != (risk_scores[i] < threshold):
                crossings += 1
        return 1 - crossings / window_size
```

**`aerorisk/evaluation/vram_stability.py`** — 显存稳定性:

```python
class VRAMStabilityCalculator:
    """VRAM-Stability Score
    
    VRAMStab = 1 - Var(VRAM_samples) / Budget²
    
    衡量长时间运行中 VRAM 使用的稳定性。
    """
    @staticmethod
    def compute(vram_samples: list, budget_gb: float) -> float:
        return 1 - np.var(vram_samples) / (budget_gb ** 2)
```

### 8.4 结果记录模板

```python
experiment_report = {
    "experiment_id": "E4",
    "base": "B1-HiVT",
    "plugin": "AeroRisk",
    "dataset": "SinD_Tianjin",
    "metrics": {
        "F1": ...,
        "GRP": ...,
        "Type_S_F1": ...,
        "FARR": ...,
        "MTTW": ...,
        "WSI": ...,
        "AUROC": ...,
    },
    "delta_vs_base": {
        "F1": "+X.X%",
        "Type_S_F1": "+X.X%",
        "FARR": "X.X%",
    },
    "storyline_checks": {
        "zero_param_upset": true,
        "model_agnostic_gain": true,
    },
    "system_metrics": {
        "peak_vram_gb": ...,
        "vram_stability": ...,
        "sustained_minutes": ...,
    },
    "dst_stats": {
        "mean_K": ...,
        "mean_alpha_guard": ...,
        "K_above_threshold_ratio": ...,
        "guard_only_fallback_ratio": ...,
    },
}
```

### 8.5 降级判据（红线）

| 指标 | 红线 | 触发动作 |
|---|---|---|
| SAV p-value | > 0.05 | 放弃"预测增益"叙事，降级为"仅解释价值" |
| SSEH 回退率 | > 10% | 暂停实验，修订 SSEH prompt / schema |
| FARR | < 0 | 取消语义豁免机制 |
| GRP | > 0.01 | 认定插件通过“沉默”刷分，重新校准豁免权重 |
| 位置碎片化性能降 | > 15% | 探索位置编码修复方案 |
| 双向融合漏报增 | 显著增加 (McNemar p < 0.05) | 取消豁免，退化为单向增强 |
| 5 分钟 OOM | 任何一次 | 修正 DA-CEM 淘汰策略 |
| DST 冲突退化频率 | > 30% | 重新校准 Watch 证据映射 |
| 折扣后仍高冲突比例 | 持续偏高 | 检查是否把“语义豁免”错误实现成硬安全票 |
| Type-S F1 下降 | 较 baseline 下降 | 修正 VLM 证据权重 |

### 8.6 测试

```python
# test_phase8_experiments.py

def test_farr_calculation():
    """FARR 计算正确"""
    assert FARRCalculator.compute(100, 60) == 0.4
    assert FARRCalculator.compute(0, 0) == 0.0

def test_grp_calculation():
    """全局召回折损计算正确"""
    assert GlobalRecallPenalty.compute(0.95, 0.94) == 0.01
    assert GlobalRecallPenalty.compute(0.95, 0.97) == 0.0

def test_wsi_calculation():
    """WSI 计算正确"""
    scores = [1, 1, 3, 1, 3, 1, 3]  # 频繁穿越
    wsi = WSICalculator.compute(scores, threshold=2, window_size=7)
    assert wsi < 0.5  # 不稳定

def test_vram_stability():
    """VRAM-Stability 计算正确"""
    stable = [20.0, 20.1, 19.9, 20.0, 20.2]  # 稳定
    score = VRAMStabilityCalculator.compute(stable, budget_gb=24.0)
    assert score > 0.99

def test_experiment_report_schema():
    """实验报告包含所有必要字段"""
    ...

def test_gt_builder_protocol():
    """GT 生成器能同时输出 Physical GT 与 Type-S 子集"""
    ...

def test_delta_analysis():
    """增量分析：AeroRisk(Base) vs Base 的 delta 正确计算"""
    ...

def test_type_s_mining_uses_5s_window():
    """Type-S 挖掘按未来 5 秒绝对安全窗口执行"""
    ...

def test_plugin_cannot_trade_recall_for_farr():
    """即便 FARR 很高，若 GRP 超标也判为失败"""
    ...
```

**通过标准**: 所有指标计算逻辑正确；实验报告模板完整。

---

## 阶段间依赖关系

```text
阶段 0 ──→ 阶段 1 ──→ 阶段 2 (Guard+CTRA)  ──────────────┐
                  │                                         │
                  └──→ 阶段 3 (VLM) ──→ 阶段 4 (DST融合) ──→ 阶段 6 (融合器) ──→ 阶段 7 (管线) ──→ 阶段 8 (实验)
                              │                                                       │
                              └──→ 阶段 5 (DA-CEM) ──────────────────────────────────┘
```

**并行机会**:
- 阶段 2 (Guard+CTRA) 和 阶段 3 (VLM) 可**并行开发**
- 阶段 5 (DA-CEM) 和 阶段 4 (DST融合) 可**并行开发**

---

## 关键技术风险与缓解

| 风险 | 影响 | 概率 | 缓解策略 |
|---|---|---|---|
| `append_frame()` 吞吐 < 2fps | VLM 语义严重滞后 | 中 | 降低 chunk_size；探索批量编码 |
| 原始轨迹频率与底座训练域不匹配 | 风险评分系统性偏移 | 高 | 强制使用 `SyncAligner` 对齐到 10Hz |
| `SyncAligner` 误用双边插值 | 未来数据泄露，因果设定失效 | 高 | 默认 `zoh`；单测强查 `allow_future_leakage=False` |
| SSEH 回退率 > 30% | VLM 分支形同虚设 | 中 | 简化 Schema；少样本 prompt-tune |
| KV Cache 淘汰后性能暴跌 | 位置编码碎片化 | 高 | 缩小淘汰粒度；保留更多锚点 |
| HiVT/Traj++ 在 SinD 上无预训练 | 底座性能低 | 中 | 使用 SinD 训练集微调底座 |
| 标准 DST 高冲突悖论 | VLM 豁免彻底失效 | 中 | 先做 Shafer 折扣，再做 DST 合成 |
| SinD 数据申请延迟 | 阻塞实验进度 | 中 | 先用 DRIFT 跑通全流程 |
| CTRA 参数估计噪声 | Guard 风险评级不准 | 中 | 先做 EMA / Kalman 平滑，再求导 |
| PiP 覆盖原始场景 | 关键信号灯/拓扑被遮挡 | 中 | 使用扩边 `Semantic Canvas`，禁止覆盖主图 |
| 多目标图像提示过密 | VLM 识别错位 | 中 | SoM 编号 + PiP 数量上限 + Single-Pass 排版 |

---

## 估算时间线

| 阶段 | 天数 | 累计 | 里程碑 |
|------|------|------|--------|
| 0 环境准备 | 1 | 1 | 项目骨架建立 |
| 1 数据管道 | 6 | 7 | SinD/DRIFT 可加载 + 10Hz 对齐 |
| 2 Guard+CTRA | 5 | 12 | CTRA 预测 + RiskMap 正确 |
| 3 VLM | 7 | 19 | Single-Pass PiP/SoM + SSEH 输出 |
| 4 DST 融合 | 5 | 24 | DST 合成 + 衰减正确 |
| 5 DA-CEM | 5 | 24* | 双锚淘汰 (与4并行) |
| 6 融合器 | 4 | 28 | 透明融合器完整 |
| 7 管线集成 | 6 | 34 | B0/B1/B2 端到端跑通 |
| 8 实验 | 11 | 45 | 增量打榜数据完成（含 GT 构建） |

*阶段 2 与 3 并行，阶段 4 与 5 并行，实际关键路径 ≈ 37 天。

---

> **文档版本**: v5.2  
> **生成日期**: 2026-03-12  
> **配套文档**: `research_UAV_driving_risk_assessment_v5.md`  
> **核心变化 vs v4.0 执行方案**:
> 1. **对齐层**: 新增 `sync_aligner.py`，统一原始轨迹到 10Hz 工作频率
> 2. **协议层**: 新增 `base_guard.py`，把所有基座收敛到 `BaseGuard` / `RiskMap`
> 3. **基座层**: 明确 B0(B0-CTRA)、B1(HiVT)、B2(Traj++) 三类消费方式
> 4. **视觉层**: 强制 SoM + PiP Single-Pass 渲染 SOP，禁止多碎片图并发提问
> 5. **融合层**: 保留折扣型 DST + 因果槽组 + Guard 硬底座约束
> 6. **标注层**: Physical GT 用未来 3s 回溯，Type-S 用未来 5s 逆向挖掘
> 7. **实验层**: 从“单方法打榜”升级为 `Base → Base + Plugin` 增量打榜
> 8. **叙事层**: 额外验证“零参数物理底座 + VLM”与“模型无关增益”两条主线
