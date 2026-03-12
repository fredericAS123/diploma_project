# AeroRisk v5.2：基于流式多模态大模型的无人机交通风险评估框架

> **文档版本**: v5.2 — 在 v5.1 基础上进一步补齐数据流适配、基座协议与实验 SOP  
> **日期**: 2026-03-12  
> **核心修订**: 新增 Dataset-to-Base 频域对齐与 `SyncAligner`；统一 `BaseGuard`/`RiskMap` 协议；明确 B0/B1/B2 三类基座的数据消费模式；强化单帧 PiP + SoM 渲染 SOP；实验主线切换为 Base→Base+Plugin 的增量打榜  
> **硬件**: 本地 3050 (8GB) / 云 4090 (24GB) / 云 H800 (80GB)  

---

## 摘要

现有无人机交通风险评估方法可分为两类：纯运动学方法（TTC/DRAC/PET）在复杂语义场景下"语义色盲"，导致大量物理虚警与迟报；视觉时空特征方法（Trajectron++、HiVT）虽能捕捉像素层面的运动趋势，但无法理解"救护车优先"、"施工占道"等社会契约规则，在长尾场景下表现不稳定。

本文提出 **AeroRisk**——一种**模型无关的可插拔流式语义增强框架（Model-Agnostic Streaming Semantic Enhancer）**。其核心不是与现有轨迹预测 SOTA 竞争，而是**赋能**它们：通过引入流式 VLM（Qwen2.5-VL）的常识与因果推理，解决纯物理模型在复杂交互场景下的语义缺陷。

本文的主要贡献包括：

1. **严格因果的数据到基座频域对齐 SOP**：提出 `SyncAligner`，将 SinD/DRIFT 的高频 CSV 轨迹以**零阶保持（ZOH）**或**纯因果外推**方式映射到底座友好的 10Hz 工作频率，保证 Oracle Perception 注入时不发生未来数据泄露；
2. **可插拔双速率异步架构**：AeroRisk 作为插件接入任意 SOTA 底座（HiVT、Trajectron++ 等），底座以统一工作频率处理轨迹，VLM 旁路以 ~2–3 fps 实际吞吐提供非阻塞语义增强；
3. **Guard 引导的单帧多目标视觉提示**：通过 SoM 标记、未来轨迹箭头和 PiP 特写，将多目标高危关系压缩为单张排版图，实现 Single-Pass VLM 推理；
4. **结构化语义证据头 (SSEH)**：将 VLM 输出约束为可验证的 JSON Schema，生成结构化语义危险先验卡片 (SHP)，消除自由文本的不可控性与幻觉风险；
5. **统一 `BaseGuard` / `RiskMap` 协议**：不论底层是 CTRA、HiVT 还是 Trajectron++，都输出标准化的 `ID, Timestamp, Risk_Score, Predict_Coords`，从而实现基座无关的触发漏斗；
6. **时延感知运动学引导语义重投影框架 (DA-KGRF)**：基于物理时间戳 $t_{obs\_end}$ 的异步融合机制，结合 **Dempster-Shafer 证据理论 (DST)** 与 **Shafer 折扣机制**，实现校准的异步时序对齐；
7. **双锚因果事件记忆 (DA-CEM)**：同时以物理锚点和语义锚点管理 KV Cache，支撑长时因果链回溯，硬性显存封顶防止 OOM；
8. **CTRA 运动模型升级**：Guard 底座从 CV/CTRV 升级为恒定转弯率和加速度 (CTRA) 模型，提升减速入弯、加速出弯和交叉口曲线运动下的 TTC 预测精度；
9. **面向增量打榜的 GT 协议**：通过“事后物理回溯 + 长尾半自动挖掘”生成 Physical GT 与 Type-S 子集 GT，保证 F1、FARR、MTTW 等核心指标可被客观计算，并直接服务于 `Base → Base + AeroRisk` 的增量验证。

系统基于 Qwen2.5-VL-3B 模型，在 RTX 4090 / H800 上验证，使用 **SinD**（信号路口语义博弈）、**DRIFT**（真实 UAV 运动稳定性）、**VisDrone**（极小目标定性分析）数据集评估。

---

## 1. 引言

### 1.1 核心叙事：从一维推演到高维因果的认知升级

交通冲突检测方法的进化遵循一条清晰的认知阶梯：

**第一阶段：1D 运动学外推（守卫底座 — Guard）**

基于历史坐标，利用物理公式（CV/CTRV/CTRA）预测 $t+\Delta t$ 的位置，计算 TTC/DRAC/PET 等替代安全指标 (SSM) [5,6]。

- **能力**：低延迟（< 1ms/pair）、确定性强、可审计；
- **局限**："语义色盲"——不知道车辆为什么减速（是因为要撞了，还是因为交警打手势？），导致在复杂交互场景下产生大量物理虚警。

**第二阶段：2D 视觉时空特征（对比基线 — B2）**

利用 CNN/Transformer 提取视频像素特征，建模轨迹交互模式。代表方法包括 Trajectron++ [45]（ECCV 2020，条件变分自编码器）和 HiVT [46]（CVPR 2022，层次化向量 Transformer）。

- **能力**：可捕捉像素层面的运动趋势与交互模式；
- **局限**："因果黑盒"——能看到急刹车，但无法理解背后的"救护车优先"、"施工占道"等社会契约规则，在极端长尾场景下表现极不稳定。

**第三阶段：高维语义因果（我们的方案 — AeroRisk）**

通过流式 VLM 提取结构化语义卡片（SSEH），通过"语义豁免"修正 1D/2D 的预测偏见，实现物理精确性（Guard）与逻辑合理性（Watch）的闭环。

- **核心价值**：不是取代前两阶段，而是作为它们的**认知增强插件**——用 VLM 的常识推理填补物理模型的语义盲区。

### 1.2 学术定位

**AeroRisk 不是一个独立的检测系统，而是一种可插拔的流式语义增强框架。**

| 维度 | 传统方法 | AeroRisk |
|---|---|---|
| 竞争方式 | 与 SOTA 正面竞争指标 | 赋能任意 SOTA，使其"再涨点" |
| 技术路线 | 训练新的端到端模型 | 插件化非侵入式增强 |
| 语义能力 | 像素特征 → 隐式理解 | 结构化语义卡片 → 显式因果推理 |
| 可解释性 | 黑盒梯度归因 | 透明证据链审计 |

核心卖点：证明 AeroRisk 是 SOTA 模型的"前额叶"——为 HiVT、Trajectron++ 等添加因果推理能力，使它们在语义复杂场景下获得显著增益。

### 1.3 本文贡献

1. 提出模型无关的可插拔双速率异步架构，将任意 SOTA 物理/轨迹底座与流式 VLM 语义旁路解耦；
2. 引入 `SyncAligner` 频域对齐模块，将 SinD/DRIFT 轨迹统一映射到 10Hz 工作域，支撑 Oracle Perception 与不同基座的无偏对接；
3. 设计 `BaseGuard` / `RiskMap` 统一协议，使 CTRA、HiVT、Trajectron++ 都能被收敛到同一触发漏斗；
4. 设计 Guard 引导的单帧多目标视觉提示渲染器，以全局降采样 + SoM 标记 + PiP 特写实现单次前向语义理解；
5. 设计 SSEH + SHP 卡片系统，通过约束解码将 VLM 输出结构化；
6. 构建 DA-KGRF 时延感知融合框架，引入 DST 证据理论与 Shafer 折扣机制处理异源证据冲突；
7. 实现基于物理复核的动态语义续命，缓解短 TTL 语义在高延迟场景中的“出炉即过期”问题；
8. 实现 DA-CEM 双锚 KV Cache 管理，支撑长时因果链与显存封顶；
9. 将 Guard 物理模型从 CV/CTRV 升级为 CTRA，适应减速入弯和加速出弯场景；
10. 在 SinD 和 DRIFT 上进行 `Base → Base + AeroRisk Plugin` 增量打榜，并通过双规 GT 协议保证指标客观性。

---

## 2. 相关工作

### 2.1 替代安全指标与交通冲突检测

替代安全指标 (SSM) 是交通安全分析的核心工具 [5,6]：

- **TTC**: $\text{TTC}_{ij}(t) = \min_{\tau > 0}\{\tau : d_{ij}(t+\tau) = 0\}$ [43]
- **DRAC**: $\text{DRAC}_{ij}(t) = (\Delta v_{ij})^2 / (2 \cdot d_{ij}(t))$
- **PET**: $|t_i^{exit} - t_j^{enter}|$
- **TIT/TET**: 冲突持续时间的积分度量 [43]

局限性：所有 SSM 本质上都是运动学外推，无法感知导致冲突的语义前因。

### 2.2 轨迹预测 SOTA（我们的底座候选）

| 方法 | 会议 | 核心思想 | 局限 |
|---|---|---|---|
| **HiVT** [46] | CVPR 2022 | 层次化向量 Transformer，局部-全局交互 | 纯坐标输入，无语义感知 |
| **Trajectron++** [45] | ECCV 2020 | 条件变分自编码器 + 语义地图 | 像素特征无法理解社会规则 |
| **AgentFormer** [47] | ICCV 2021 | Agent-aware Transformer | 交互建模依赖训练分布 |

这些模型是我们的**底座 (Base)**——AeroRisk 作为插件接入后，验证在语义复杂场景下的增益。

### 2.3 VLM 在驾驶场景中的应用

- **DriveLM** [11]: 基于图结构的驾驶场景问答；
- **Driving with LLMs** [12]: 融合目标级向量信息实现可解释驾驶；
- **LingoQA** [13]: ECCV 2024, 驾驶场景视频问答基准；
- **DASH** [14]: 驾驶场景理解综合基准。

上述工作主要关注离线场景理解或车载视角。面向 UAV 俯瞰视角的**流式交通风险评估**尚属空白。

### 2.4 Dempster-Shafer 证据理论

Dempster-Shafer 理论 (DST) [48,49] 是一种不确定性推理框架，其核心优势在于能显式建模证据之间的冲突与不确定性，特别适合异源传感器融合场景。

**基本信念赋值 (BBA)**: 设辨识框架 $\Theta$，基本信念赋值 $m: 2^\Theta \to [0,1]$ 满足：

$$m(\emptyset) = 0, \quad \sum_{A \subseteq \Theta} m(A) = 1$$

**Dempster 合成规则**: 合并两个独立证据源 $m_1, m_2$：

$$m_{1,2}(A) = (m_1 \oplus m_2)(A) = \frac{1}{1-K} \sum_{B \cap C = A \neq \emptyset} m_1(B) \cdot m_2(C)$$

其中冲突系数:

$$K = \sum_{B \cap C = \emptyset} m_1(B) \cdot m_2(C)$$

$K$ 度量两个证据源之间的冲突程度。当 $K \to 1$ 时，两个证据源高度矛盾，标准 Dempster 归一化会把冲突质量完全重新分配到非空集合，进而产生“高冲突悖论”。

**Shafer 折扣机制**: 为避免“Guard 判高危、VLM 判可豁免”时直接导致 $K \to 1$，我们引入基于证据可靠度的折扣：设 Guard 原始信念赋值为 $m_G$，由 VLM 估计 Guard 需被折扣的幅度 $\alpha_{Guard}\in[0,1]$，则折扣后的 Guard 信念赋值为

$$m_G^{disc}(A) = (1-\alpha_{Guard})\,m_G(A), \quad \forall A \neq \Theta$$

$$m_G^{disc}(\Theta) = \alpha_{Guard} + (1-\alpha_{Guard})\,m_G(\Theta)$$

即：VLM 的“强安全语义”不是作为一个和 Guard 对冲的硬安全选票，而是把 Guard 的一部分确定性质量转移到“未知/待定”集合 $\Theta$。这样只有在 VLM 语义与当前场景真正无关、且无法通过物理轨迹复核时，系统才会保留高冲突并回退到 Guard-only。

**在 AeroRisk 中的应用**: Guard 分支和 VLM Watch 分支仍然通过 DST 合成获得联合风险判定，但在合成之前，先由语义豁免证据对 Guard 质量函数做折扣，从而将“柔性豁免”编码为不确定性调节，而不是“安全票硬碰高危票”。

### 2.5 CTRA 运动模型

恒定转弯率和加速度模型 (Constant Turn Rate and Acceleration, CTRA) 是 CTRV 的自然升级 [50]，其核心优势在于能同时刻画**入弯减速**与**出弯加速**，更适合信号交叉口的真实车辆动力学。

**状态向量**: $\mathbf{x} = [x, y, v, a, \theta, \omega]^T$

其中 $(x, y)$ 为位置，$v$ 为速度标量，$a$ 为纵向加速度，$\theta$ 为航向角，$\omega$ 为转弯率（偏航角速度）。

**状态转移方程**: 在小时间步 $\Delta t$ 下，CTRA 使用“恒定 $a$ + 恒定 $\omega$”做短时预测：

$$v_{t+\Delta t} = v_t + a_t \Delta t, \qquad \theta_{t+\Delta t} = \theta_t + \omega_t \Delta t$$

当 $|\omega_t| > \varepsilon$ 时，位置更新可写为

$$x_{t+\Delta t} = x_t + \int_0^{\Delta t} (v_t + a_t \tau)\cos(\theta_t + \omega_t \tau)\,d\tau$$

$$y_{t+\Delta t} = y_t + \int_0^{\Delta t} (v_t + a_t \tau)\sin(\theta_t + \omega_t \tau)\,d\tau$$

工程实现中采用闭式近似或高频离散积分；当 $\omega \approx 0$ 时退化为直线加速度模型：

$$x_{t+\Delta t} = x_t + v_t \cos(\theta_t)\Delta t + \frac{1}{2}a_t\cos(\theta_t)\Delta t^2$$

$$y_{t+\Delta t} = y_t + v_t \sin(\theta_t)\Delta t + \frac{1}{2}a_t\sin(\theta_t)\Delta t^2$$

**在 AeroRisk 中的应用**: Guard 分支使用 CTRA 而非 CTRV/CV 模型进行短期轨迹预测和 TTC 计算。CTRA 在以下场景显著优于 CTRV：
- **信号路口左转/右转**中的减速入弯、加速出弯；
- **排队蠕行博弈**中的小速度、大加速度变化；
- **拥堵溢出传播**下的走走停停车辆行为。

### 2.6 流式视频理解与 KV Cache 管理

- **StreamingLLM** [51]: 注意力 Sink Token + 滑动窗口；
- **StreamingVLM** [16]: MIT 提出的流式视觉语言模型架构；
- **LOOK-M** [52]: 多模态 KV Cache 压缩。

位置编码碎片化是流式 VLM 架构的固有挑战——当 KV Cache 中间段被淘汰后，Qwen2.5-VL 的 3D mRoPE 位置编码出现不连续跳跃。

### 2.7 无人机交通轨迹数据集

| 数据集 | 场景 | 特色 | 本文用途 |
|---|---|---|---|
| **SinD** [53] | 信号交叉路口 | 7 类交通参与者 + 信号灯状态 + HD Map | 核心打榜：语义博弈场景 |
| **DRIFT** [1] | 真实 UAV 动态视角 | 多无人机 + 原始视频 + 2D 轨迹 | 运动稳定性验证 |
| **VisDrone** [15] | 复杂城市场景 | 极小目标检测挑战 | 定性分析 CAMP 效果 |

**弃用说明**: v4.0 计划使用的 highD/inD 数据集仅提供 CSV 轨迹文件，**不提供原始视频流**。由于 AeroRisk 的 VLM 分支依赖视频帧作为输入，这两个数据集无法支撑流式 VLM 实验。因此 v5.0 彻底弃用 highD/inD，转用提供完整视频流的 SinD 和 DRIFT。

### 2.8 数据集到基座的频域对齐

开源 UAV 数据集的原始采样频率与下游预测底座的训练域通常并不一致。例如 SinD 常以 25Hz/50Hz 轨迹发布，而 HiVT 等主流轨迹预测模型常在 10Hz 工作域中训练和推理。若直接把原始 CSV 轨迹灌入基座，将引入时间步长失配、速度量纲偏移和预测窗不对齐的问题。

为此，AeroRisk v5.2 在数据层引入 **`SyncAligner`**：

1. 以数据集原始 `frame_id` 和 `native_fps` 为输入，恢复连续物理时间轴；
2. 默认采用**零阶保持（Zero-Order Hold, ZOH）**将所有轨迹重采样到统一的 10Hz 工作频率；仅在离线误差分析中允许对比“纯因果线性外推”，严禁使用任何跨越当前时间戳、读取未来观测点的双边线性插值；
3. 对 `x,y,v_x,v_y,heading` 做同步对齐，并在对齐后重新计算 $a$ 与 $\omega$；
4. 将“原生时间轴”和“工作时间轴”同时保留，前者用于 GT 回溯，后者用于 B0/B1/B2 基座推演。

这样可保证：底座模型在统一频域中比较，VLM 仍然以原始视频帧为视觉输入，二者通过时间戳实现严格对齐；更重要的是，系统在任意时刻 $T$ 只允许消费 $\leq T$ 的历史物理坐标，严格满足实时预测的因果律。

---

## 3. 方法

### 3.1 系统概览

AeroRisk 采用**可插拔双速率异步架构**：

```
┌─────────────────────────────────────────────────────────────────┐
│                      AeroRisk v5.2 系统架构                      │
│                                                                 │
│  ┌──────────────────────────────┐                               │
│  │   视频流输入 (原生 fps)        │                               │
│  └──────────┬───────────────────┘                               │
│             │                                                   │
│     ┌───────┴────────┐                                          │
│     ▼                ▼                                          │
│  ┌──────────────┐  ┌──────────────┐                             │
│  │ 可插拔底座     │  │  流式 VLM     │                             │
│  │ (Base Model) │  │ Qwen2.5-VL   │                             │
│  │ HiVT /       │  │  (8 fps 名义  │                             │
│  │ Trajectron++ │  │  ~2-3 fps 实际)│                             │
│  │ / Guard-only │  │               │                             │
│  └────┬─────────┘  └──────┬───────┘                             │
│       │                   │                                     │
│       ▼                   ▼                                     │
│  ┌──────────┐      ┌──────────────┐                             │
│  │  Guard    │      │   SSEH       │                             │
│  │ CTRA+SSM │      │ 结构化语义    │                             │
│  │ r_guard  │      │ 证据头       │                             │
│  └────┬─────┘      │ SHP 卡片     │                             │
│       │            └──────┬───────┘                             │
│       │                   │                                     │
│       │    ┌──────────────┘                                     │
│       │    │  DA-KGRF + DST                                     │
│       │    │  时延感知融合 + 证据理论                               │
│       │    │                                                    │
│       ▼    ▼                                                    │
│  ┌───────────────────────────┐                                  │
│  │    透明有序融合器           │                                  │
│  │  (DST 合成 + 因果槽组)    │                                  │
│  │        r_final            │                                  │
│  └───────────┬───────────────┘                                  │
│              │                                                  │
│              ▼                                                  │
│  ┌───────────────────────────┐                                  │
│  │  DA-CEM 双锚因果事件记忆   │                                  │
│  │  (KV Cache 管理)          │                                  │
│  └───────────────────────────┘                                  │
└─────────────────────────────────────────────────────────────────┘
```

**"插件化"核心原则**：

- Guard 分支**不绑定特定检测器**，作为可插拔接口：
  - 离线实验采用 **"完美感知假设 (Oracle Perception)"**，直接读取数据集 CSV 轨迹；
- **底座可替换**：HiVT、Trajectron++ 等任意 SOTA 均可作为 Guard 的轨迹预测底座；
- Watch 旁路始终**非阻塞**，底座无需等待 VLM。

**解耦仿真协议（逻辑防御）**：

为反击审稿人对"单卡无法同时跑两个大模型"及"实时性"的质疑：

1. **物理对齐声明**：实验采用"时间戳严格对齐的解耦仿真"——先运行底座（Guard/HiVT/Trajectron++）生成轨迹日志，再由 VLM（Watch）根据画面时间戳读取日志进行异步融合；
2. **因果保证**：通过 `Temporal Buffer` 确保 VLM 输出只影响其输入帧**之后**的系统状态，严禁未来数据泄露。

#### 3.1.1 Dataset-to-Base Alignment：Oracle Perception 注入协议

为在单卡和受限算力环境下完成严谨的脱机实验，AeroRisk 采用 **完美感知假设 (Oracle Perception)**：直接将 SinD / DRIFT 提供的 CSV 轨迹视为物理真值，并通过 `SyncAligner` 以严格因果方式对齐到统一工作频率后，注入不同基座。

三类基座的数据消费模式定义如下：

- **B0（CTRA 内置引擎）**：在时刻 $T$ 读取目标车辆过去 5 帧的 $v$ 与 $heading$，通过 $\Delta v / \Delta t$ 与 $\Delta \theta / \Delta t$ 求得 $a$ 和 $\omega$，再执行 CTRA 外推；
- **B1（HiVT）**：以 `track_id` 为键聚合邻域轨迹，截取过去 $T_{hist}$ 的坐标窗口，打包为时空张量输入 Transformer 底座；
- **B2（Trajectron++）**：读取历史坐标流的同时，额外截取对应无人机视频帧，提取环境拓扑语义后执行多模态概率预测。

上述三类基座在输入模态上不同，但都共享同一条物理时间轴与统一工作频率，确保对比公平。

### 3.2 运动学守卫分支 (Guard Branch)

Guard 分支是系统的**安全底座**，不依赖 VLM。

#### 3.2.1 CTRA 运动模型

从 v5.0 的 CTRV 进一步升级为 CTRA，状态向量 $\mathbf{x} = [x, y, v, a, \theta, \omega]^T$。

对每一对潜在冲突的交通参与者 $(i, j)$，使用 CTRA 预测未来轨迹，计算以下 SSM：

$$\text{TTC}_{ij}(t) = \min_{\tau > 0} \{ \tau : d_{ij}^{CTRA}(t+\tau) \leq r_{collision} \}$$

其中 $d_{ij}^{CTRA}(t+\tau)$ 由 CTRA 状态转移方程（§2.5）迭代计算。

$$\text{DRAC}_{ij}(t) = \frac{(\Delta v_{ij})^2}{2 \cdot d_{ij}(t)}$$

$$\text{PET}_{ij} = |t_i^{exit} - t_j^{enter}|$$

**速度/加速度来源优先级**（硬性规定）：
1. 数据集原生标注的速度（SinD/DRIFT 提供）；
2. 世界坐标系下的差分速度 $v = \Delta x / \Delta t$；
3. 世界坐标系下的二阶差分加速度 $a = \Delta v / \Delta t$；
4. 禁止从像素轨迹直接反推物理速度或加速度。

**工程细节**：为避免数值抖动，$a$ 与 $\omega$ 不直接从原始轨迹做裸差分，而是先对 $v$ 与 $heading$ 做轻量时序平滑，再求导。推荐两种等价实现：

1. **指数加权移动平均 (EMA)**：对 $v$ 与 $heading$ 先做低成本平滑，再计算 $\Delta v / \Delta t$ 与 $\Delta \theta / \Delta t$；
2. **一维卡尔曼滤波器 (1D Kalman Filter)**：分别对速度和航向做状态滤波，再求导得到 $a$ 与 $\omega$。

最终，$a$ 与 $\omega$ 仍可叠加长度为 $w$ 的稳健窗口估计（中值滤波或 Savitzky–Golay 平滑），仅在 Guard 内部使用，不向 VLM 暴露。这样可避免二阶差分把标注抖动放大为“乱飞轨迹”和海量 Sentinel 误触发。

#### 3.2.2 Guard 风险评级

$$r_{guard}(t) = f_{guard}(\text{TTC}, \text{DRAC}, \text{PET}, \Delta v, d) \in \{1, 2, 3, 4, 5\}$$

| 等级 | 含义 | 触发条件 |
|---|---|---|
| 1 | 安全 | TTC > 4s |
| 2 | 关注 | TTC ∈ [2, 4]s |
| 3 | 预警 | TTC ∈ [1.5, 2]s 或 DRAC > 2.0 |
| 4 | 高危 | TTC ∈ [1.0, 1.5]s 或 DRAC > 3.35 |
| 5 | 极危 | TTC < 1.0s 或 DRAC > 5.0 |

Guard 分支的判定具有**最高优先级**：$r_{guard} \geq 4$ 时，无论 VLM 判定如何，系统直接发出高危告警（Guard 硬底座不可豁免）。

#### 3.2.3 统一 `BaseGuard` / `RiskMap` 接口协议

v5.2 将 Guard 分支进一步形式化为统一协议：不论底层是 B0、B1 还是 B2，都必须被封装为 `BaseGuard`，并持续输出标准化的 `RiskMap`。其中，`RiskMap` 至少包含 `ID, Timestamp, Risk_Score, Predict_Coords` 四类字段，用于触发漏斗、PiP/SoM 渲染和后续 DST 融合。

标准接口如下：

```python
class BaseGuard(ABC):
    """统一 Guard 协议：任意底座都输出 RiskMap"""
    @abstractmethod
    def predict(self, history_tracks, horizon) -> PredictedTrajectories: ...
    
    @abstractmethod
    def emit_risk_map(self, frame_data) -> RiskMap: ...
```

在该协议下：

- **B0 / Guard-only**：直接基于 CTRA + SSM 计算 `Risk_Score`；
- **B1 / HiVT**：由 HiVT 预测轨迹后再映射为 `RiskMap`；
- **B2 / Trajectron++**：由多模态概率轨迹分布聚合为 `RiskMap`。

**按需触发漏斗**：系统默认静音绝大多数安全帧，仅当 TTC 或综合风险超过关注阈值（如 TTC < 3.0s）时，才对该帧发起一次 VLM 语义核查请求；若同帧存在多对高危交互体，则统一合并为单张复合图处理。

### 3.3 Guard 引导的单帧多目标视觉提示 + SSEH

#### 3.3.1 单帧多目标视觉提示渲染

为避免“多辆高危车分别裁切、多次调用 VLM”带来的空间认知错乱与 Token 爆炸，AeroRisk v5.2 采用 **Guard-guided Single-Frame Multi-Target Visual Prompting**：

1. **全局降采样**：将 4K 原始图压缩到 VLM 友好的工作分辨率（如 1024×1024）；
2. **SoM 标记**：借鉴 Set-of-Mark Prompting [54]，在主图上直接为 Guard 筛出的高危目标绘制颜色一致、可指代的框和编号；
3. **趋势注入**：使用 OpenCV 的 `arrowedLine` / `polylines` 在主图上绘制 CTRA 预测的短时未来轨迹箭头；
4. **PiP 画中画**：不直接覆盖原图，而是先将 1024×1024 主图放置到一个 1024×1472 的纯黑扩边画布上，并将底部 448 像素的 **Semantic Canvas** 作为专用 PiP 展示区；高危目标的 224×224 原分辨率特写统一排版到该展示区，并通过引导线连接 PiP 和主图目标；
5. **Single-Pass 推理**：无论当前帧存在多少对高危交互体，VLM 每轮只接收一张排版后的复合图像。

该设计一方面保留全局拓扑，另一方面通过 PiP 保留局部细节，从而显著缓解“局部裁切丢语境”和“多图输入引起跨图错位”的问题。更关键的是，**PiP 不再遮挡原始交通拓扑、信号灯和远处排队结构**，避免因为画中画覆盖导致新的视觉污染。

**工程约束**：严禁把 PiP 直接贴在原始交通场景之上；任何 PiP、文字标签和辅助说明都必须落在扩边后的黑色语义画布区域内。

#### 3.3.2 SHP 卡片数据结构

与现有代码 (`temporal_encoding/model/video_stream_inference.py`) 中 `ask()` 接口的输出对齐，SHP 卡片定义如下：

```python
@dataclass
class SHPCard:
    """Semantic Hazard Prior — 结构化语义证据卡"""
    # ── 时间元数据 ──
    chunk_start_frame: int
    chunk_end_frame: int
    t_obs_start: float          # 物理时间 = chunk_start_frame / native_fps
    t_obs_end: float            # 物理时间 = chunk_end_frame / native_fps
    t_emit: float               # VLM 完成生成的时刻 (time.time())
    native_fps: float
    semantic_fps: float
    
    # ── 不变特征 h_inv ──
    scene_topology: str         # enum: straight|merge_zone|diverge_zone|intersection|
                                #        roundabout|ramp|unknown
    signal_control_type: str    # enum: signalized|unsignalized|yield|stop_sign|unknown
    
    # ── 慢动态特征 h_slow ──
    lane_blockage: str          # enum: none|partial|full|unknown
    spillback: str              # enum: none|emerging|propagating|severe|unknown
    visibility_level: str       # enum: good|moderate|poor|unknown
    obstacle_on_drivable: str   # enum: none|static|slow_vehicle|debris|unknown
    
    # ── 关系特征 h_rel ──
    priority_relation: str      # enum: clear|ambiguous|contested|unknown
    occlusion_pattern: str      # enum: none|light|moderate|heavy_mutual|unknown
    
    # ── 元信息 ──
    confidence: float           # [0, 1]
    event_description: str      # 自然语言描述 (备用, 不参与融合计算)
```

**JSON Schema 约束解码**：在 `VideoStreamingInference.ask()` 中接入 JSON Schema 驱动的 token masking [24,25]，保证输出严格符合预定义 Schema。回退机制：若连续 $k$ 次校验失败，输出 `{all_slots: "unknown", confidence: 0.0}`。

#### 3.3.3 槽位级 TTL

| 槽位类别 | TTL | 衰减方式 |
|---|---|---|
| $h_{inv}$（不变特征）| $\tau_{inv}$: 15–60s | 直接继承，不衰减 |
| $h_{slow}$（慢动态）| $\tau_{slow}$: 2–10s | $\alpha_{age}(\delta) = e^{-\delta/\tau_{slow}}$ |
| $h_{rel}$（关系特征）| $\tau_{rel}$: 0.3–1.0s | 过期直接标 `unknown` |

TTL 起点锚定在 $t_{obs\_end}$（VLM 实际观测的最后一帧的物理时间），而非 $t_{emit}$。

### 3.4 时延感知运动学引导语义重投影框架 (DA-KGRF)

#### 3.4.1 核心问题

VLM 的自回归生成引入随机延迟 $\delta_{infer} = t_{emit} - t_{obs\_end}$。当融合器在物理时间 $T$ 需要使用语义证据时，最新可用 SHP 卡片的时延为 $\delta = T - t_{obs\_end}$。

#### 3.4.2 语义特征分层衰减 (WSSD)

$$\tilde{h}_{inv}(T) = h_{inv}(t_{obs\_end}) \quad \text{(直接继承)}$$

$$\tilde{h}_{slow}(T) = h_{slow}(t_{obs\_end}) \cdot \alpha_{age}(\delta), \quad \alpha_{age}(\delta) = e^{-\delta / \tau_{slow}}$$

$$\tilde{h}_{rel}(T) = \begin{cases} h_{rel}(t_{obs\_end}) \cdot \alpha_{age}(\delta) & \text{if } \delta \leq \tau_{rel} \\ \texttt{unknown} & \text{if } \delta > \tau_{rel} \end{cases}$$

其中，v5.2 将 $\tau_{rel}$ 升级为**动态半衰期**：

$$\tau_{rel}^{dyn} = \text{clip}\left(\tau_0 \cdot \frac{1}{1 + \lambda_v |v| + \lambda_a |a_{rel}|},\, \tau_{min},\, \tau_{max}\right)$$

在低速蠕行博弈、停车让行等场景中，$|v|$ 与 $|a_{rel}|$ 较小，因而允许 $\tau_{rel}^{dyn}$ 延长至 2.0s 左右；在高速穿越场景中则自动缩短，避免陈旧语义污染当前决策。

#### 3.4.3 带折扣的 DST 证据融合

将 Guard 和 Watch 分支各自对风险等级的判定建模为 **Dempster-Shafer 信念赋值**：

设辨识框架 $\Theta = \{R_{low}, R_{mid}, R_{high}\}$（低风险、中等风险、高风险）。

**Guard 证据**: 基于 SSM 计算得到信念赋值 $m_G$：

$$m_G(R_{high}) = \sigma\left(\frac{\theta_{ttc} - \text{TTC}}{\tau_G}\right), \quad m_G(\Theta) = 1 - m_G(R_{high}) - m_G(R_{low})$$

**Watch 证据**: 基于 SHP 卡片经 WSSD 衰减后得到 $m_W$：

$$m_W(A) = f_{DST}(\tilde{\mathbf{h}}_{inv}, \tilde{\mathbf{h}}_{slow}, \tilde{\mathbf{h}}_{rel}, \text{confidence}) \cdot \alpha_{age}(\delta)$$

**Shafer 折扣**: 若 Watch 识别到“让行成立”“交警指挥”“拥堵受控释放”等强语义豁免，则不直接投“安全票”，而是生成针对 Guard 的折扣系数 $\alpha_{Guard}$，并先计算

$$m_G^{disc}(A) = (1-\alpha_{Guard}) m_G(A), \quad A \neq \Theta$$

$$m_G^{disc}(\Theta) = \alpha_{Guard} + (1-\alpha_{Guard}) m_G(\Theta)$$

**DST 合成**:

$$m_{fused}(A) = \frac{1}{1-K} \sum_{B \cap C = A} m_G^{disc}(B) \cdot m_W(C)$$

**冲突退化机制**: 当冲突系数 $K > K_{threshold}$（如 0.7）时，表明当前语义既没有成功折扣 Guard，也未通过后续物理验证，系统自动退化为纯 Guard 模式：

$$r_{final} = r_{guard} \quad \text{if } K > K_{threshold}$$

这保证了：真正合理的“语义豁免”首先表现为对 Guard 确定性的折扣；只有 VLM 产生幻觉或完全脱离路况时，才会保留高冲突并回退到物理保底。

#### 3.4.4 时延倒逼的特征验证机制

v5.2 进一步加入 **Latency-driven Feature Verification**，解决 $h_{rel}$ 语义“出炉即作废”的问题。

设某个过期到达的语义特征为 $h_{rel}^{old}$，其观测时间为 $t_{obs\_end}$，到达当前时刻 $T$ 的延迟为 $\delta=T-t_{obs\_end}$。若 Guard 回溯过去 $\delta$ 时间段内的真实物理轨迹，发现其与语义意图一致（例如 Watch 判定“让行”，而 Guard 观察到对应车辆确实持续减速并放弃优先权），则执行**语义续命**：

$$\alpha_{age}^{reset} = 1.0, \qquad h_{rel}^{verified}(T)=h_{rel}^{old}$$

否则保留原始衰减或直接标记为 `unknown`。这使得 AeroRisk 能把“延迟到达但已被物理行为证实”的语义，从软证据升级为已验证语义。

#### 3.4.5 非阻塞异步缓冲

1. VLM 生成时记录 $t_{obs\_end}$；生成完成时记录 $t_{emit}$；
2. 融合器在 $T$ 查询时使用最新有效 SHP，不阻塞等待；
3. 若 VLM 正在生成中，使用上一可用卡片或退化为纯 Guard。

#### 3.4.6 自适应帧调度器

```text
原始视频帧流
     │
     ├──── Guard/底座 Branch：全量处理（原生 fps）
     │
     └──── VLM 环形帧缓冲区 B（容量 |B|=16，约 2s）
              │
              └── VLM 空闲时：取最新 chunk_size 帧 → 编码
                               跳过中间积压帧 (Latest-Frame-Wins)
```

实际语义帧率 $\text{effective\_fps} \approx 2\text{–}3 \text{ fps}$（由 `append_frame()` 的 ViT 编码耗时决定，RTX 4090 上单帧 ~330–500ms）。

**Sentinel 优先快进**: 当 Guard 触发高优先级事件时，VLM 优先处理最新帧而非排队帧。

### 3.5 双锚因果事件记忆 (DA-CEM)

#### 3.5.1 双锚保留策略

基于现有 `kv_cache_eviction.py` 中的三级淘汰框架扩展：

1. **物理锚点**: Guard 标记的高风险帧（$r_{guard} \geq 3$），优先保留其 KV Cache token；
2. **语义锚点**: VLM 输出中包含关键语义前因的 chunk，优先保留。

#### 3.5.2 强制显存上限与淘汰策略

$$N_{cache} \leq N_{max} \approx 130\text{k tokens} \quad \text{(4090 / 24GB)}$$

三级淘汰（与现有 `EvictionConfig` 对应）：

| 级别 | 触发 | 淘汰策略 | 对应代码 |
|---|---|---|---|
| L1 | $N > 0.7 N_{max}$ | 无锚点旧 chunk FIFO | `_evict_sink_window` |
| L2 | $N > 0.85 N_{max}$ | + 均匀时序采样 | `_evict_temporal_sampling` |
| L3 | $N > 0.95 N_{max}$ | 仅保留 Sink + 最新窗口 | 强制模式 |

Sink Token 保留：始终保留首 chunk（system prompt + 首帧视觉），现有代码中通过 `set_first_chunk_info()` 自动检测。

#### 3.5.3 位置编码碎片化缓解

沿用 v4.0 策略：
1. 整 chunk 淘汰，减少碎片化；
2. 淘汰后不重编号 position ID（等价于视频跳切）；
3. 通过消融实验量化影响。

### 3.6 透明有序融合器

#### 3.6.1 因果槽组融合

将语义槽位按因果依赖分组，消除重复计分：

| 槽组 | 包含槽位 | 因果逻辑 |
|---|---|---|
| $\mathcal{G}_1$ 流量链 | `spillback`, `lane_blockage`, `visibility_level` | 溢出 → 阻塞 → 视线受阻 |
| $\mathcal{G}_2$ 控制权 | `signal_control_type`, `priority_relation` | 信号决定优先权 |
| $\mathcal{G}_3$ 场景结构 | `scene_topology`, `obstacle_on_drivable_area` | 静态要素 |
| $\mathcal{G}_4$ 遮挡 | `occlusion_pattern` | 感知层面 |

**组内去重**: $\Delta_{\mathcal{G}_j} = \max_{k \in \mathcal{G}_j} [\alpha_{age,k}(\delta) \cdot \Delta_k(s_k)]$

**组间融合**:
$$g = r_{guard} + \sum_{j} w_{\mathcal{G}_j} \cdot \Delta_{\mathcal{G}_j} + \sum_{j < l} w_{jl} \cdot \psi(\Delta_{\mathcal{G}_j}, \Delta_{\mathcal{G}_l})$$

其中 $\psi(\Delta_i, \Delta_j) = \text{sign}(\Delta_i) \cdot \text{sign}(\Delta_j) \cdot \min(|\Delta_i|, |\Delta_j|)$

#### 3.6.2 融合规则

$$r_{final}(T) = \begin{cases} r_{guard} & \text{if } r_{guard} \geq \theta_{hard} \text{ (硬底座不可豁免)} \\ r_{guard} & \text{if } K > K_{threshold} \text{ (DST 冲突退化)} \\ g(r_{guard}, r_{watch}, \mathbf{s}, \boldsymbol{\alpha}_{age}) & \text{otherwise} \end{cases}$$

**豁免约束**：$r_{final} \geq r_{guard} - 2$（最多降 2 级）；仅当 VLM 相关槽位置信度 $\geq \theta_{conf}$ 且 TTL 有效时生效。

### 3.7 流式推理引擎

基于已实现的 `temporal_encoding/model/` 代码库：

- **`stream_qwen_model.py`**: `StreamQwenModel` — 三分支 mRoPE 位置追踪（Branch 1/2/3）；
- **`video_stream_inference.py`**: `VideoStreamingInference` — `append_frame()` / `ask()` / `ask_stream()` / `ask_choice()` 接口；
- **`kv_cache_eviction.py`**: `KVCacheEvictor` — Sink + Sliding Window + Temporal Sampling 三级淘汰；
- **`cache_manager.py`**: `KVCacheManager` — snapshot/restore + 淘汰集成。

**需要扩展的功能**：
1. SSEH 约束解码集成（在 `ask()` 中接入 JSON Schema token masking）；
2. SHP 时间元数据管理（`t_obs_end`, `t_emit` 自动记录）；
3. 双锚淘汰策略（Physics Anchor + Semantic Anchor 优先保留）；
4. Sentinel 事件优先级队列。

---

## 4. 实验设计

### 4.1 数据集

| 数据集 | 用途 | 核心场景 | 视频可用性 |
|---|---|---|---|
| **SinD** [53] | 核心打榜（语义博弈） | 信号路口、左转冲突、机非混行 | ✅ 4K 无人机视频 |
| **DRIFT** [1] | 运动稳定性验证 | 真实 UAV 动态视角 | ✅ 原始视频 |
| **VisDrone** [15] | 定性分析 | 极小目标、高密度 | ✅ 无人机视频 |

### 4.2 三类基座 + 插件对照

| 梯度 | 类别 | 代表模型 | 核心痛点 |
|---|---|---|---|
| **B0** | 物理内置基座 | **CTRA + SSM** | 纯运动学，语义色盲 |
| **B1** | 1D 纯轨迹 SOTA | **HiVT** (CVPR 2022) [46] | 只有坐标，无语义感知，易受环境突变干扰 |
| **B2** | 视觉轨迹 SOTA | **Trajectron++** (ECCV 2020) [45] | 像素特征无法理解社会规则 |
| **B3** | 消融基线 | **Qwen2.5-VL (单帧模式)** | 缺乏时间序列记忆，无法判断运动趋势 |
| **Ours** | **AeroRisk Plugin** | **Base + 流式 VLM** | 在不改底座参数下提供语义豁免/增强 |

**核心实验口径**：主表不再执着于“谁的绝对 F1 最高”，而是围绕 `Base` 与 `Base + AeroRisk Plugin` 组织，对比插件带来的**误报压制**与**长尾补益**。

### 4.3 实验矩阵

#### A. 性能打榜实验 (Main Results)

在 SinD 和 DRIFT 上进行全量对比：

| 实验组 | 方法 | 底座 | 说明 |
|---|---|---|---|
| E1 | B0-Base | CTRA + SSM | 纯物理基线 |
| E2 | Ours-A | CTRA + AeroRisk | 零参数物理基座 + 插件 |
| E3 | B1-Base | HiVT | 1D SOTA 基线 |
| E4 | Ours-B | HiVT + AeroRisk | HiVT 增量增强 |
| E5 | B2-Base | Trajectron++ | 2D SOTA 基线 |
| E6 | Ours-C | Trajectron++ + AeroRisk | Traj++ 增量增强 |
| E7 | B3: 单帧 Qwen-VL | — | 无流式、无结构化 |

**核心验证目标**:

1. 证明 `Ours-A`（CTRA + VLM）在 Type-S / FARR 等关键指标上可以逼近甚至超越未经增强的黑盒底座；
2. 证明 `Ours-B` 与 `Ours-C` 均能相对于各自 `Base` 获得稳定增益，从而支撑“模型无关”的学术定位。

#### B. 消融实验 (Ablation Studies)

| 编号 | 消融对象 | 配置 | 验证目标 |
|---|---|---|---|
| A1-A4 | DA-KGRF 时延融合 | A1: 无衰减 / A2: 简单衰减 / A3: 完整 WSSD / A4: 完整 WSSD + 语义续命 | 时延建模必要性 |
| B1-B3 | DA-CEM 记忆机制 | B1: 无记忆 / B2: 固定窗口 / B3: 双锚点 | 长时因果对风险判定的价值 |
| C1-C4 | 视觉提示与结构化输出 | C1: 多裁切多次推理 / C2: 单帧无标记 / C3: 单帧 SoM / C4: 单帧 SoM + PiP + SSEH | 单次前向视觉提示的价值 |
| D1-D4 | 融合策略 | D1: 简单加权 / D2: 标准 DST / D3: DST + 折扣 / D4: DST + 折扣 + 因果槽组 | 证据冲突时的系统稳定性 |

#### C. 鲁棒性与泛化实验 (Robustness Tests)

| 实验 | 说明 |
|---|---|
| R1 跨数据集泛化 | SinD 训练/调参 → DRIFT 零样本测试 |
| R2 视觉退化测试 | 人工加入噪声/雨雾遮挡/低光照 |
| R3 语义噪声鲁棒性 | 人为注入错误 VLM 语义，验证“折扣失败 → 高冲突 → Guard 回退”链路 |
| R4 VLM 故障测试 | 完全关闭 VLM，验证系统降级但不失效 |

**视觉退化测试 R2 详细设计**：

| 退化类型 | 模拟方式 | 预期行为 |
|---|---|---|
| 高斯噪声 | $\sigma \in \{10, 25, 50\}$ | VLM 置信度下降 → 自动降权 |
| 运动模糊 | 模拟 UAV 高速移动 | SSEH 回退率上升 → Guard 主导 |
| 雨雾遮挡 | 半透明叠加 | 视觉特征退化 → $h_{rel}$ 标记 unknown |
| 夜间低光照 | 整体降低亮度 | Guard 通过物理惯性维持系统下限 |

#### D. 系统工程实验 (Efficiency & Real-time)

| 实验 | 说明 |
|---|---|
| S1 采样率灵敏度 | Watch 在 1fps/4fps/8fps/10fps 下的性能变化 |
| S2 时延补偿验证 | DA-KGRF 在模拟 0.5s/1.0s/1.5s/2.0s 延迟下的风险对齐准确度 |
| S3 显存稳定性 | 长时运行（5min/10min/30min）的显存波动 |
| S4 DST 冲突系数分析 | 统计 $K$ 与折扣系数 $\alpha_{Guard}$ 的联合分布，验证柔性豁免是否生效 |

#### E. 增量测试 (Delta Analysis)

展示 `Base` → `Base + AeroRisk` 的性能跳变：

| 对照对 | 学术目标 |
|---|---|
| `B0-Base → Ours-A` | 验证“零参数物理引擎 + VLM 大脑”是否可挑战未增强黑盒底座 |
| `B1-Base → Ours-B` | 验证纯坐标流底座上的通用插件增益 |
| `B2-Base → Ours-C` | 验证像素+轨迹底座上的通用插件增益 |

| 指标 | B0 | Ours-A | $\Delta$ | B1 | Ours-B | $\Delta$ | B2 | Ours-C | $\Delta$ |
|---|---|---|---|---|---|---|---|---|---|
| F1 | — | — | — | — | — | — | — | — | — |
| Type-S F1 | — | — | — | — | — | — | — | — | — |
| FARR | — | — | — | — | — | — | — | — | — |
| MTTW | — | — | — | — | — | — | — | — | — |

### 4.4 标准答案 (Ground Truth) 生成协议

由于 SinD/DRIFT 不直接提供可用于打榜的风险时序标签，v5.2 采用“双规 GT”生成法则：

#### 4.4.1 Physical GT：事后客观物理回溯

利用上帝视角脚本离线遍历全数据集，对所有目标对 $(i,j)$ 计算**未来 3 秒窗口**内的最小空间距离、PET 和极端避险轨迹。若满足

$$\text{PET}_{ij}^{future} < 1.5\text{s} \quad \text{or} \quad d_{ij}^{min} < d_{crit}$$

则将对应时间切片标记为 `Risk=1`，并记录首次满足条件的时刻作为 MTTW 的客观参照时刻。

#### 4.4.2 Type-S Subset GT：长尾半自动挖掘

脚本首先反向筛选“基线模型给出低 TTC / 高风险，但**未来 5 秒**内实际上绝对安全”的虚警切片；再从中抽取约 200 个长尾样本进行少量人工审核，标注其语义原因，例如：

- `[交警/人工指挥]`
- `[拥堵蠕行/队列释放]`
- `[礼让行人/礼让非机动车]`
- `[静态障碍导致的保守绕行]`

若人工确认该切片属于“物理高警报但语义合理安全”，则记为 `Risk=0` 且进入 **Type-S 子集**，专门用于度量语义豁免能力与 FARR。该子集优先覆盖交警指挥、队列释放、礼让行人、拥堵蠕行等基线最易翻车的长尾场景。

#### 4.4.3 指标与 GT 的绑定关系

- **F1 / AUROC / Weighted F1**：基于 Physical GT 计算；
- **Global Recall / Recall Drop**：基于 Physical GT 计算，用于约束插件不能通过“沉默”换取高 FARR；
- **MTTW**：相对 Physical GT 首次高危时刻计算；
- **FARR**：仅在 Type-S 子集上统计“基线误报被 AeroRisk 压制”的比例；
- **Type-S Specific F1**：仅在 Type-S 子集上报告；
- **WSI**：在已标注事件窗口内统计预警抖动。

### 4.5 核心指标矩阵

#### 4.5.1 核心准确性指标 (Detection Performance)

| 指标 | 含义 | 备注 |
|---|---|---|
| **F1-Score / AUROC** | 全局风险检测能力 | 事件级 |
| **Global Recall Penalty (GRP)** | 插件相对基座的全局召回折损 | Physical GT 上必须接近 0 |
| **Type-S Specific F1** | 复杂语义场景子集得分 | 交警、占道、信号冲突等 |
| **FARR (False Alarm Reduction Rate)** | 语义豁免机制对物理误报的压制率 | **核心涨点指标** |
| Weighted F1 | 风险等级分类准确率 | 1–5 等级 |
| MAE | 风险等级偏差 | 保守/激进评估 |

**FARR 定义**:
$$\text{FARR} = 1 - \frac{N_{FA}^{AeroRisk}}{N_{FA}^{Base}}$$

其中 $N_{FA}$ 为误报数。FARR > 0 表示 AeroRisk 成功降低了基座的误报率。

**Global Recall Penalty 定义**:
$$\text{GRP} = \max\left(0, \text{Recall}_{Base} - \text{Recall}_{AeroRisk}\right)$$

GRP 用于防止“VLM 永远输出安全”这类沉默策略在 Type-S 子集上刷高 FARR。理想情况下，AeroRisk 在显著压制误报的同时，应满足 $\text{GRP} \leq 0.01$，即全局召回下降不超过 1 个百分点。

#### 4.5.2 预警质量指标 (Early Warning Quality)

| 指标 | 含义 |
|---|---|
| **MTTW (Mean Time to Warning)** | 平均预警提前时间 |
| **WSI (Warning Stability Index)** | 预警信号稳定性，防止阈值附近频繁抖动 |
| `lead_time_guard` / `lead_time_watch` | Guard/Watch 各自的首次越线时间 |

**WSI 定义**:
$$\text{WSI} = 1 - \frac{\text{count}(\text{alert crossings in window } W)}{|W|}$$

其中 alert crossing 指风险分数在报警阈值附近的穿越次数。WSI 越高越稳定。

#### 4.5.3 物理对齐指标 (Kinematic Alignment)

| 指标 | 含义 |
|---|---|
| TTC Deviation | VLM 预测风险与真实冲突时间的一致性误差 |
| Conflict Recall @ Criticality | 对 PET < 1.5s 高危冲突的召回率 |

#### 4.5.4 系统开销指标

| 指标 | 含义 |
|---|---|
| **VRAM-Stability Score** | 长时运行显存波动方差（证明 DA-CEM 封顶能力） |
| Peak VRAM | 峰值显存 (4090 / H800) |
| Sustained Duration | 24GB 下可持续运行的视频时长 |
| Per-event Semantic Cost | 每次 VLM 推理的平均耗时 |

**VRAM-Stability Score 定义**:
$$\text{VRAM\text{-}Stability} = 1 - \frac{\text{Var}(\text{VRAM}(t))}{\text{VRAM}_{budget}^2}$$

#### 4.5.5 统计显著性

- **McNemar's test**: 对 SAV 的显著性检验，报告 $p$-value；
- **Krippendorff's alpha**: 对 Type-S 语义标注的一致性报告（$\alpha \geq 0.67$ 方可使用）。

### 4.6 参数消融

| 变量 | 取值 |
|---|---|
| `semantic_fps` | 4, 8, 10 |
| `chunk_size` | 2, 4, 6 |
| `max_cache_tokens` | 100k, 130k, 150k |
| $\tau_{inv}$ | 15, 30, 60 s |
| $\tau_{slow}$ | 2, 5, 10 s |
| $\tau_{rel}$ | 0.3, 0.5, 1.0 s |
| $K_{threshold}$ (DST 冲突退化) | 0.5, 0.7, 0.9 |
| $\alpha_{Guard}$ 映射强度 | 0.2, 0.4, 0.6 |
| SoM/PiP 模式 | 无标记, SoM, SoM+PiP |
| CTRA vs CTRV vs CV | 对比三种运动模型的 TTC 预测精度 |

### 4.7 长视频因果链测试

```text
阶段 1 (t=5s)：出现占道障碍或 spillback，无几何冲突
阶段 2 (t=11s)：后车接近并急避让，形成真实冲突
阶段 3 (t=20s)：询问系统解释

Q1. 最早出现的因果性危险因素是什么？
Q2. 后续受影响的参与者有哪些？
Q3. 哪些证据支撑当前解释？
```

验证 DA-CEM 的因果链回溯能力。

### 4.8 预期结果与降级判据

**预期结果**：

1. AeroRisk(HiVT) 和 AeroRisk(Traj++) 在 Type-S F1 上显著优于原始 HiVT/Traj++；
2. FARR > 0：AeroRisk 有效压制底座的误报率，且主要收益集中在 Type-S 子集；
3. MTTW 提前量：在语义复杂场景子集上，AeroRisk 的预警提前量优于纯物理底座；
4. Guard 底座不劣化：AeroRisk 插件不拖累底座在纯物理场景下的性能；
5. **全局召回基本不降**：相对原始基座的 Recall 折损不超过 1%；
6. VRAM-Stability 通过：4090 上可持续运行 ≥ 10 分钟不 OOM。

**v5.2 学术主线**：

1. **零参数的逆袭**：`Ours-A` 是否能以“简陋物理公式 + VLM 大脑”挑战未经增强的顶会黑盒基座；
2. **绝对的普适性**：`Ours-B` 与 `Ours-C` 是否都能在各自底座上带来稳定的 FARR 增益，理想目标为 70%+ 的误报压制率；
3. **不对称打榜**：论文叙事从“绝对分数冠军”转为“插件化增益冠军”。

**降级判据（红线）**：

| 指标 | 红线 | 触发动作 |
|---|---|---|
| SAV p-value | > 0.05 | 放弃"预测增益"，降级为"仅解释价值" |
| SSEH 回退率 | > 10% | 暂停实验，修订 SSEH 设计 |
| FARR | < 0 (AeroRisk 反增误报) | 取消语义豁免机制 |
| Global Recall Penalty | > 1% | 认定插件通过“沉默”刷分，重新校准豁免权重 |
| 位置碎片化性能降 | > 15% | 探索位置编码修复 |
| 双向融合漏报增 | 显著增加 | 取消豁免，退化为单向融合 |
| 5 分钟 OOM | 任何一次 | 修正 DA-CEM 淘汰策略 |
| DST 冲突退化频率 | > 30% 的帧 | 重新校准 Watch 证据映射 |
| 折扣后仍高冲突比例 | 持续偏高 | 检查语义豁免是否被错误编码为“安全票” |

---

## 5. 实现与资源规划

### 5.1 硬件环境

| 资源 | 用途 |
|---|---|
| **本地 3050 (8GB)** | 代码编写、单元测试、CPU 逻辑验证 |
| **云 4090 (24GB)** | 主实验：流式推理、多基座打榜、参数消融 |
| **云 H800 (80GB)** | 大规模实验、离线 VLM 对照、可选 LoRA 微调 |

### 5.2 执行计划 (12 周)

```text
Week 1-2: 数据准备与环境
  - 获取 SinD / DRIFT 数据集
  - 实现数据适配器（SinD CSV+视频、DRIFT 轨迹+视频）
  - 复现 HiVT、Trajectron++ 基线在 SinD 上的性能
  - 固化物理时间协议

Week 3-4: Guard 底座 + CTRA 升级
  - 实现 CTRA 运动模型及 TTC 计算
  - 实现 Guard-only 基线 (E1)
  - 对比 CTRA vs CTRV vs CV 的 TTC 预测精度
  - 实现可插拔底座接口

Week 5-6: VLM 语义集成
  - 实现 SoM + PiP 单帧视觉提示渲染器
  - 实现 SSEH 约束解码 + SHP 卡片
  - 实现帧调度器 + Sentinel 优先队列
  - 实现 DA-KGRF 分层衰减 + 折扣型 DST 融合
  - 消融实验 A (DA-KGRF), C (SSEH)

Week 7-8: 记忆 + 融合 + 双锚
  - 扩展 kv_cache_eviction.py 支持双锚保留
  - 实现因果槽组融合器 + 折扣型 DST 合成
  - 实现时延倒逼的语义续命验证
  - 消融实验 B (DA-CEM), D (融合策略)

Week 9-10: 全系统打榜
  - 生成 Physical GT 与 Type-S Subset GT
  - AeroRisk(HiVT) 和 AeroRisk(Traj++) 完整实验
  - 鲁棒性实验 R1-R4
  - 工程实验 S1-S4
  - 增量测试 Delta Analysis

Week 11-12: 论文撰写
  - 汇总打榜表格
  - 失败案例分析 + 误报/漏报分析
  - 论文初稿 + 修图
```

### 5.3 微调策略

仅在以下前提全部成立后，才进入 LoRA 微调：

1. AeroRisk 插件在至少一个底座上显示统计显著的 Type-S F1 增益；
2. SSEH 回退率 < 10%；
3. DA-CEM 已证明长历史补链价值。

微调目标：`event_type` 准确性、`unknown` 输出校准、`support_frames` 定位精度。

---

## 6. 论文亮点总结

| # | 亮点 | 验证方式 |
|---|---|---|
| 1 | 模型无关可插拔 | 同一框架下 HiVT + AeroRisk 和 Traj++ + AeroRisk 均涨点 |
| 2 | 单帧 PiP + SoM 视觉提示 | 消融实验 C: 多裁切 vs SoM vs SoM+PiP |
| 3 | 折扣型 DST 融合 | 消融实验 D: 标准 DST vs 折扣型 DST |
| 4 | CTRA 运动模型 | 参数消融: CTRA vs CTRV vs CV 在路口场景的 TTC 精度 |
| 5 | FARR 误报压制 | 核心涨点指标，定量证明语义豁免的价值 |
| 6 | 语义噪声鲁棒性 | R3: 注入错误语义后系统自动退化到 Guard |
| 7 | 因果槽组消重 | 消融实验 D4 vs D1: ECE 校准度提升 |
| 8 | 显存封顶保证 | S3: VRAM-Stability Score |

---

## 参考文献

[1] The-DRIFT 官方仓库. https://github.com/AIxMobility/The-DRIFT  
[5] FHWA SSAM Report. https://www.fhwa.dot.gov/publications/research/safety/10020/  
[6] Wang et al. Surrogate safety review, 2021.  
[11] DriveLM. https://github.com/OpenDriveLab/DriveLM  
[12] Driving with LLMs. arXiv:2310.01957  
[13] LingoQA. ECCV 2024.  
[14] DASH benchmark. https://yanneu.github.io/DASH/  
[15] VisDrone-VID2019 小目标挑战. ICCVW 2019.  
[16] StreamingVLM. https://github.com/mit-han-lab/streaming-vlm  
[24] Outlines Structured Generation. https://dottxt-ai.github.io/outlines/  
[25] LM Format Enforcer. https://github.com/noamgat/lm-format-enforcer  
[43] Minderhoud & Bovy. Extended TTC measures. AAP 2001.  
[45] Salzmann et al. Trajectron++: Dynamically-feasible trajectory forecasting with heterogeneous data. ECCV 2020.  
[46] Zhou et al. HiVT: Hierarchical Vector Transformer for Multi-Agent Motion Prediction. CVPR 2022.  
[47] Yuan et al. AgentFormer: Agent-Aware Transformers for Socio-Temporal Multi-Agent Forecasting. ICCV 2021.  
[48] Dempster, A.P. Upper and lower probabilities induced by a multivalued mapping. The Annals of Mathematical Statistics, 38(2):325–339, 1967.  
[49] Shafer, G. A Mathematical Theory of Evidence. Princeton University Press, 1976.  
[50] Motion model engineering notes for Constant Turn Rate and Acceleration (CTRA) / Constant Turn Rate and Velocity (CTRV), automated tracking practice, accessed 2026.  
[51] Xiao et al. Efficient Streaming Language Models with Attention Sinks. arXiv:2309.17453.  
[52] LOOK-M. Multi-modal KV Cache Compression. arXiv:2406.18139.  
[53] Xu et al. SIND: A Drone Dataset at Signalized Intersection in China. ITSC 2022.  
[54] Yang et al. Set-of-Mark Prompting Unleashes Extraordinary Visual Grounding in GPT-4V. arXiv:2310.11441, 2023.  

---

> **文档版本**: v5.2  
> **核心修订摘要**:  
> 1. **数据流适配**: 新增 `SyncAligner`，统一 CSV 轨迹到 10Hz 工作频率  
> 2. **统一协议**: 基座全部收敛到 `BaseGuard` / `RiskMap` 标准接口  
> 3. **基座划分**: 明确 B0(CTRA)、B1(HiVT)、B2(Traj++) 三类消费模式  
> 4. **视觉增强**: 单帧 SoM + PiP 排版，支持 Single-Pass 多目标理解  
> 5. **融合机制**: 保留 CTRA + DST + Shafer 折扣 + 语义续命主线  
> 6. **指标体系**: Physical GT 采用未来 3s 回溯，Type-S 采用未来 5s 虚警逆挖  
> 7. **实验叙事**: 主打 `Base → Base + Plugin` 增量分析，而非绝对分数竞赛  
> 8. **核心宣称**: 验证“零参数物理底座 + VLM”与“模型无关插件增益”两条主线
