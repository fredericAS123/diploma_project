# 🚁 流式 VLM 无人机导航 — 总体方案与实施计划

> **日期**: 2026-03-07  
> **目标**: 半个月内在 AirSim 仿真中实现基于流式 Qwen2.5-VL-3B 的双层规划无人机导航 Demo  
> **硬件条件**: 本地 PC (AirSim) + 云端 4090 (模型推理) + A800 (少量微调)

---

## 一、技术架构总览

### 1.1 双层规划架构

```
┌──────────────────────────────────────────────────────────┐
│                    AirSim 仿真环境                         │
│       (Unreal Engine, Multirotor, NED 坐标系)              │
└────────┬──────────────────┬──────────────────┬───────────┘
         │ RGB帧流 (2-5 FPS) │ 深度图          │ 位姿/碰撞
         ▼                  ▼                  ▼
┌────────────────────────────────────────────────────────────┐
│                  AirSim 桥接层 (本地 PC)                     │
│  • 帧采集 + 网络传输 (gRPC/WebSocket)                       │
│  • 控制指令执行 (moveToPositionAsync / moveByVelocity)      │
│  • 碰撞检测 + 状态回传                                      │
└────────┬──────────────────────────────────────┬────────────┘
         │ 压缩帧+位姿 (网络)                     │ 控制指令
         ▼                                      ▲
┌────────────────────────────────────────────────────────────┐
│               云端 4090 推理服务器                            │
│                                                            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │        上层: 全局路径规划 (Global Planner)              │  │
│  │  • 输入: 起点 + 终点 + 3D 障碍物地图                    │  │
│  │  • 算法: A* / RRT* (离散化 3D 网格)                    │  │
│  │  • 输出: 全局航点序列 [(x,y,z), ...]                   │  │
│  │  • 频率: 一次性规划 / 偏离较大时重规划                    │  │
│  └────────────────────┬─────────────────────────────────┘  │
│                       │ 全局航点 + 下一个目标点               │
│                       ▼                                    │
│  ┌──────────────────────────────────────────────────────┐  │
│  │      下层: 流式 VLM 局部决策 (Streaming Local)         │  │
│  │  • 模型: Qwen2.5-VL-3B (流式滑动窗口改造)              │  │
│  │  • 输入: 视频帧流 + 系统提示(含全局路径) + 位姿          │  │
│  │  • 推理模式: 主动响应 (每帧编码 + 条件性生成)            │  │
│  │  • 输出: JSON 结构化指令                               │  │
│  │    {action, yaw_adjust, speed, reasoning}              │  │
│  │  • 频率: 2-5 FPS 编码, 按需生成 (~0.5-2s/次)           │  │
│  └────────────────────┬─────────────────────────────────┘  │
│                       │ 控制指令                            │
└───────────────────────┼────────────────────────────────────┘
                        │ (网络回传)
                        ▼
              AirSim 执行 moveByVelocity
```

### 1.2 核心问题解答

#### Q1: 如何全局规划路径？以什么形式告诉流式模型？

**全局路径规划**:
- 使用 AirSim 的 `simGetMeshPositionVertexCounts()` 或预定义障碍物位置构建 3D 占位栅格
- 在栅格上运行 A* 算法得到航点序列
- 对于 Demo 阶段: 直接手动指定起点/终点/中间航点即可，不需要完美的全局规划

**告诉流式模型的方式** (参考 LMDrive + DriveVLM):
```
System Prompt:
你是无人机导航助手。你的任务是安全飞向目标点并避开障碍物。

任务路径: 从(0,0,-5)出发 → 经过(20,0,-5) → 到达(40,10,-5)
当前位置: (12.3, 1.5, -5.0), 朝向: 北偏东15°
下一个航点: (20.0, 0.0, -5.0), 距离8.2m, 方位: 右前方12°

规则:
1. 前方无障碍时输出 SAFE，不需要额外指令
2. 发现偏离路径(>3m)时输出纠偏指令
3. 检测到障碍物时输出避障指令
4. 输出格式: {"action":"forward|left|right|up|down|stop","yaw":角度,"speed":m/s,"reason":"..."}
```

#### Q2: 如何理解"偏离路径"？如何让模型在合适时间反应？

**偏离检测 — 双重机制**:
1. **程序化检测 (主要)**: 计算当前位置到全局路径线段的垂直距离，超过阈值(如3m)时在 prompt 中标注
2. **视觉检测 (辅助)**: 模型通过视觉判断前方是否有障碍需要绕行

**主动响应机制** (参考 VideoLLM-online 的 Streaming EOS):
- **方案 A (推荐 Demo 阶段)**: 定时检查制 — 每 N 帧(如每5帧=1-2秒)强制询问一次模型
- **方案 B (进阶)**: 实现类似 VideoLLM-online 的"条件生成" — 每帧编码后检查 logits，只在模型想说话时生成
- **方案 C**: 混合制 — 程序化检测+阈值触发模型生成

**推荐 Demo 阶段方案**: 方案 A + 程序化偏离检测
```python
# 伪代码
for frame in video_stream:
    engine.append_frame(frame)  # 始终编码
    
    deviation = calc_path_deviation(current_pos, global_path)
    obstacle_near = check_depth_obstacle(depth_image)
    
    if deviation > 3.0 or obstacle_near or frame_count % 5 == 0:
        prompt = build_nav_prompt(current_pos, next_waypoint, deviation)
        action, _ = engine.ask(prompt, max_new_tokens=50)
        execute_action(action)
    else:
        # 沿当前方向继续飞行
        continue_current_trajectory()
```

#### Q3: 全局和局部如何协作？

```
全局规划器                              局部 VLM 决策
    │                                      │
    │── 初始规划: 全局航点序列 ──────────────→│ 接收全局路径
    │                                      │
    │                                      │← 帧流编码 (持续)
    │                                      │
    │                                      │── 检查偏离/障碍
    │                                      │   ├─ 偏离<3m → SAFE, 继续
    │                                      │   ├─ 偏离>3m → 生成纠偏指令
    │                                      │   └─ 障碍 → 生成避障指令
    │                                      │
    │← (偏离>10m 或到达中间点) ─────────────│ 请求重规划
    │── 重新规划剩余路径 ──────────────────→│ 更新全局路径
    │                                      │
    │← 到达终点 ────────────────────────────│ 任务完成
```

协作规则:
1. **全局→局部**: 全局规划器提供航点序列，局部 VLM 只关注"当前位置→下一航点"
2. **局部→全局**: 当偏离过大(>10m)或到达航点时，触发全局重规划
3. **实时路径跟踪**: 当 VLM 的避障绕行导致偏离原路径，全局规划器在绕行完成后重新规划剩余路径

#### Q4: 推理速度分析

| 模块 | 延迟 (4090) | 说明 |
|------|------------|------|
| AirSim 取帧 (384×384) | ~10ms | 未压缩 RGB + 深度 |
| 网络传输 (局域网) | ~5-15ms | JPEG 压缩后 ~50KB/帧 |
| ViT 编码 | ~10-15ms | 固定分辨率 384×384 |
| LLM 增量前向 (KV Cache) | ~15-25ms | 仅新帧 tokens |
| LLM 生成 (50 tokens) | ~250-500ms | 仅在需要时触发 |
| **总计 (仅编码)** | **~40-65ms → 15-25 FPS** | |
| **总计 (含生成)** | **~300-560ms → 2-3 FPS** | |

**结论**: 编码+检测 15+ FPS 完全可行; 生成指令 2-3 FPS 也足够（避障/纠偏不需要极高频率）

---

## 二、关键开源项目参考

| 项目 | 用途 | GitHub |
|------|------|--------|
| **PromptCraft-Robotics** | AirSim 封装层 (直接复用) | microsoft/PromptCraft-Robotics |
| **VideoLLM-online** | 主动响应机制 (Streaming EOS) | showlab/videollm-online |
| **LiveCC** | Qwen2-VL 流式理解训练 | showlab/livecc |
| **StreamingVLM** | KV Cache 管理参考 (已参考) | mit-han-lab/streaming-vlm |
| **LMDrive** | 端到端导航框架 (航点预测+PID) | opendilab/LMDrive |
| **DriveVLM** | CoT 分层规划思路 | tsinghua-mars-lab/DriveVLM |

---

## 三、15天实施计划 (分 5 个阶段)

### 阶段 1: AirSim 环境搭建 + 基础通信 (Day 1-2)
**目标**: 本地 AirSim 能跑无人机，云端能接收帧并返回指令

- [ ] 本地安装 AirSim + Unreal Engine (或使用预编译二进制)
- [ ] 配置 AirSim settings.json (384×384, 90° FOV)
- [ ] 编写 AirSim Python 客户端: 取帧 + 飞行控制 + 碰撞检测
- [ ] 搭建本地↔云端通信 (gRPC 或 Flask REST API)
- [ ] 验证: 本地帧能到达云端，云端指令能控制本地无人机

### 阶段 2: 流式推理引擎改造为导航模式 (Day 3-5)
**目标**: 改造现有 VideoStreamingInference 为 NavigationStreamEngine

核心改造点:
- [ ] 新增 `NavigationStreamEngine` 类 (继承或重构 `VideoStreamingInference`)
- [ ] 新增"主动检测循环" — 每帧编码后检查是否需要生成
- [ ] 系统 prompt 改为导航专用
- [ ] 输出格式改为 JSON 结构化指令
- [ ] 集成全局路径信息到 prompt
- [ ] 集成程序化偏离检测 + 深度障碍检测

### 阶段 3: 全局路径规划 + 双层协作 (Day 6-8)
**目标**: 全局 A* + 局部 VLM 协作运转

- [ ] 实现 3D A* / 简化网格路径规划
- [ ] 实现路径跟踪 (当前位置→最近路径点→下一航点)
- [ ] 实现偏离检测 (点到线段距离)
- [ ] 实现全局-局部协作状态机
- [ ] 实现航点到达检测 + 自动切换下一航点

### 阶段 4: 端到端集成测试 + 初步 Demo (Day 9-11)
**目标**: 在 AirSim 中完成一次从起点到终点的导航

- [ ] 集成所有模块: AirSim桥接 + 全局规划 + 流式VLM + 控制执行
- [ ] 选择/构建简单测试场景 (几栋楼 + 几棵树)
- [ ] 调试端到端流程: 起飞 → 全局规划 → 流式编码 → 避障 → 到达
- [ ] 记录 Demo 视频
- [ ] 基础指标收集: 成功率, 碰撞率, 路径偏离

### 阶段 5: 优化与增强 (Day 12-15)
**目标**: 提升质量、准备更复杂场景

- [ ] (可选) 在 A800 上用 LoRA 微调导航专用模型
- [ ] 优化延迟: 减少不必要的模型调用
- [ ] 添加更复杂场景测试
- [ ] 实现"主动响应"进阶模式 (参考 VideoLLM-online)
- [ ] 完善 Demo 录制 + 数据可视化

---

## 四、文件结构规划

```
diploma_project/
├── navigation/                          # 新增导航模块
│   ├── __init__.py
│   ├── airsim_bridge.py                 # AirSim 客户端封装
│   ├── global_planner.py                # 全局 A* 路径规划
│   ├── local_vlm_navigator.py           # 流式 VLM 局部导航引擎
│   ├── nav_state_machine.py             # 全局-局部协作状态机
│   ├── communication/
│   │   ├── __init__.py
│   │   ├── server.py                    # 云端推理服务 (Flask/gRPC)
│   │   └── client.py                    # 本地 AirSim 端客户端
│   ├── config/
│   │   ├── airsim_settings.json         # AirSim 配置
│   │   └── nav_config.yaml              # 导航参数配置
│   └── scripts/
│       ├── run_airsim_client.py          # 启动本地端
│       ├── run_inference_server.py       # 启动云端推理
│       └── run_demo.py                  # 端到端 Demo
├── temporal_encoding/                   # 现有代码 (保持不变)
│   └── model/
│       ├── stream_qwen_model.py
│       ├── video_stream_inference.py
│       ├── cache_manager.py
│       └── kv_cache_eviction.py
└── prompts/                             # 分步提示词文档
    ├── STEP1_AIRSIM_SETUP.md
    ├── STEP2_NAV_ENGINE.md
    ├── STEP3_GLOBAL_PLANNER.md
    ├── STEP4_INTEGRATION.md
    └── STEP5_OPTIMIZATION.md
```

---

## 五、关键设计决策

### 5.1 Demo 阶段简化策略 (重要!)

为了在 15 天内完成，做以下关键简化:

1. **全局路径**: 手动指定航点序列，不做自动地图构建
2. **通信**: 如果本地和云端在同一局域网，用 Flask REST API (最简单)
3. **避障**: 深度图阈值检测 + VLM 视觉确认，不做精确 3D 重建
4. **控制**: 直接 `moveToPositionAsync` + 速度控制，不做复杂 PID
5. **模型**: 直接用 Qwen2.5-VL-3B-Instruct 原始模型 + 精心设计的 prompt，不做微调
6. **帧率**: 2 FPS 采集，每 5 帧(2.5秒)触发一次 VLM 判断
7. **场景**: AirSim 自带的 Blocks/Neighborhood 环境即可

### 5.2 为什么不直接用纯程序化避障?

本项目的核心学术价值在于:
1. **流式视频理解**: 利用 KV Cache 滑动窗口实现无限视频流处理 (已完成的工作)
2. **时序理解优势**: VLM 能理解"正在靠近障碍物"而非仅"当前有障碍物" — 这是图片模型做不到的
3. **自然语言交互**: 可以用自然语言改变任务目标、查询状态
4. **泛化能力**: 不需要为每种障碍物编写规则，VLM 自行判断

### 5.3 进阶优化路线图 (Demo 之后)

1. **主动响应训练**: 在 A800 上用 LiveCC 的训练方案，用 AirSim 数据微调模型学会"何时该说话"
2. **视觉路径叠加**: 在图像上叠加全局路径投影线，让模型直观理解路径
3. **多传感器融合**: RGB + 深度 + IMU 联合输入
4. **在线学习**: 从导航经验中收集数据，持续微调
5. **Sim2Real**: 从仿真迁移到真实无人机

---

## 六、风险与缓解

| 风险 | 概率 | 影响 | 缓解策略 |
|------|------|------|---------|
| 3B 模型避障判断不准 | 高 | 中 | 深度图程序化兜底 + VLM 仅做辅助确认 |
| 网络延迟过大 (>200ms) | 中 | 高 | 1) 降低判断频率 2) 本地简单避障优先 3) 异步指令 |
| AirSim 环境配置困难 | 中 | 中 | 使用预编译二进制 + Blocks 环境 |
| 模型输出格式不稳定 | 高 | 中 | 正则解析 + 默认安全动作兜底 |
| 15天时间不够 | 中 | 高 | 优先保证最简 Demo (直线飞行+简单避障) |

---

## 七、参考文献

1. StreamingVLM (MIT-HAN-Lab, arXiv:2510.09608) — KV Cache 管理 + 流式推理
2. VideoLLM-online (ShowLab, CVPR 2024) — 主动响应 Streaming EOS 机制
3. LiveCC (ShowLab, CVPR 2025) — Qwen2-VL 流式实时评论训练
4. LMDrive (OpenDILab, CVPR 2024) — 端到端 VLM 驾驶，航点预测框架
5. DriveVLM (Tsinghua MARS, CoRL 2024) — CoT 分层规划
6. PromptCraft-Robotics (Microsoft) — LLM + AirSim 无人机控制
7. StreamingLLM (arXiv:2309.17453) — Attention Sink + Sliding Window
