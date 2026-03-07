# 阶段 2: 流式推理引擎改造为导航模式

> **预计工期**: Day 3-5  
> **执行环境**: 云端 4090  
> **前置条件**: 阶段 1 完成，AirSim 通信链路已跑通  
> **目标**: 将现有 `VideoStreamingInference` 改造为导航专用的 `NavigationStreamEngine`

---

## 背景与设计思路

### 现有系统 vs 导航系统对比

| 维度 | 现有 VideoStreamingInference | 导航 NavigationStreamEngine |
|------|------------------------------|----------------------------|
| 交互模式 | 用户手动停下视频提问 | 系统自动判断何时响应 |
| 输入 | 视频帧 | 视频帧 + 位姿 + 路径信息 |
| 输出 | 自由文本回答 | JSON 结构化控制指令 |
| Prompt | 通用视频分析 | 导航专用 (含全局路径) |
| 触发方式 | 外部调用 ask() | 内部自动触发 (定时/事件驱动) |
| ask() 行为 | snapshot/restore (不污染 cache) | 同样 snapshot/restore |
| 生成长度 | 较长 (几十~几百 tokens) | 短 (JSON, ~30-50 tokens) |

### 核心设计: 不破坏现有代码

`NavigationStreamEngine` **继承** `VideoStreamingInference`，复用:
- `append_frame()`: 帧编码逻辑完全不变
- `ask()`: QA 逻辑完全不变 (snapshot/restore)
- `KVCacheManager` + `KVCacheEvictor`: 缓存管理不变
- `StreamQwenModel`: 3 分支 position 计算不变

新增:
- 导航专用 prompt 构建
- 自动触发机制
- 结构化输出解析
- 深度图障碍检测辅助

---

## 任务清单

### Task 2.1: NavigationStreamEngine 核心类

**文件: `navigation/local_vlm_navigator.py`**

```python
"""
NavigationStreamEngine — 基于流式 VLM 的无人机局部导航引擎

核心设计:
  1) 继承 VideoStreamingInference 的全部帧编码和 QA 能力
  2) 新增导航专用 prompt 模板 (含全局路径、当前位姿、下一航点)
  3) 新增自动触发机制 (定时 / 偏离 / 障碍)
  4) 新增结构化输出解析 (JSON → NavAction)
  5) 新增深度图辅助障碍检测

推理流程:
  初始化 → set_mission(global_path) → 循环:
    encode_and_check(frame, depth, position, yaw) →
      ├─ 编码帧 (append_frame)
      ├─ 更新位姿状态
      ├─ 检查触发条件
      └─ 如需响应: ask() → 解析 JSON → NavAction
"""

import json
import re
import math
import time
import numpy as np
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass, field
from PIL import Image

# 导入现有引擎
import sys
sys.path.insert(0, "d:/diploma_project")
from temporal_encoding.model.video_stream_inference import VideoStreamingInference
from temporal_encoding.model.kv_cache_eviction import EvictionConfig


@dataclass
class NavAction:
    """导航动作指令"""
    action: str = "forward"       # forward, left, right, forward_left, forward_right, stop, hover
    yaw_adjust: float = 0.0      # 偏航调整角度 (度, 正=右转, 负=左转)
    speed: float = 2.0            # 飞行速度 (m/s)
    altitude_adjust: float = 0.0  # 高度调整 (m, 正=上升, 负=下降)
    reasoning: str = ""           # 决策原因
    confidence: float = 0.5       # 置信度
    is_emergency: bool = False    # 是否紧急避障


@dataclass 
class NavState:
    """导航状态"""
    position: Tuple[float, float, float] = (0, 0, 0)
    yaw: float = 0.0
    next_waypoint: Tuple[float, float, float] = (0, 0, 0)
    waypoint_index: int = 0
    deviation: float = 0.0
    obstacle_distance: float = float('inf')
    frame_count: int = 0
    last_action_time: float = 0.0
    last_action: Optional[NavAction] = None
    mission_active: bool = False


class NavigationStreamEngine(VideoStreamingInference):
    """
    流式 VLM 导航引擎 — 在 VideoStreamingInference 基础上增加导航能力
    """
    
    # ── 导航系统 prompt ──
    NAV_SYSTEM_PROMPT = """You are a drone navigation assistant with real-time visual perception.

Your task: Safely navigate the drone to the target waypoint while avoiding obstacles.

RULES:
1. If the path ahead is clear and aligned with the target, output SAFE (no action needed).
2. If you see an obstacle ahead (building, tree, wall, etc.), output an avoidance maneuver.
3. If the drone is deviating from the path, output a correction.
4. Always prioritize safety — stop if unsure.

OUTPUT FORMAT (strict JSON, no extra text):
{"action":"forward|left|right|forward_left|forward_right|stop|hover|up|down","yaw":0,"speed":2.0,"reason":"brief explanation"}

- action: movement direction
- yaw: yaw adjustment in degrees (positive=right, negative=left), range [-45, 45]
- speed: flight speed in m/s, range [0, 5]
- reason: very brief explanation (max 15 words)"""

    def __init__(
        self,
        model,
        processor,
        device: str = "cuda",
        eviction_config: Optional[EvictionConfig] = None,
        # 导航参数
        check_interval: int = 5,          # 每 N 帧强制检查
        deviation_threshold: float = 3.0, # 偏离阈值 (米)
        obstacle_threshold: float = 5.0,  # 障碍物距离阈值 (米)
        waypoint_reach_dist: float = 3.0, # 到达航点距离 (米)
    ):
        # 调用父类构造函数
        super().__init__(model, processor, device, eviction_config)
        
        # 覆盖系统 prompt
        self.system_prompt = self.NAV_SYSTEM_PROMPT
        
        # 导航参数
        self.check_interval = check_interval
        self.deviation_threshold = deviation_threshold
        self.obstacle_threshold = obstacle_threshold
        self.waypoint_reach_dist = waypoint_reach_dist
        
        # 导航状态
        self.nav_state = NavState()
        self.global_path: List[Tuple[float, float, float]] = []
        self._nav_frame_counter = 0
        
        print("🚁 NavigationStreamEngine Initialized")

    # ── 任务设置 ──

    def set_mission(self, global_path: List[Tuple[float, float, float]]):
        """设置导航任务的全局路径"""
        self.global_path = global_path
        self.nav_state = NavState()
        self.nav_state.mission_active = True
        self.nav_state.next_waypoint = global_path[1] if len(global_path) > 1 else global_path[0]
        self.nav_state.waypoint_index = 1 if len(global_path) > 1 else 0
        self._nav_frame_counter = 0
        print(f"📍 Mission set: {len(global_path)} waypoints")
        print(f"   Start: {global_path[0]} → Goal: {global_path[-1]}")

    # ── 核心导航循环方法 ──

    def encode_and_check(
        self,
        frame: Image.Image,
        depth: Optional[np.ndarray] = None,
        position: Tuple[float, float, float] = (0, 0, 0),
        yaw: float = 0.0,
        as_video: bool = False,
        fps: Optional[float] = None,
    ) -> Optional[NavAction]:
        """
        编码一帧并检查是否需要导航响应。
        
        Returns:
            NavAction 如果需要动作，None 如果无需动作 (SAFE)
        """
        # 1) 编码帧 (调用父类)
        encode_result = self.append_frame(
            frame, 
            text_content="Navigation frame.", 
            as_video=as_video,
            fps=fps
        )
        
        # 2) 更新状态
        self.nav_state.position = position
        self.nav_state.yaw = yaw
        self.nav_state.frame_count += 1
        self._nav_frame_counter += 1
        
        # 3) 计算偏离度和障碍距离
        self.nav_state.deviation = self._calc_deviation(position)
        if depth is not None:
            self.nav_state.obstacle_distance = self._estimate_obstacle_distance(depth)
        
        # 4) 检查是否到达当前航点
        dist_to_wp = self._distance_3d(position, self.nav_state.next_waypoint)
        if dist_to_wp < self.waypoint_reach_dist:
            self._advance_waypoint()
        
        # 5) 判断是否需要触发 VLM 推理
        trigger_reason = self._check_trigger()
        if trigger_reason is None:
            return None  # 不需要动作
        
        # 6) 构造导航 prompt 并询问模型
        nav_prompt = self._build_nav_prompt(trigger_reason)
        
        try:
            response_text, metrics = self.ask(
                question=nav_prompt,
                max_new_tokens=80,
                min_new_tokens=5,
                do_sample=False,  # 导航决策用 greedy
            )
            
            # 7) 解析 JSON 响应
            action = self._parse_nav_response(response_text)
            action.reasoning = f"[{trigger_reason}] {action.reasoning}"
            
            self.nav_state.last_action = action
            self.nav_state.last_action_time = time.time()
            
            print(f"🧭 Nav Decision [{trigger_reason}]: {action.action}, "
                  f"yaw={action.yaw_adjust:.1f}°, speed={action.speed:.1f}m/s, "
                  f"reason={action.reasoning} "
                  f"(latency={metrics['total_latency']*1000:.0f}ms)")
            
            return action
            
        except Exception as e:
            print(f"⚠️ Navigation inference error: {e}")
            # 安全默认动作: 减速悬停
            return NavAction(action="hover", speed=0, reasoning=f"Inference error: {e}")

    # ── 触发条件检查 ──

    def _check_trigger(self) -> Optional[str]:
        """
        检查是否应该触发 VLM 导航推理。
        返回触发原因字符串，或 None (不触发)。
        """
        # 紧急: 障碍物很近
        if self.nav_state.obstacle_distance < self.obstacle_threshold:
            return "obstacle"
        
        # 偏离路径
        if self.nav_state.deviation > self.deviation_threshold:
            return "deviation"
        
        # 定时检查
        if self._nav_frame_counter % self.check_interval == 0:
            return "periodic"
        
        return None

    # ── Prompt 构建 ──

    def _build_nav_prompt(self, trigger_reason: str) -> str:
        """构建导航查询 prompt"""
        pos = self.nav_state.position
        wp = self.nav_state.next_waypoint
        
        # 计算方位信息
        dx = wp[0] - pos[0]
        dy = wp[1] - pos[1]
        dist = math.sqrt(dx**2 + dy**2)
        
        # 目标相对于当前朝向的角度
        target_bearing = math.degrees(math.atan2(dy, dx))
        relative_angle = target_bearing - self.nav_state.yaw
        # 归一化到 [-180, 180]
        while relative_angle > 180: relative_angle -= 360
        while relative_angle < -180: relative_angle += 360
        
        # 方位描述
        if abs(relative_angle) < 15:
            direction = "directly ahead"
        elif relative_angle > 0:
            direction = f"to the right ({relative_angle:.0f}°)"
        else:
            direction = f"to the left ({relative_angle:.0f}°)"
        
        prompt_parts = [
            f"Current position: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})",
            f"Current heading: {self.nav_state.yaw:.0f}°",
            f"Next waypoint: ({wp[0]:.1f}, {wp[1]:.1f}, {wp[2]:.1f}), distance: {dist:.1f}m, {direction}",
            f"Path deviation: {self.nav_state.deviation:.1f}m",
        ]
        
        if self.nav_state.obstacle_distance < 50:
            prompt_parts.append(f"Nearest obstacle: {self.nav_state.obstacle_distance:.1f}m ahead")
        
        if trigger_reason == "obstacle":
            prompt_parts.append("⚠️ OBSTACLE DETECTED — decide avoidance maneuver NOW.")
        elif trigger_reason == "deviation":
            prompt_parts.append("⚠️ PATH DEVIATION — correct course to reach the waypoint.")
        else:
            prompt_parts.append("Routine check: assess the scene and decide next action.")
        
        # 全局路径概要 (只给当前和后续 2 个航点)
        remaining = self.global_path[self.nav_state.waypoint_index:]
        if len(remaining) > 3:
            remaining = remaining[:3]
        path_str = " → ".join([f"({p[0]:.0f},{p[1]:.0f},{p[2]:.0f})" for p in remaining])
        prompt_parts.append(f"Remaining path: {path_str}")
        
        return "\n".join(prompt_parts)

    # ── 响应解析 ──

    def _parse_nav_response(self, response_text: str) -> NavAction:
        """解析 VLM 的 JSON 响应为 NavAction"""
        # 尝试提取 JSON
        json_match = re.search(r'\{[^{}]*\}', response_text)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return NavAction(
                    action=data.get("action", "hover"),
                    yaw_adjust=float(data.get("yaw", 0)),
                    speed=float(data.get("speed", 2.0)),
                    reasoning=data.get("reason", response_text[:50]),
                    confidence=0.8,
                )
            except (json.JSONDecodeError, ValueError):
                pass
        
        # JSON 解析失败 — 尝试关键词匹配
        text_lower = response_text.lower()
        if "safe" in text_lower or "clear" in text_lower:
            return NavAction(action="forward", speed=2.0, reasoning="Path clear")
        elif "left" in text_lower:
            return NavAction(action="left", yaw_adjust=-20, speed=1.5, reasoning="Turn left")
        elif "right" in text_lower:
            return NavAction(action="right", yaw_adjust=20, speed=1.5, reasoning="Turn right")
        elif "stop" in text_lower or "obstacle" in text_lower:
            return NavAction(action="hover", speed=0, reasoning="Obstacle, hover")
        
        # 兜底: 安全悬停
        return NavAction(
            action="hover", speed=0, 
            reasoning=f"Unparseable response: {response_text[:30]}", 
            confidence=0.2
        )

    # ── 辅助计算 ──

    def _calc_deviation(self, position: Tuple[float, float, float]) -> float:
        """计算当前位置到全局路径的最小距离 (2D)"""
        if len(self.global_path) < 2:
            return 0.0
        
        min_dist = float('inf')
        px, py = position[0], position[1]
        
        for i in range(len(self.global_path) - 1):
            ax, ay = self.global_path[i][0], self.global_path[i][1]
            bx, by = self.global_path[i+1][0], self.global_path[i+1][1]
            
            # 点到线段距离
            dist = self._point_to_segment_dist(px, py, ax, ay, bx, by)
            min_dist = min(min_dist, dist)
        
        return min_dist
    
    @staticmethod
    def _point_to_segment_dist(px, py, ax, ay, bx, by) -> float:
        """计算点 (px,py) 到线段 (ax,ay)-(bx,by) 的距离"""
        dx, dy = bx - ax, by - ay
        if dx == 0 and dy == 0:
            return math.sqrt((px - ax)**2 + (py - ay)**2)
        
        t = max(0, min(1, ((px - ax) * dx + (py - ay) * dy) / (dx**2 + dy**2)))
        proj_x = ax + t * dx
        proj_y = ay + t * dy
        return math.sqrt((px - proj_x)**2 + (py - proj_y)**2)
    
    @staticmethod
    def _distance_3d(a, b) -> float:
        return math.sqrt(sum((ai - bi)**2 for ai, bi in zip(a, b)))
    
    def _estimate_obstacle_distance(self, depth: np.ndarray) -> float:
        """从深度图估计前方最近障碍物距离"""
        h, w = depth.shape
        # 取图像中心 1/3 区域的最小深度
        cy, cx = h // 2, w // 2
        rh, rw = h // 6, w // 6
        center_region = depth[cy-rh:cy+rh, cx-rw:cx+rw]
        
        # 过滤无效值
        valid = center_region[(center_region > 0.1) & (center_region < 100)]
        if len(valid) == 0:
            return float('inf')
        
        # 取最近的 5% 分位数 (避免噪声)
        return float(np.percentile(valid, 5))
    
    def _advance_waypoint(self):
        """前进到下一个航点"""
        if self.nav_state.waypoint_index < len(self.global_path) - 1:
            self.nav_state.waypoint_index += 1
            self.nav_state.next_waypoint = self.global_path[self.nav_state.waypoint_index]
            print(f"✅ Waypoint reached! Advancing to waypoint {self.nav_state.waypoint_index}: "
                  f"{self.nav_state.next_waypoint}")
        else:
            self.nav_state.mission_active = False
            print("🏁 Mission Complete! All waypoints reached.")
    
    @property
    def mission_complete(self) -> bool:
        return not self.nav_state.mission_active
    
    def reset_navigation(self):
        """重置导航状态 (同时重置引擎)"""
        self.reset()  # 父类 reset
        self.nav_state = NavState()
        self.global_path = []
        self._nav_frame_counter = 0
        print("🔄 Navigation Reset")
```

实现要点说明:
1. **不修改现有代码**: `NavigationStreamEngine` 继承 `VideoStreamingInference`，所有帧编码、KV Cache 管理、ask/restore 机制完全复用
2. **系统 prompt 覆盖**: 在 `__init__` 中将 `self.system_prompt` 替换为导航专用 prompt
3. **`encode_and_check()`**: 核心方法，每帧调用，自动决定是否触发 VLM 推理
4. **触发条件三重**: obstacle (紧急) > deviation (纠偏) > periodic (定时巡检)
5. **JSON 解析容错**: 正则提取 JSON + 关键词兜底 + 安全默认动作

---

### Task 2.2: 导航 prompt 优化与测试

**文件: `navigation/scripts/test_nav_engine.py`**

编写测试脚本，使用静态图片模拟导航场景:

```python
"""
测试 NavigationStreamEngine 在静态场景下的行为。
不需要 AirSim，用本地图片模拟。
"""

def test_nav_engine():
    # 1. 加载模型
    from temporal_encoding.model.stream_qwen_model import StreamQwenModel
    from transformers import AutoProcessor
    
    model = StreamQwenModel.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="cuda"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    
    engine = NavigationStreamEngine(
        model=model,
        processor=processor,
        device="cuda",
        eviction_config=EvictionConfig(max_cache_tokens=50000),
        check_interval=3,
    )
    
    # 2. 设置任务
    engine.set_mission([
        (0, 0, -5),
        (20, 0, -5),
        (40, 10, -5),
        (50, 10, -5),
    ])
    
    # 3. 模拟帧输入 (用纯色图片或实际场景图片)
    from PIL import Image
    
    # 场景1: 空旷道路 → 应输出 SAFE/forward
    clear_road = Image.new('RGB', (384, 384), color=(135, 206, 235))  # 天蓝色
    action = engine.encode_and_check(
        frame=clear_road,
        position=(5, 0.5, -5),
        yaw=0,
    )
    print(f"Scene 1 (clear road): {action}")
    
    # 场景2: 前方有障碍 → 应输出避障
    # (用深度图模拟近距离障碍)
    depth_obstacle = np.full((384, 384), 2.0, dtype=np.float32)  # 2m 处有障碍
    action = engine.encode_and_check(
        frame=clear_road,
        depth=depth_obstacle,
        position=(10, 0, -5),
        yaw=0,
    )
    print(f"Scene 2 (obstacle): {action}")
    
    # 场景3: 偏离路径 → 应输出纠偏
    action = engine.encode_and_check(
        frame=clear_road,
        position=(15, 5, -5),  # 偏离主路径 5m
        yaw=0,
    )
    print(f"Scene 3 (deviation): {action}")

if __name__ == "__main__":
    test_nav_engine()
```

测试要验证:
- 空旷场景: 模型输出 forward / SAFE
- 障碍场景: 模型输出 left/right 避让
- 偏离场景: 模型输出纠偏指令
- JSON 格式稳定可解析
- 单次推理延迟 < 1 秒

---

### Task 2.3: 更新通信服务器以使用 NavigationStreamEngine

更新 `navigation/communication/server.py`:

将服务器中的 `VideoStreamingInference` 替换为 `NavigationStreamEngine`。

关键改动:
1. `/api/init_mission` 端点调用 `engine.set_mission(global_path)`
2. `/api/encode_frame` 调用 `engine.append_frame()` (不变)
3. `/api/navigate` 端点改为调用 `engine.encode_and_check()`
   - 接收图像 + 深度 + 位姿
   - 内部自动判断是否触发 VLM
   - 返回 NavAction 或 None
4. 新增 `/api/encode_and_navigate` 合并端点 (减少网络往返):
   - 同时编码帧 + 检查导航
   - 返回 {encoded: True, action: NavAction|null}

---

## 验收标准

- [ ] `NavigationStreamEngine` 能正常初始化和加载模型
- [ ] `set_mission()` 正确设置全局路径
- [ ] `encode_and_check()` 能编码帧并在正确条件下触发推理
- [ ] VLM 输出的 JSON 能被正确解析为 `NavAction`
- [ ] 解析失败时有合理的兜底策略
- [ ] 偏离检测算法正确计算点到路径距离
- [ ] 深度图障碍检测能从深度矩阵中提取前方最近距离
- [ ] 航点到达检测和自动前进工作正常
- [ ] 单帧编码+检查延迟 < 100ms (不含生成)
- [ ] 编码+生成总延迟 < 1 秒

---

## 注意事项

1. **不要修改 `temporal_encoding/model/` 下的任何现有文件** — 所有改动在 `navigation/` 目录
2. JSON 输出不稳定是 3B 模型的常见问题，必须有强健的解析容错
3. 导航 prompt 中的坐标精度保留 1 位小数即可，过多数字会浪费 token
4. `ask()` 的 `max_new_tokens` 设为 80 足够生成一个 JSON 对象
5. `do_sample=False` (greedy) 在导航决策中更稳定
6. 深度图的无效值 (0 或 inf) 需要过滤
