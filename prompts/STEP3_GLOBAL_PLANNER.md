# 阶段 3: 全局路径规划 + 双层协作状态机

> **预计工期**: Day 6-8  
> **执行环境**: 云端 4090 (规划算法) + 本地 PC (AirSim 验证)  
> **前置条件**: 阶段 2 完成，NavigationStreamEngine 可用  
> **目标**: 实现 3D A* 全局规划 + 全局-局部协作状态机

---

## 任务清单

### Task 3.1: 3D 全局路径规划器

**文件: `navigation/global_planner.py`**

功能要求:

1. `GlobalPlanner` 类，实现 3D 空间的 A* 路径规划
2. 设计思路:
   - 将 3D 空间离散化为均匀栅格 (分辨率可配, 默认 1m)
   - 支持从 AirSim 获取障碍物信息构建占位图
   - A* 搜索找到起点到终点的最优路径
   - 路径简化 (删除共线中间点)
   - 支持动态重规划

3. 关键方法:

```python
class GlobalPlanner:
    def __init__(self, grid_resolution=1.0, search_space=None):
        """
        Args:
            grid_resolution: 栅格分辨率 (米)
            search_space: 搜索空间范围 {"x": (-50, 100), "y": (-50, 100), "z": (-20, 0)}
        """
        pass
    
    def set_obstacles_from_airsim(self, client):
        """从 AirSim 获取场景障碍物信息"""
        # 方法1 (简单): 预定义障碍物位置列表
        # 方法2 (进阶): 使用 simListSceneObjects + simGetObjectPose 获取
        pass
    
    def set_obstacles_manual(self, obstacles: List[Dict]):
        """手动设置障碍物
        Args:
            obstacles: [{"center": (x,y,z), "size": (sx,sy,sz)}, ...]
        """
        pass
    
    def plan(self, start: Tuple, goal: Tuple) -> List[Tuple]:
        """
        A* 搜索规划全局路径
        Returns: 航点列表 [(x,y,z), ...]
        """
        pass
    
    def replan(self, current_pos: Tuple, goal: Tuple) -> List[Tuple]:
        """从当前位置重新规划到目标"""
        pass
    
    def simplify_path(self, path: List[Tuple]) -> List[Tuple]:
        """简化路径 - 删除共线点"""
        pass
```

4. A* 实现细节:

```python
import heapq
import numpy as np
from typing import List, Tuple, Dict, Set

class GlobalPlanner:
    def __init__(self, grid_resolution=1.0, search_space=None):
        self.resolution = grid_resolution
        self.search_space = search_space or {
            "x": (-50, 100), "y": (-50, 100), "z": (-15, 0)
        }
        self.obstacles: Set[Tuple[int, int, int]] = set()  # 栅格坐标的障碍物集合
        
    def _world_to_grid(self, pos):
        """世界坐标 → 栅格坐标"""
        return (
            int(round(pos[0] / self.resolution)),
            int(round(pos[1] / self.resolution)),
            int(round(pos[2] / self.resolution)),
        )
    
    def _grid_to_world(self, grid_pos):
        """栅格坐标 → 世界坐标"""
        return (
            grid_pos[0] * self.resolution,
            grid_pos[1] * self.resolution,
            grid_pos[2] * self.resolution,
        )
    
    def _is_valid(self, pos):
        """检查栅格位置是否合法"""
        x, y, z = pos
        sp = self.search_space
        gx_min, gx_max = int(sp["x"][0]/self.resolution), int(sp["x"][1]/self.resolution)
        gy_min, gy_max = int(sp["y"][0]/self.resolution), int(sp["y"][1]/self.resolution)
        gz_min, gz_max = int(sp["z"][0]/self.resolution), int(sp["z"][1]/self.resolution)
        
        if not (gx_min <= x <= gx_max and gy_min <= y <= gy_max and gz_min <= z <= gz_max):
            return False
        return pos not in self.obstacles
    
    def _heuristic(self, a, b):
        """3D 欧几里得距离启发式"""
        return math.sqrt(sum((ai-bi)**2 for ai, bi in zip(a, b)))
    
    def _get_neighbors(self, pos):
        """获取 26 邻域 (3D)，或 6 邻域 (简化)"""
        x, y, z = pos
        # 使用 6 邻域 (上下左右前后) 加 12 对角方向 = 18 邻域
        deltas = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    deltas.append((dx, dy, dz))
        
        neighbors = []
        for dx, dy, dz in deltas:
            nb = (x+dx, y+dy, z+dz)
            if self._is_valid(nb):
                cost = math.sqrt(dx**2 + dy**2 + dz**2)
                neighbors.append((nb, cost))
        return neighbors
    
    def plan(self, start, goal):
        """A* 3D 路径搜索"""
        start_grid = self._world_to_grid(start)
        goal_grid = self._world_to_grid(goal)
        
        open_set = [(0, start_grid)]
        came_from = {}
        g_score = {start_grid: 0}
        
        while open_set:
            _, current = heapq.heappop(open_set)
            
            if current == goal_grid:
                # 重建路径
                path = [self._grid_to_world(current)]
                while current in came_from:
                    current = came_from[current]
                    path.append(self._grid_to_world(current))
                path.reverse()
                return self.simplify_path(path)
            
            for neighbor, move_cost in self._get_neighbors(current):
                tentative_g = g_score[current] + move_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self._heuristic(neighbor, goal_grid)
                    heapq.heappush(open_set, (f, neighbor))
        
        print("⚠️ No path found!")
        return [start, goal]  # 兜底: 直接连线
    
    def simplify_path(self, path):
        """Douglas-Peucker 路径简化"""
        if len(path) <= 2:
            return path
        # 简化版: 删除共线点
        simplified = [path[0]]
        for i in range(1, len(path) - 1):
            prev = simplified[-1]
            next_pt = path[i + 1]
            curr = path[i]
            # 如果 curr 不在 prev-next 连线上 (距离 > 阈值), 保留
            dist = self._point_to_line_3d(curr, prev, next_pt)
            if dist > self.resolution * 0.5:
                simplified.append(curr)
        simplified.append(path[-1])
        return simplified
    
    @staticmethod
    def _point_to_line_3d(point, line_start, line_end):
        """点到 3D 线段的距离"""
        p = np.array(point)
        a = np.array(line_start)
        b = np.array(line_end)
        ab = b - a
        if np.linalg.norm(ab) < 1e-9:
            return np.linalg.norm(p - a)
        t = np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0, 1)
        projection = a + t * ab
        return np.linalg.norm(p - projection)
    
    def set_obstacles_manual(self, obstacles):
        """添加长方体障碍物"""
        self.obstacles.clear()
        for obs in obstacles:
            cx, cy, cz = obs["center"]
            sx, sy, sz = obs["size"]
            for x in np.arange(cx - sx/2, cx + sx/2, self.resolution):
                for y in np.arange(cy - sy/2, cy + sy/2, self.resolution):
                    for z in np.arange(cz - sz/2, cz + sz/2, self.resolution):
                        self.obstacles.add(self._world_to_grid((x, y, z)))
        print(f"📦 Loaded {len(self.obstacles)} obstacle cells from {len(obstacles)} objects")
```

**Demo 阶段简化**: 如果 A* 计算太慢，直接用手动指定的航点序列:

```python
# 简化版全局规划 — 直接手动给定航点
class SimpleGlobalPlanner:
    def __init__(self):
        self.path = []
    
    def plan_manual(self, waypoints: List[Tuple]) -> List[Tuple]:
        """手动指定航点序列作为全局路径"""
        self.path = waypoints
        return self.path
    
    def replan_from(self, current_pos, remaining_path):
        """从当前位置接续剩余路径"""
        return [current_pos] + remaining_path
```

---

### Task 3.2: 全局-局部协作状态机

**文件: `navigation/nav_state_machine.py`**

状态机定义:

```
┌─────────┐
│  IDLE   │ ← 初始状态 / 任务完成
└────┬────┘
     │ start_mission()
     ▼
┌─────────┐
│PLANNING │ ← 全局路径规划中
└────┬────┘
     │ path_ready
     ▼
┌─────────┐      obstacle/deviation       ┌──────────┐
│CRUISING │ ──────────────────────────────→│ REACTING │
│(巡航中) │ ←──────────────────────────────│ (响应中)  │
└────┬────┘      action_executed           └──────────┘
     │ waypoint_reached                         │
     ▼                                          │ major_deviation
┌─────────┐                                     │
│ADVANCING│ ← 切换到下一航点                      │
└────┬────┘                                     │
     │                                          ▼
     │                                    ┌──────────┐
     │                                    │REPLANNING│
     │                                    │ (重规划)  │
     │                                    └────┬─────┘
     │                                         │
     ▼                                         │
┌─────────┐                                    │
│COMPLETED│ ← 所有航点到达          path_ready ──┘
└─────────┘
```

```python
"""
NavigationStateMachine — 全局-局部协作状态机

管理全局路径规划器和局部 VLM 导航引擎之间的协作关系。
"""

from enum import Enum
from dataclasses import dataclass
import time

class NavPhase(Enum):
    IDLE = "idle"
    PLANNING = "planning"
    CRUISING = "cruising"       # 正常巡航 (只编码, 定时检查)
    REACTING = "reacting"       # VLM 正在生成响应
    REPLANNING = "replanning"   # 全局重规划
    ADVANCING = "advancing"     # 切换到下一航点
    COMPLETED = "completed"
    EMERGENCY = "emergency"     # 紧急停止

class NavigationStateMachine:
    def __init__(
        self,
        global_planner,       # GlobalPlanner 或 SimpleGlobalPlanner
        local_navigator,      # NavigationStreamEngine
        drone_controller,     # DroneController (或 InferenceClient)
        replan_deviation: float = 10.0,   # 偏离多远触发重规划
        emergency_distance: float = 1.5,  # 紧急停止距离
    ):
        self.planner = global_planner
        self.navigator = local_navigator
        self.drone = drone_controller
        self.replan_deviation = replan_deviation
        self.emergency_distance = emergency_distance
        
        self.phase = NavPhase.IDLE
        self.global_path = []
        self.start = None
        self.goal = None
        
        # 统计
        self.stats = {
            "frames_encoded": 0,
            "vlm_calls": 0,
            "replans": 0,
            "collisions": 0,
            "waypoints_reached": 0,
            "start_time": 0,
        }
    
    def start_mission(self, start, goal, manual_path=None):
        """启动导航任务"""
        self.start = start
        self.goal = goal
        self.stats["start_time"] = time.time()
        
        print(f"🚀 Mission Start: {start} → {goal}")
        
        # 全局规划
        self.phase = NavPhase.PLANNING
        if manual_path:
            self.global_path = manual_path
        else:
            self.global_path = self.planner.plan(start, goal)
        
        print(f"📍 Global path: {len(self.global_path)} waypoints")
        for i, wp in enumerate(self.global_path):
            print(f"   WP{i}: ({wp[0]:.1f}, {wp[1]:.1f}, {wp[2]:.1f})")
        
        # 设置局部导航器
        self.navigator.set_mission(self.global_path)
        
        self.phase = NavPhase.CRUISING
        return self.global_path
    
    def step(self, frame, depth=None, position=None, yaw=0.0):
        """
        主循环的一步。每帧调用一次。
        
        Returns:
            dict: {
                "phase": NavPhase,
                "action": NavAction or None,
                "should_move": bool,
                "target_position": (x,y,z) or None,
            }
        """
        result = {
            "phase": self.phase,
            "action": None,
            "should_move": False,
            "target_position": None,
        }
        
        if self.phase == NavPhase.COMPLETED or self.phase == NavPhase.IDLE:
            return result
        
        if self.phase == NavPhase.EMERGENCY:
            # 紧急状态: 等待手动解除
            return result
        
        self.stats["frames_encoded"] += 1
        
        # ── 紧急碰撞检测 (程序化, 不经过 VLM) ──
        if depth is not None:
            min_depth = self._get_front_min_depth(depth)
            if min_depth < self.emergency_distance:
                self.phase = NavPhase.EMERGENCY
                result["phase"] = NavPhase.EMERGENCY
                result["action"] = NavAction(
                    action="stop", speed=0, 
                    reasoning="EMERGENCY: Too close to obstacle",
                    is_emergency=True
                )
                self.stats["collisions"] += 1
                print(f"🚨 EMERGENCY STOP: Obstacle at {min_depth:.1f}m")
                return result
        
        # ── 检查任务完成 ──
        if self.navigator.mission_complete:
            self.phase = NavPhase.COMPLETED
            elapsed = time.time() - self.stats["start_time"]
            print(f"🏁 Mission Complete in {elapsed:.1f}s")
            print(f"   Stats: {self.stats}")
            result["phase"] = NavPhase.COMPLETED
            return result
        
        # ── 核心: 编码帧 + 检查导航 ──
        action = self.navigator.encode_and_check(
            frame=frame,
            depth=depth,
            position=position,
            yaw=yaw,
        )
        
        if action is not None:
            self.stats["vlm_calls"] += 1
            result["action"] = action
            result["should_move"] = True
            self.phase = NavPhase.REACTING
            
            # 检查是否需要重规划
            if self.navigator.nav_state.deviation > self.replan_deviation:
                self.stats["replans"] += 1
                remaining = self.global_path[self.navigator.nav_state.waypoint_index:]
                new_path = [position] + remaining
                self.navigator.set_mission(new_path)
                self.global_path = new_path
                print(f"🔄 Replanned! New path: {len(new_path)} waypoints")
            
            self.phase = NavPhase.CRUISING
        else:
            # 无动作 → 继续沿当前方向飞向下一航点
            result["should_move"] = True
            result["target_position"] = self.navigator.nav_state.next_waypoint
        
        result["phase"] = self.phase
        return result
    
    def _get_front_min_depth(self, depth):
        """获取前方中心区域最小深度"""
        h, w = depth.shape
        cy, cx = h // 2, w // 2
        region = depth[cy-h//6:cy+h//6, cx-w//6:cx+w//6]
        valid = region[(region > 0.1) & (region < 100)]
        return float(np.min(valid)) if len(valid) > 0 else float('inf')
    
    def recover_from_emergency(self):
        """手动解除紧急状态"""
        if self.phase == NavPhase.EMERGENCY:
            self.phase = NavPhase.CRUISING
            print("✅ Emergency cleared, resuming navigation")
```

---

### Task 3.3: 动作执行器 (Action Executor)

**文件: `navigation/action_executor.py`**

将 VLM 输出的 NavAction 转化为 AirSim 无人机控制指令:

```python
"""
将 NavAction 转化为 AirSim 飞行控制指令
"""

import math

class ActionExecutor:
    def __init__(self, drone_controller, default_altitude=-5.0):
        self.drone = drone_controller
        self.default_altitude = default_altitude
    
    def execute(self, action: NavAction, current_pos, current_yaw):
        """执行导航动作"""
        if action.action == "stop" or action.action == "hover":
            # 原地悬停
            self.drone.fly_by_velocity(0, 0, 0, duration=0.5)
            return
        
        # 计算新的朝向
        new_yaw = current_yaw + action.yaw_adjust
        
        # 先调整朝向 (如果偏转较大)
        if abs(action.yaw_adjust) > 5:
            self.drone.rotate_to_yaw(new_yaw)
        
        # 计算速度分量 (NED 坐标系)
        speed = action.speed
        yaw_rad = math.radians(new_yaw)
        
        # 前进方向映射
        if action.action in ["forward", "forward_left", "forward_right"]:
            vx = speed * math.cos(yaw_rad)
            vy = speed * math.sin(yaw_rad)
        elif action.action == "left":
            vx = speed * math.cos(yaw_rad + math.pi/4)
            vy = speed * math.sin(yaw_rad + math.pi/4)
        elif action.action == "right":
            vx = speed * math.cos(yaw_rad - math.pi/4)
            vy = speed * math.sin(yaw_rad - math.pi/4)
        elif action.action == "up":
            vx, vy = 0, 0
        elif action.action == "down":
            vx, vy = 0, 0
        else:
            vx = speed * math.cos(yaw_rad)
            vy = speed * math.sin(yaw_rad)
        
        # 高度控制
        vz = -action.altitude_adjust  # NED: 负值向上
        
        # 执行速度控制 (持续 0.5 秒)
        self.drone.fly_by_velocity(vx, vy, vz, duration=0.5)
    
    def fly_toward_waypoint(self, target_pos, current_pos, speed=2.0):
        """沿直线飞向航点 (无VLM指令时的默认行为)"""
        self.drone.fly_to(
            target_pos[0], target_pos[1], target_pos[2],
            speed=speed
        )
```

---

### Task 3.4: 集成测试 — 简单直线导航

编写集成测试: 起飞 → 直线飞到目标 → 降落

```python
"""
集成测试: AirSim 直线导航 (不含 VLM, 仅测试控制链路)
"""

def test_straight_line_navigation():
    drone = DroneController()
    drone.connect()
    drone.takeoff(height=-5)
    
    waypoints = [
        (0, 0, -5),
        (20, 0, -5),
        (40, 0, -5),
    ]
    
    for wp in waypoints[1:]:
        print(f"Flying to {wp}...")
        drone.fly_to(wp[0], wp[1], wp[2], speed=3.0)
        
        # 等待到达
        while True:
            pos = drone.get_position()
            dist = math.sqrt(sum((a-b)**2 for a,b in zip(pos, wp)))
            if dist < 2.0:
                print(f"  Reached {wp}, distance={dist:.1f}m")
                break
            time.sleep(0.5)
    
    drone.land()
    drone.disconnect()
    print("✅ Straight line navigation test passed!")
```

---

## 验收标准

- [ ] A* 全局规划器能在简单 3D 环境中找到路径
- [ ] 路径简化正确工作
- [ ] 状态机正确在各状态间转换
- [ ] 紧急停止机制工作正常 (深度图 < 1.5m)
- [ ] ActionExecutor 能将 NavAction 转化为 AirSim 控制
- [ ] 偏离过大时能触发重规划
- [ ] 直线导航测试通过

---

## 注意事项

1. AirSim 的 NED 坐标系: X=北, Y=东, Z=下。飞行高度用负值
2. A* 在大搜索空间可能很慢，Demo 阶段推荐:
   - 搜索空间限制在 100m × 100m × 15m 以内
   - 栅格分辨率 2m (减少节点数)
   - 或直接用 `SimpleGlobalPlanner` 手动给航点
3. `fly_by_velocity` 的 duration 参数控制持续时间，需要和帧采集频率协调
4. 碰撞检测不能只靠深度图 — 还要用 `check_collision()` API 作为最后防线
