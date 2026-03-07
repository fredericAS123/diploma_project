# 阶段 4: 端到端集成 + Demo

> **预计工期**: Day 9-11  
> **执行环境**: 本地 PC (AirSim) + 云端 4090 (推理)  
> **前置条件**: 阶段 1-3 全部完成  
> **目标**: 在 AirSim 中完成一次完整的从起点到终点的无人机视觉导航 Demo

---

## 任务清单

### Task 4.1: 主控循环 (Main Navigation Loop)

**文件: `navigation/scripts/run_demo.py`**

这是整个系统的入口脚本，串联所有模块:

```python
"""
🚁 流式 VLM 无人机导航 — 端到端 Demo
=====================================

启动顺序:
  1. 本地: 启动 AirSim (Blocks.exe 或 AirSimNH.exe)
  2. 云端: python navigation/communication/server.py
  3. 本地: python navigation/scripts/run_demo.py

配置:
  - 服务器地址: SERVER_URL
  - 飞行参数: FLIGHT_ALTITUDE, CRUISE_SPEED
  - 帧采集: FRAME_INTERVAL (秒)
  - 导航参数: CHECK_INTERVAL, DEVIATION_THRESHOLD
"""

import time
import math
import sys
import argparse
import json
import numpy as np
from PIL import Image
from datetime import datetime

# 添加项目路径
sys.path.insert(0, "d:/diploma_project")

from navigation.airsim_bridge import DroneController
from navigation.communication.client import InferenceClient

# ── 配置 ──
SERVER_URL = "http://云端IP:5000"
FLIGHT_ALTITUDE = -5.0        # AirSim NED, 负值 = 向上
CRUISE_SPEED = 2.0            # 巡航速度 m/s
FRAME_INTERVAL = 0.5          # 帧采集间隔 (秒) → 2 FPS
MAX_MISSION_TIME = 300        # 最大任务时间 (秒)
CHECK_INTERVAL = 5            # 每 N 帧触发 VLM 检查
DEVIATION_THRESHOLD = 3.0     # 偏离阈值 (米)
OBSTACLE_THRESHOLD = 5.0      # 障碍物距离阈值 (米)
EMERGENCY_DISTANCE = 1.5      # 紧急停止距离 (米)
WAYPOINT_REACH_DIST = 3.0     # 航点到达判定距离 (米)


def run_navigation_demo(args):
    """主导航循环"""
    
    print("=" * 60)
    print("🚁 Streaming VLM Drone Navigation Demo")
    print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # ── 1. 初始化 ──
    print("\n[1/5] Initializing...")
    
    drone = DroneController()
    drone.connect()
    print("  ✅ AirSim connected")
    
    client = InferenceClient(args.server_url)
    print(f"  ✅ Inference server: {args.server_url}")
    
    # ── 2. 全局路径 (Demo: 手动指定) ──
    print("\n[2/5] Planning global path...")
    
    # Blocks 环境的简单路径 (避开方块障碍)
    global_path = [
        (0, 0, FLIGHT_ALTITUDE),
        (15, 0, FLIGHT_ALTITUDE),
        (30, 5, FLIGHT_ALTITUDE),
        (45, 5, FLIGHT_ALTITUDE),
        (50, 0, FLIGHT_ALTITUDE),
    ]
    
    # 如果用户指定了路径文件
    if args.path_file:
        with open(args.path_file) as f:
            global_path = [tuple(wp) for wp in json.load(f)]
    
    print(f"  📍 Path: {len(global_path)} waypoints")
    for i, wp in enumerate(global_path):
        print(f"     WP{i}: ({wp[0]:.1f}, {wp[1]:.1f}, {wp[2]:.1f})")
    
    # 初始化云端任务
    client.init_mission(
        start=list(global_path[0]),
        goal=list(global_path[-1]),
        global_path=[list(wp) for wp in global_path],
    )
    print("  ✅ Mission initialized on server")
    
    # ── 3. 起飞 ──
    print("\n[3/5] Taking off...")
    drone.takeoff(height=FLIGHT_ALTITUDE)
    time.sleep(2)
    print(f"  ✅ Airborne at altitude {FLIGHT_ALTITUDE}m")
    
    # ── 4. 导航主循环 ──
    print("\n[4/5] Starting navigation loop...")
    print("-" * 60)
    
    frame_count = 0
    waypoint_idx = 1  # 从第二个航点开始 (第一个是起点)
    mission_start = time.time()
    
    # 统计
    stats = {
        "frames": 0,
        "vlm_calls": 0,
        "actions_taken": 0,
        "emergency_stops": 0,
        "collisions": 0,
        "path_deviation_max": 0,
        "start_time": mission_start,
    }
    
    # 日志
    log_entries = []
    
    try:
        while True:
            loop_start = time.time()
            
            # 检查超时
            elapsed = time.time() - mission_start
            if elapsed > MAX_MISSION_TIME:
                print(f"\n⏰ Mission timeout ({MAX_MISSION_TIME}s)")
                break
            
            # 检查任务完成
            if waypoint_idx >= len(global_path):
                print(f"\n🏁 Mission Complete! All {len(global_path)} waypoints reached!")
                break
            
            # ── 采集传感器数据 ──
            rgb_frame = drone.get_rgb_frame()
            depth_frame = drone.get_depth_frame()
            position = drone.get_position()
            yaw = drone.get_yaw()
            collision = drone.check_collision()
            
            if collision:
                stats["collisions"] += 1
                print(f"💥 COLLISION detected at ({position[0]:.1f}, {position[1]:.1f})")
            
            # ── 计算导航状态 ──
            current_wp = global_path[waypoint_idx]
            dist_to_wp = math.sqrt(sum((a-b)**2 for a,b in zip(position, current_wp)))
            
            # 偏离计算
            deviation = calc_path_deviation(position, global_path)
            stats["path_deviation_max"] = max(stats["path_deviation_max"], deviation)
            
            # 障碍物距离
            obstacle_dist = estimate_obstacle_distance(depth_frame)
            
            # ── 紧急停止 (程序化, 不等 VLM) ──
            if obstacle_dist < EMERGENCY_DISTANCE:
                print(f"🚨 EMERGENCY STOP! Obstacle at {obstacle_dist:.1f}m")
                drone.fly_by_velocity(0, 0, 0, duration=1.0)
                stats["emergency_stops"] += 1
                time.sleep(1)
                # 尝试向上抬升
                drone.fly_by_velocity(0, 0, -1.0, duration=2.0)
                continue
            
            # ── 航点到达检测 ──
            if dist_to_wp < WAYPOINT_REACH_DIST:
                print(f"✅ Waypoint {waypoint_idx} reached! "
                      f"({current_wp[0]:.0f},{current_wp[1]:.0f},{current_wp[2]:.0f})")
                waypoint_idx += 1
                stats["actions_taken"] += 1
                if waypoint_idx >= len(global_path):
                    continue
                current_wp = global_path[waypoint_idx]
                dist_to_wp = math.sqrt(sum((a-b)**2 for a,b in zip(position, current_wp)))
            
            # ── 决定是否调用 VLM ──
            trigger_reason = None
            if obstacle_dist < OBSTACLE_THRESHOLD:
                trigger_reason = "obstacle"
            elif deviation > DEVIATION_THRESHOLD:
                trigger_reason = "deviation"
            elif frame_count % CHECK_INTERVAL == 0:
                trigger_reason = "periodic"
            
            # ── 发送帧到云端编码 ──
            depth_stats = {
                "min": float(np.min(depth_frame[depth_frame > 0.1])) if np.any(depth_frame > 0.1) else 100,
                "front_center": float(obstacle_dist),
            }
            
            try:
                encode_result = client.encode_frame(
                    image=rgb_frame,
                    depth_stats=depth_stats,
                    position=position,
                    yaw=yaw,
                    frame_id=frame_count,
                )
                stats["frames"] += 1
            except Exception as e:
                print(f"⚠️ Frame encode error: {e}")
                time.sleep(0.5)
                continue
            
            # ── 如果需要 VLM 决策 ──
            action_taken = False
            if trigger_reason is not None:
                try:
                    nav_result = client.navigate(
                        position=position,
                        yaw=yaw,
                        next_waypoint=current_wp,
                        deviation=deviation,
                        obstacle_distance=obstacle_dist,
                        trigger_reason=trigger_reason,
                    )
                    stats["vlm_calls"] += 1
                    
                    action = nav_result.get("action", "forward")
                    yaw_adjust = nav_result.get("yaw_adjust", 0)
                    speed = nav_result.get("speed", CRUISE_SPEED)
                    reasoning = nav_result.get("reasoning", "")
                    
                    print(f"  Frame {frame_count:4d} | "
                          f"Pos ({position[0]:6.1f},{position[1]:6.1f}) | "
                          f"WP{waypoint_idx} dist={dist_to_wp:5.1f}m | "
                          f"Dev={deviation:4.1f}m | "
                          f"Obs={obstacle_dist:5.1f}m | "
                          f"[{trigger_reason:9s}] → {action} yaw={yaw_adjust:+.0f}° "
                          f"speed={speed:.1f} | {reasoning}")
                    
                    # 执行动作
                    execute_nav_action(drone, action, yaw_adjust, speed, yaw, position, current_wp)
                    action_taken = True
                    stats["actions_taken"] += 1
                    
                    # 记录日志
                    log_entries.append({
                        "time": elapsed,
                        "frame": frame_count,
                        "position": position,
                        "waypoint": current_wp,
                        "deviation": deviation,
                        "obstacle_dist": obstacle_dist,
                        "trigger": trigger_reason,
                        "action": action,
                        "reasoning": reasoning,
                    })
                    
                except Exception as e:
                    print(f"⚠️ Navigate error: {e}")
            
            # ── 默认行为: 飞向下一航点 ──
            if not action_taken:
                fly_toward_waypoint(drone, position, current_wp, CRUISE_SPEED)
            
            frame_count += 1
            
            # ── 帧率控制 ──
            loop_time = time.time() - loop_start
            sleep_time = max(0, FRAME_INTERVAL - loop_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    except KeyboardInterrupt:
        print("\n⚠️ Mission interrupted by user")
    
    finally:
        # ── 5. 清理 ──
        print("\n[5/5] Landing and cleanup...")
        drone.land()
        time.sleep(2)
        drone.disconnect()
        
        # 打印统计
        total_time = time.time() - mission_start
        print("\n" + "=" * 60)
        print("📊 Mission Statistics")
        print("=" * 60)
        print(f"  Total time:        {total_time:.1f}s")
        print(f"  Frames encoded:    {stats['frames']}")
        print(f"  VLM calls:         {stats['vlm_calls']}")
        print(f"  Actions taken:     {stats['actions_taken']}")
        print(f"  Emergency stops:   {stats['emergency_stops']}")
        print(f"  Collisions:        {stats['collisions']}")
        print(f"  Max deviation:     {stats['path_deviation_max']:.1f}m")
        print(f"  Waypoints reached: {waypoint_idx}/{len(global_path)}")
        print(f"  Success:           {'✅' if waypoint_idx >= len(global_path) else '❌'}")
        
        # 保存日志
        log_file = f"nav_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(log_file, 'w') as f:
            json.dump({
                "stats": stats,
                "path": [list(wp) for wp in global_path],
                "log": log_entries,
            }, f, indent=2, default=str)
        print(f"  Log saved: {log_file}")


# ── 辅助函数 ──

def calc_path_deviation(position, path):
    """计算当前位置到全局路径的最小 2D 距离"""
    min_dist = float('inf')
    px, py = position[0], position[1]
    for i in range(len(path) - 1):
        ax, ay = path[i][0], path[i][1]
        bx, by = path[i+1][0], path[i+1][1]
        dx, dy = bx - ax, by - ay
        if dx == 0 and dy == 0:
            d = math.sqrt((px-ax)**2 + (py-ay)**2)
        else:
            t = max(0, min(1, ((px-ax)*dx + (py-ay)*dy) / (dx**2 + dy**2)))
            proj_x, proj_y = ax + t*dx, ay + t*dy
            d = math.sqrt((px-proj_x)**2 + (py-proj_y)**2)
        min_dist = min(min_dist, d)
    return min_dist


def estimate_obstacle_distance(depth):
    """从深度图估计前方障碍物距离"""
    h, w = depth.shape
    cy, cx = h // 2, w // 2
    region = depth[cy-h//6:cy+h//6, cx-w//6:cx+w//6]
    valid = region[(region > 0.1) & (region < 100)]
    if len(valid) == 0:
        return 100.0
    return float(np.percentile(valid, 5))


def execute_nav_action(drone, action, yaw_adjust, speed, current_yaw, current_pos, target_wp):
    """将 VLM 动作转化为 AirSim 控制"""
    if action in ["stop", "hover"]:
        drone.fly_by_velocity(0, 0, 0, duration=0.5)
        return
    
    # 调整朝向
    new_yaw = current_yaw + yaw_adjust
    if abs(yaw_adjust) > 5:
        drone.rotate_to_yaw(new_yaw)
    
    # 计算速度分量
    yaw_rad = math.radians(new_yaw)
    
    if action in ["forward", "forward_left", "forward_right"]:
        vx = speed * math.cos(yaw_rad)
        vy = speed * math.sin(yaw_rad)
    elif action == "left":
        vx = speed * math.cos(yaw_rad + math.pi/6)
        vy = speed * math.sin(yaw_rad + math.pi/6)
    elif action == "right":
        vx = speed * math.cos(yaw_rad - math.pi/6)
        vy = speed * math.sin(yaw_rad - math.pi/6)
    elif action == "up":
        drone.fly_by_velocity(0, 0, -1.0, duration=1.0)
        return
    elif action == "down":
        drone.fly_by_velocity(0, 0, 1.0, duration=1.0)
        return
    else:
        vx = speed * math.cos(yaw_rad)
        vy = speed * math.sin(yaw_rad)
    
    drone.fly_by_velocity(vx, vy, 0, duration=0.5)


def fly_toward_waypoint(drone, current_pos, target_wp, speed):
    """默认行为: 飞向下一航点"""
    drone.fly_to(target_wp[0], target_wp[1], target_wp[2], speed=speed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Streaming VLM Drone Navigation Demo")
    parser.add_argument("--server_url", default=SERVER_URL, help="Inference server URL")
    parser.add_argument("--path_file", default=None, help="JSON file with waypoints")
    parser.add_argument("--altitude", type=float, default=FLIGHT_ALTITUDE)
    parser.add_argument("--speed", type=float, default=CRUISE_SPEED)
    parser.add_argument("--check_interval", type=int, default=CHECK_INTERVAL)
    args = parser.parse_args()
    
    FLIGHT_ALTITUDE = args.altitude
    CRUISE_SPEED = args.speed
    CHECK_INTERVAL = args.check_interval
    
    run_navigation_demo(args)
```

---

### Task 4.2: 云端推理服务完整实现

**文件: `navigation/communication/server.py`**

基于阶段 1 的框架，完成完整的推理服务:

```python
"""
云端推理服务 — Flask REST API

启动命令: python navigation/communication/server.py --port 5000

依赖:
  pip install flask torch transformers Pillow
"""

from flask import Flask, request, jsonify
import torch
import base64
import io
import time
from PIL import Image

app = Flask(__name__)

# 全局变量
engine = None       # NavigationStreamEngine
model = None
processor = None

def load_model():
    """加载模型 (启动时调用)"""
    global model, processor, engine
    
    from temporal_encoding.model.stream_qwen_model import StreamQwenModel
    from temporal_encoding.model.kv_cache_eviction import EvictionConfig
    from navigation.local_vlm_navigator import NavigationStreamEngine
    from transformers import AutoProcessor
    
    print("Loading Qwen2.5-VL-3B...")
    model = StreamQwenModel.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    
    engine = NavigationStreamEngine(
        model=model,
        processor=processor,
        device="cuda",
        eviction_config=EvictionConfig(max_cache_tokens=80000),
        check_interval=5,
        deviation_threshold=3.0,
        obstacle_threshold=5.0,
    )
    print("✅ Model loaded, server ready!")


@app.route('/api/init_mission', methods=['POST'])
def init_mission():
    global engine
    data = request.json
    
    engine.reset_navigation()
    global_path = [tuple(wp) for wp in data['global_path']]
    engine.set_mission(global_path)
    
    return jsonify({"status": "initialized", "waypoints": len(global_path)})


@app.route('/api/encode_frame', methods=['POST'])
def encode_frame():
    data = request.json
    
    # 解码图像
    img_bytes = base64.b64decode(data['image'])
    image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    
    # 编码帧
    t0 = time.time()
    result = engine.append_frame(image, text_content="Navigation frame.")
    latency = (time.time() - t0) * 1000
    
    return jsonify({
        "status": "encoded",
        "cache_len": engine.cache_manager.get_seq_length(),
        "frame_id": data.get('frame_id', -1),
        "latency_ms": latency,
    })


@app.route('/api/navigate', methods=['POST'])
def navigate():
    data = request.json
    
    position = tuple(data['position'])
    yaw = data['yaw']
    next_waypoint = tuple(data['next_waypoint'])
    deviation = data['deviation']
    obstacle_distance = data['obstacle_distance']
    trigger_reason = data['trigger_reason']
    
    # 更新导航器状态
    engine.nav_state.position = position
    engine.nav_state.yaw = yaw
    engine.nav_state.next_waypoint = next_waypoint
    engine.nav_state.deviation = deviation
    engine.nav_state.obstacle_distance = obstacle_distance
    
    # 构造导航 prompt 并询问
    t0 = time.time()
    nav_prompt = engine._build_nav_prompt(trigger_reason)
    
    try:
        response_text, metrics = engine.ask(
            question=nav_prompt,
            max_new_tokens=80,
            do_sample=False,
        )
        action = engine._parse_nav_response(response_text)
        latency = (time.time() - t0) * 1000
        
        return jsonify({
            "action": action.action,
            "yaw_adjust": action.yaw_adjust,
            "speed": action.speed,
            "reasoning": action.reasoning,
            "confidence": action.confidence,
            "raw_response": response_text,
            "latency_ms": latency,
        })
    except Exception as e:
        return jsonify({
            "action": "hover",
            "yaw_adjust": 0,
            "speed": 0,
            "reasoning": f"Error: {str(e)}",
            "confidence": 0,
            "latency_ms": (time.time() - t0) * 1000,
        })


@app.route('/api/encode_and_navigate', methods=['POST'])
def encode_and_navigate():
    """合并端点: 编码帧 + 导航决策 (减少网络往返)"""
    data = request.json
    
    # 解码图像
    img_bytes = base64.b64decode(data['image'])
    image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    
    position = tuple(data['position'])
    yaw = data['yaw']
    
    # 深度统计 (可选)
    depth = None  # 深度图不通过网络传输, 用统计值代替
    
    t0 = time.time()
    action = engine.encode_and_check(
        frame=image,
        depth=depth,
        position=position,
        yaw=yaw,
    )
    latency = (time.time() - t0) * 1000
    
    if action is not None:
        return jsonify({
            "encoded": True,
            "has_action": True,
            "action": action.action,
            "yaw_adjust": action.yaw_adjust,
            "speed": action.speed,
            "reasoning": action.reasoning,
            "latency_ms": latency,
        })
    else:
        return jsonify({
            "encoded": True,
            "has_action": False,
            "latency_ms": latency,
        })


@app.route('/api/reset', methods=['POST'])
def reset():
    engine.reset_navigation()
    return jsonify({"status": "reset"})


@app.route('/api/status', methods=['GET'])
def status():
    return jsonify({
        "model_loaded": engine is not None,
        "cache_len": engine.cache_manager.get_seq_length() if engine else 0,
        "frame_count": engine.frame_count if engine else 0,
        "mission_active": engine.nav_state.mission_active if engine else False,
    })


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--host', default='0.0.0.0')
    args = parser.parse_args()
    
    load_model()
    app.run(host=args.host, port=args.port, threaded=False)
```

---

### Task 4.3: 测试场景设计

**场景 1: 直线飞行 (最简单)**
```json
{
  "name": "straight_line",
  "path": [[0,0,-5], [50,0,-5]],
  "obstacles": [],
  "description": "直线飞行，无障碍"
}
```

**场景 2: L 形路径 (转弯)**
```json
{
  "name": "l_shape",
  "path": [[0,0,-5], [30,0,-5], [30,20,-5]],
  "obstacles": [],
  "description": "L形路径，需要右转"
}
```

**场景 3: 绕障 (核心场景)**
```json
{
  "name": "obstacle_avoidance",
  "path": [[0,0,-5], [15,0,-5], [30,5,-5], [45,0,-5]],
  "obstacles": [
    {"center": [20, 0, -5], "size": [4, 8, 10]}
  ],
  "description": "前方有障碍物，需要绕行"
}
```

**文件: `navigation/config/test_scenarios.json`**
存储所有测试场景配置，供 `run_demo.py` 加载使用。

---

### Task 4.4: Demo 录制与可视化

添加可视化功能到主循环:

1. **实时可视化窗口** (OpenCV):
   - 显示当前相机画面
   - 叠加文字信息: 位置、速度、航点、VLM 决策
   - 叠加小地图: 显示路径 + 当前位置 + 航点

2. **录制视频**:
   - 保存每帧 RGB + 叠加信息
   - 最终输出 MP4 视频作为 Demo

```python
# 可视化伪代码
import cv2

def create_overlay(frame, position, waypoint, action, stats):
    """在帧上叠加导航信息"""
    img = np.array(frame)
    
    # 位置信息
    cv2.putText(img, f"Pos: ({position[0]:.1f}, {position[1]:.1f})", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    
    # 航点信息
    cv2.putText(img, f"WP: ({waypoint[0]:.1f}, {waypoint[1]:.1f})", 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
    
    # 动作
    if action:
        cv2.putText(img, f"Action: {action}", 
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
    
    return Image.fromarray(img)
```

---

## 验收标准

- [ ] 场景1 (直线): 无人机从起点飞到终点，无碰撞
- [ ] 场景2 (L形): 无人机正确转弯到达终点
- [ ] 场景3 (绕障): 无人机检测到障碍并绕行
- [ ] 整个流程可以录制视频
- [ ] 端到端延迟 (帧采集→决策→执行) < 2 秒
- [ ] VLM 输出的 JSON 解析成功率 > 80%
- [ ] 导航成功率 > 70% (场景1、2应接近100%)

---

## 调试技巧

1. **先不接 VLM，纯程序化飞**: 只用 `fly_to_waypoint` 验证控制链路
2. **固定 VLM 输出**: 让服务器返回固定的 `{"action":"forward","yaw":0,"speed":2,"reason":"test"}`，验证控制链路
3. **离线测试 VLM**: 用 AirSim 截图保存为文件，在云端单独测试 VLM 对这些图片的理解
4. **降低速度**: 调试时将 `CRUISE_SPEED` 降到 1.0，避免飞太快来不及反应
5. **打印丰富日志**: 每帧打印位置、偏离、障碍距离，方便定位问题
