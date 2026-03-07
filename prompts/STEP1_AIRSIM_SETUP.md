# 阶段 1: AirSim 环境搭建 + 基础通信

> **预计工期**: Day 1-2  
> **执行环境**: 本地 PC (AirSim) + 云端 4090 (推理服务)  
> **目标**: 本地 AirSim 无人机飞行 + 帧采集，云端接收帧并返回指令，双向通信跑通

---

## 任务清单

### Task 1.1: AirSim 安装与配置 (本地 PC)

**操作步骤**:

1. 下载 AirSim 预编译二进制包 (无需自己编译 Unreal):
   - 从 https://github.com/Microsoft/AirSim/releases 下载最新的 `AirSimNH.zip` (Neighborhood 环境) 或 `Blocks.zip` (简单方块环境)
   - 解压到 `D:\AirSim\` 目录
   - **推荐先用 Blocks 环境** (简单、加载快，适合调试)

2. 配置 AirSim `settings.json`:
   - 文件位置: `C:\Users\<你的用户名>\Documents\AirSim\settings.json`
   - 创建或替换为以下内容:

```json
{
  "SettingsVersion": 1.2,
  "SimMode": "Multirotor",
  "ViewMode": "SpringArmChase",
  "Vehicles": {
    "Drone1": {
      "VehicleType": "SimpleFlight",
      "DefaultVehicleState": "Armed",
      "X": 0, "Y": 0, "Z": 0,
      "Cameras": {
        "front_center": {
          "CaptureSettings": [
            {
              "ImageType": 0,
              "Width": 384,
              "Height": 384,
              "FOV_Degrees": 90,
              "AutoExposureSpeed": 100,
              "MotionBlurAmount": 0
            },
            {
              "ImageType": 1,
              "Width": 384,
              "Height": 384
            }
          ],
          "X": 0.25, "Y": 0, "Z": -0.18,
          "Pitch": -10, "Roll": 0, "Yaw": 0
        }
      }
    }
  },
  "SubWindows": [
    {"WindowID": 0, "CameraName": "front_center", "ImageType": 0, "VehicleName": "Drone1"}
  ]
}
```

3. 安装 AirSim Python 包:
```bash
pip install airsim msgpack-rpc-python
pip install numpy Pillow opencv-python
```

4. 启动 AirSim 环境:
   - 双击 `Blocks.exe` (或 `AirSimNH.exe`)
   - 等待 Unreal Engine 窗口打开
   - 确认左上角显示 "Connected" 或按 F1 查看帮助

---

### Task 1.2: AirSim 客户端封装 (本地 PC)

在 `d:\diploma_project\navigation\` 目录下创建以下文件:

**文件: `navigation/airsim_bridge.py`**

功能要求:
1. `DroneController` 类，封装 AirSim MultirotorClient
2. 实现以下方法:
   - `connect()`: 连接 AirSim，启用 API 控制，解锁电机
   - `takeoff()`: 起飞到指定高度 (默认 -5m，NED 坐标系 Z 为负上)
   - `land()`: 安全降落
   - `get_position() -> (x, y, z)`: 获取当前位置
   - `get_yaw() -> float`: 获取当前朝向角 (度)
   - `get_rgb_frame() -> PIL.Image`: 获取前方 RGB 图像 (384×384)
   - `get_depth_frame() -> np.ndarray`: 获取前方深度图 (384×384, 单位米)
   - `get_rgb_and_depth() -> (PIL.Image, np.ndarray)`: 同时获取 RGB + 深度
   - `fly_to(x, y, z, speed=3.0)`: 飞到指定坐标 (异步)
   - `fly_by_velocity(vx, vy, vz, duration=1.0)`: 速度控制
   - `rotate_to_yaw(yaw_deg)`: 旋转到指定朝向
   - `check_collision() -> bool`: 检测是否发生碰撞
   - `get_distance_to(target_pos) -> float`: 计算到目标点距离
   - `disconnect()`: 断开连接

3. 关键实现细节:
   - AirSim 使用 NED 坐标系 (North-East-Down)，Z 轴向下为正，飞行高度用负值
   - RGB 图像从 `simGetImages` 获取未压缩数据，转为 PIL Image
   - 深度图使用 `DepthPlanar` 类型获取浮点数据
   - 所有异步方法 (`Async`) 需要 `.join()` 等待完成
   - 位姿通过 `simGetVehiclePose()` 获取，四元数转欧拉角取 yaw

4. 参考代码模板 (来自 Microsoft PromptCraft-Robotics):

```python
import airsim
import numpy as np
from PIL import Image
import math

class DroneController:
    def __init__(self):
        self.client = airsim.MultirotorClient()
        
    def connect(self):
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
    
    def get_rgb_frame(self):
        responses = self.client.simGetImages([
            airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False)
        ])
        response = responses[0]
        img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        img_rgb = img1d.reshape(response.height, response.width, 3)
        # AirSim returns BGR, convert to RGB
        img_rgb = img_rgb[:, :, ::-1].copy()
        return Image.fromarray(img_rgb)
    
    def get_depth_frame(self):
        responses = self.client.simGetImages([
            airsim.ImageRequest("front_center", airsim.ImageType.DepthPlanar, True, False)
        ])
        depth = airsim.list_to_2d_float_array(
            responses[0].image_data_float,
            responses[0].width,
            responses[0].height
        )
        return depth  # shape (H, W), 单位米
    
    # ... 其他方法按上述需求实现
```

---

### Task 1.3: 云端推理通信服务

**架构选择**: Flask REST API (最简单，适合 Demo)

**文件: `navigation/communication/server.py`** (部署在云端 4090)

功能要求:
1. Flask 服务，监听端口 5000
2. 提供以下 REST 端点:

```
POST /api/encode_frame
  Body: {
    "image": base64编码的JPEG图像,
    "depth_stats": {"min": float, "max": float, "front_center": float},
    "position": [x, y, z],
    "yaw": float,
    "frame_id": int
  }
  Response: {
    "status": "encoded",
    "cache_len": int,
    "frame_id": int
  }

POST /api/navigate
  Body: {
    "position": [x, y, z],
    "yaw": float,
    "next_waypoint": [x, y, z],
    "deviation": float,
    "obstacle_distance": float,
    "trigger_reason": "periodic|deviation|obstacle"
  }
  Response: {
    "action": "forward|left|right|stop|forward_left|forward_right",
    "yaw_adjust": float,  // 度
    "speed": float,       // m/s
    "reasoning": "...",
    "latency_ms": float
  }

POST /api/init_mission
  Body: {
    "start": [x, y, z],
    "goal": [x, y, z],
    "global_path": [[x,y,z], [x,y,z], ...],
    "system_prompt": "可选自定义系统提示"
  }
  Response: {"status": "initialized"}

POST /api/reset
  Response: {"status": "reset"}
```

3. 内部逻辑:
   - `/api/init_mission`: 初始化 `VideoStreamingInference` (或后续的 `NavigationStreamEngine`)，设置系统 prompt
   - `/api/encode_frame`: 将 base64 图像解码为 PIL Image，调用 `engine.append_frame()`
   - `/api/navigate`: 构造导航 prompt (包含位置、航点、偏离信息)，调用 `engine.ask()` 获取决策
   - `/api/reset`: 重置引擎状态

**文件: `navigation/communication/client.py`** (运行在本地 PC)

功能要求:
1. `InferenceClient` 类，封装 HTTP 请求
2. 方法与 server 端点一一对应
3. 图像用 JPEG 压缩后 base64 编码传输
4. 添加超时处理和重试逻辑

```python
import requests
import base64
import io
from PIL import Image

class InferenceClient:
    def __init__(self, server_url="http://云端IP:5000"):
        self.server_url = server_url
        self.timeout = 10  # seconds
    
    def encode_frame(self, image: Image.Image, depth_stats, position, yaw, frame_id):
        buf = io.BytesIO()
        image.save(buf, format='JPEG', quality=85)
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        
        resp = requests.post(f"{self.server_url}/api/encode_frame", json={
            "image": img_b64,
            "depth_stats": depth_stats,
            "position": list(position),
            "yaw": float(yaw),
            "frame_id": frame_id,
        }, timeout=self.timeout)
        return resp.json()
    
    def navigate(self, position, yaw, next_waypoint, deviation, obstacle_distance, trigger_reason):
        resp = requests.post(f"{self.server_url}/api/navigate", json={
            "position": list(position),
            "yaw": float(yaw),
            "next_waypoint": list(next_waypoint),
            "deviation": float(deviation),
            "obstacle_distance": float(obstacle_distance),
            "trigger_reason": trigger_reason,
        }, timeout=self.timeout)
        return resp.json()
```

---

### Task 1.4: 端到端通信验证

**文件: `navigation/scripts/test_communication.py`**

编写测试脚本验证整个通信链路:

```python
# 测试流程:
# 1. 启动 AirSim 环境 (手动)
# 2. 启动云端推理服务 (手动: python server.py)
# 3. 运行此脚本

def test_basic_flow():
    """测试基本通信流程"""
    drone = DroneController()
    drone.connect()
    
    client = InferenceClient("http://云端IP:5000")
    
    # 初始化任务
    client.init_mission(
        start=[0, 0, -5],
        goal=[50, 0, -5],
        global_path=[[0,0,-5], [25,0,-5], [50,0,-5]]
    )
    
    # 起飞
    drone.takeoff(height=-5)
    
    # 采集 5 帧并发送
    for i in range(5):
        rgb = drone.get_rgb_frame()
        pos = drone.get_position()
        yaw = drone.get_yaw()
        
        result = client.encode_frame(rgb, {}, pos, yaw, i)
        print(f"Frame {i}: {result}")
        time.sleep(0.5)
    
    # 请求导航决策
    nav = client.navigate(
        position=drone.get_position(),
        yaw=drone.get_yaw(),
        next_waypoint=[25, 0, -5],
        deviation=0.5,
        obstacle_distance=100.0,
        trigger_reason="periodic"
    )
    print(f"Navigation decision: {nav}")
    
    drone.land()
    drone.disconnect()
```

---

## 验收标准

- [ ] AirSim Blocks 环境正常启动
- [ ] Python 脚本能控制无人机起飞、移动、降落
- [ ] 能以 384×384 分辨率采集 RGB + 深度图
- [ ] 云端 Flask 服务正常启动，能加载 Qwen2.5-VL-3B 模型
- [ ] 本地帧能通过网络发送到云端并成功编码
- [ ] 云端能返回导航指令到本地
- [ ] 往返延迟 < 500ms (局域网环境)

---

## 注意事项

1. AirSim 使用 NED 坐标系，Z 轴向下为正。飞行高度 5 米 = Z = -5
2. 如果本地和云端不在同一网络，需要配置内网穿透或使用云端 AirSim
3. 图像传输用 JPEG 压缩可大幅减少带宽 (~50KB vs ~440KB)
4. Flask 服务器需要在加载模型后才能接受请求，注意启动顺序
5. 首帧编码会比较慢 (包含 system prompt)，后续帧会快很多
