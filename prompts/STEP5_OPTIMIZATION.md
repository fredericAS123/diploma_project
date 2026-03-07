# 阶段 5: 优化、微调与进阶功能

> **预计工期**: Day 12-15  
> **执行环境**: A800 (微调) + 云端 4090 (推理) + 本地 PC (AirSim)  
> **前置条件**: 阶段 4 完成，基础 Demo 可运行  
> **目标**: 提升导航质量、实现进阶功能

---

## 任务清单

### Task 5.1: 数据收集 + LoRA 微调 (A800)

**目标**: 用 AirSim 中收集的数据微调 Qwen2.5-VL-3B，使其更好地输出导航 JSON

#### 5.1.1 数据收集脚本

**文件: `navigation/scripts/collect_training_data.py`**

在 AirSim 中自动飞行并收集训练数据:

```python
"""
自动收集导航训练数据

策略:
1. 在多个场景中飞行多条路径
2. 在每个位置记录: RGB帧 + 位姿 + 航点 + 理想动作
3. 理想动作通过程序化计算:
   - 无障碍 + 对齐航点 → forward
   - 有障碍 → 计算绕行方向
   - 偏离路径 → 计算纠偏方向
4. 生成 SFT 格式数据集
"""

import json
import time
import math
import os
from PIL import Image

def compute_ideal_action(position, yaw, next_waypoint, obstacle_distance, deviation):
    """计算理想导航动作 (作为训练标签)"""
    dx = next_waypoint[0] - position[0]
    dy = next_waypoint[1] - position[1]
    target_bearing = math.degrees(math.atan2(dy, dx))
    relative_angle = target_bearing - yaw
    
    # 归一化
    while relative_angle > 180: relative_angle -= 360
    while relative_angle < -180: relative_angle += 360
    
    if obstacle_distance < 5.0:
        # 避障: 选择偏转较小的方向
        if relative_angle >= 0:
            return {"action": "right", "yaw": 30, "speed": 1.5, 
                    "reason": "Obstacle ahead, turning right to avoid"}
        else:
            return {"action": "left", "yaw": -30, "speed": 1.5,
                    "reason": "Obstacle ahead, turning left to avoid"}
    
    if deviation > 3.0:
        # 纠偏
        yaw_adj = max(-45, min(45, relative_angle * 0.5))
        return {"action": "forward", "yaw": round(yaw_adj), "speed": 2.0,
                "reason": f"Path deviation {deviation:.1f}m, correcting course"}
    
    if abs(relative_angle) > 15:
        # 调整朝向
        yaw_adj = max(-45, min(45, relative_angle * 0.3))
        return {"action": "forward", "yaw": round(yaw_adj), "speed": 2.0,
                "reason": "Adjusting heading toward waypoint"}
    
    # 正常前进
    return {"action": "forward", "yaw": 0, "speed": 2.5,
            "reason": "Path clear, proceeding forward"}


def collect_data_episode(drone, path, save_dir, episode_id):
    """收集一个飞行回合的数据"""
    samples = []
    frame_id = 0
    
    for wp_idx in range(1, len(path)):
        target = path[wp_idx]
        drone.fly_to(target[0], target[1], target[2], speed=2.0)
        
        # 飞行过程中持续采集
        while True:
            pos = drone.get_position()
            dist = math.sqrt(sum((a-b)**2 for a,b in zip(pos, target)))
            if dist < 2.0:
                break
            
            rgb = drone.get_rgb_frame()
            depth = drone.get_depth_frame()
            yaw = drone.get_yaw()
            deviation = calc_path_deviation(pos, path)
            obstacle_dist = estimate_obstacle_distance(depth)
            
            # 计算理想动作
            ideal_action = compute_ideal_action(pos, yaw, target, obstacle_dist, deviation)
            
            # 保存图像
            img_filename = f"ep{episode_id:03d}_frame{frame_id:05d}.jpg"
            img_path = os.path.join(save_dir, "images", img_filename)
            rgb.save(img_path, quality=90)
            
            # 记录样本
            sample = {
                "image": img_filename,
                "position": list(pos),
                "yaw": float(yaw),
                "next_waypoint": list(target),
                "deviation": float(deviation),
                "obstacle_distance": float(obstacle_dist),
                "ideal_action": ideal_action,
            }
            samples.append(sample)
            
            frame_id += 1
            time.sleep(0.3)  # ~3 FPS 采集
    
    return samples


def convert_to_sft_format(samples, output_file):
    """转换为 LLaMA-Factory / ms-swift 兼容的 SFT 格式"""
    sft_data = []
    
    for s in samples:
        # 构造导航 prompt (与 NavigationStreamEngine._build_nav_prompt 一致)
        pos = s["position"]
        wp = s["next_waypoint"]
        dx, dy = wp[0]-pos[0], wp[1]-pos[1]
        dist = math.sqrt(dx**2 + dy**2)
        
        user_text = (
            f"Current position: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})\n"
            f"Current heading: {s['yaw']:.0f}°\n"
            f"Next waypoint: ({wp[0]:.1f}, {wp[1]:.1f}, {wp[2]:.1f}), distance: {dist:.1f}m\n"
            f"Path deviation: {s['deviation']:.1f}m\n"
        )
        
        if s['obstacle_distance'] < 50:
            user_text += f"Nearest obstacle: {s['obstacle_distance']:.1f}m ahead\n"
        
        user_text += "Assess the scene and decide the navigation action."
        
        sft_sample = {
            "messages": [
                {
                    "role": "system",
                    "content": NavigationStreamEngine.NAV_SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"images/{s['image']}"},
                        {"type": "text", "text": user_text}
                    ]
                },
                {
                    "role": "assistant",
                    "content": json.dumps(s['ideal_action'])
                }
            ]
        }
        sft_data.append(sft_sample)
    
    with open(output_file, 'w') as f:
        json.dump(sft_data, f, indent=2)
    
    print(f"✅ Saved {len(sft_data)} SFT samples to {output_file}")
```

#### 5.1.2 LoRA 微调配置

**文件: `navigation/finetune/train_nav_lora.yaml`** (适配 LLaMA-Factory)

```yaml
### Model ###
model_name_or_path: Qwen/Qwen2.5-VL-3B-Instruct

### Method ###
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 32
lora_alpha: 64
lora_dropout: 0.05
lora_target: all  # q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj

### Dataset ###
dataset: nav_airsim  # 需要在 dataset_info.json 中注册
template: qwen2_vl
cutoff_len: 2048
preprocessing_num_workers: 8

### Output ###
output_dir: output/qwen25vl_nav_lora
logging_steps: 10
save_steps: 200
save_total_limit: 3

### Training ###
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 3
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
gradient_checkpointing: true

### Eval ###
eval_strategy: steps
eval_steps: 200
per_device_eval_batch_size: 1
```

**文件: `navigation/finetune/dataset_info.json`**

```json
{
  "nav_airsim": {
    "file_name": "nav_sft_data.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "messages"
    },
    "tags": {
      "role_tag": "role",
      "content_tag": "content",
      "system_tag": "system",
      "user_tag": "user",
      "assistant_tag": "assistant"
    }
  }
}
```

**微调命令**:
```bash
# 在 A800 上运行
cd LLaMA-Factory
llamafactory-cli train navigation/finetune/train_nav_lora.yaml
```

**预估**:
- 2000 样本 × 3 epochs ≈ 2-4 小时 (A800)
- VRAM: ~16-20GB (LoRA + gradient checkpointing)
- 之后在 4090 推理服务器上加载 LoRA adapter

---

### Task 5.2: 主动响应机制 (进阶)

**参考**: VideoLLM-online 的 Streaming EOS 机制

**改造思路**: 不再依赖定时/阈值触发，而是让模型自己决定何时说话

```python
"""
ProactiveNavigator — 主动响应导航引擎

核心改进:
  每帧编码后，检查模型 logits 中"继续等待 token"的概率。
  如果概率低于阈值 → 模型有话要说 → 触发生成。
  
实现方式:
  1. 定义一个 "SAFE token" (如逗号 `,` 或自定义 token)
  2. append_frame 后的最后一个 logits 中，检查该 token 的概率
  3. 概率 > threshold → 无事发生，继续编码
  4. 概率 < threshold → 触发 ask() 生成导航指令
  
简化版 (无需微调):
  使用 ask() 但只生成 1 个 token，检查是否为 "{"  
  如果是 "{" → 继续生成完整 JSON
  如果不是 → 认为模型认为安全，跳过
"""

class ProactiveNavigator(NavigationStreamEngine):
    
    def __init__(self, *args, proactive_threshold=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.proactive_threshold = proactive_threshold
    
    def encode_and_check_proactive(self, frame, depth, position, yaw):
        """
        主动响应版: 让模型自己决定是否需要说话
        """
        # 编码帧
        self.append_frame(frame, text_content="Navigation frame.")
        
        # 更新状态
        self.nav_state.position = position
        self.nav_state.yaw = yaw
        
        # 构造简短检查 prompt
        pos = position
        wp = self.nav_state.next_waypoint
        check_prompt = (
            f"Pos({pos[0]:.0f},{pos[1]:.0f}), "
            f"WP({wp[0]:.0f},{wp[1]:.0f}), "
            f"Dev={self.nav_state.deviation:.1f}m. "
            f"Need action? If yes output JSON, if no output SAFE."
        )
        
        # 只生成 1 个 token 探测
        response, _ = self.ask(
            question=check_prompt,
            max_new_tokens=1,
            min_new_tokens=1,
            do_sample=False,
        )
        
        first_char = response.strip()
        
        if first_char == '{' or first_char.startswith('{'):
            # 模型开始输出 JSON → 需要完整响应
            full_prompt = self._build_nav_prompt("proactive")
            response_text, metrics = self.ask(
                question=full_prompt,
                max_new_tokens=80,
                do_sample=False,
            )
            return self._parse_nav_response(response_text)
        else:
            # SAFE — 不需要动作
            return None
```

---

### Task 5.3: 视觉路径叠加

在图像上绘制全局路径投影，让 VLM 直观看到路径:

```python
"""
在 RGB 帧上叠加全局路径投影线

需要已知相机内参和无人机位姿，将 3D 航点投影到图像平面。
对于 AirSim，相机参数可以从 settings.json 获取。
"""

import cv2
import numpy as np
from PIL import Image

class PathVisualizer:
    def __init__(self, image_width=384, image_height=384, fov_deg=90):
        self.w = image_width
        self.h = image_height
        self.fov = fov_deg
        
        # 计算相机内参
        self.fx = self.w / (2 * np.tan(np.radians(fov_deg / 2)))
        self.fy = self.fx  # 正方形图像
        self.cx = self.w / 2
        self.cy = self.h / 2
    
    def project_path_on_image(self, image, waypoints, drone_pos, drone_yaw):
        """
        将 3D 航点投影到图像上并绘制路径线
        
        Args:
            image: PIL Image
            waypoints: List of (x, y, z) in world frame
            drone_pos: (x, y, z) drone position
            drone_yaw: yaw angle in degrees
        
        Returns:
            PIL Image with path overlay
        """
        img = np.array(image).copy()
        
        yaw_rad = np.radians(drone_yaw)
        
        # 世界坐标 → 相机坐标 (NED → Camera: X_cam=Y_ned, Y_cam=-Z_ned, Z_cam=X_ned)
        R_yaw = np.array([
            [np.cos(yaw_rad), np.sin(yaw_rad), 0],
            [-np.sin(yaw_rad), np.cos(yaw_rad), 0],
            [0, 0, 1]
        ])
        
        projected_points = []
        for wp in waypoints:
            # 世界坐标差
            delta = np.array([wp[0] - drone_pos[0], wp[1] - drone_pos[1], wp[2] - drone_pos[2]])
            
            # 旋转到机体坐标系
            body = R_yaw @ delta
            
            # NED to camera: forward=X_body → Z_cam, right=Y_body → X_cam, down=Z_body → Y_cam
            cam_x = body[1]   # right → image x
            cam_y = body[2]   # down → image y
            cam_z = body[0]   # forward → depth
            
            if cam_z <= 0.5:  # 在相机后方或太近
                projected_points.append(None)
                continue
            
            # 投影
            u = int(self.fx * cam_x / cam_z + self.cx)
            v = int(self.fy * cam_y / cam_z + self.cy)
            
            if 0 <= u < self.w and 0 <= v < self.h:
                projected_points.append((u, v))
            else:
                projected_points.append(None)  # 超出画面
        
        # 绘制路径线
        valid_pts = [(i, p) for i, p in enumerate(projected_points) if p is not None]
        for idx in range(len(valid_pts) - 1):
            i1, p1 = valid_pts[idx]
            i2, p2 = valid_pts[idx + 1]
            cv2.line(img, p1, p2, (0, 255, 0), 2)
        
        # 绘制航点圆点
        for i, pt in valid_pts:
            color = (0, 0, 255) if i == 0 else (0, 255, 255)  # 第一个红色，其余黄色
            cv2.circle(img, pt, 6, color, -1)
            cv2.putText(img, f"WP{i}", (pt[0]+8, pt[1]-8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return Image.fromarray(img)
```

---

### Task 5.4: 性能优化

1. **减少不必要的 VLM 调用**:
   - 前方完全空旷时 (深度 > 30m) 不调用 VLM
   - 与上次检查相比位置变化很小时不调用
   - 使用更长的 check_interval (如 10 帧)

2. **异步编码和控制**:
   ```python
   # 帧编码和飞行控制可以并行
   import threading
   
   encode_thread = threading.Thread(target=client.encode_frame, args=(...))
   encode_thread.start()
   
   # 同时执行上一次的飞行指令
   execute_nav_action(drone, last_action, ...)
   
   encode_thread.join()
   ```

3. **批量帧编码**:
   - 每次发送 2-4 帧作为视频 chunk (as_video=True)
   - 减少 HTTP 请求次数
   - 利用 temporal_patch_size=2 的特性

4. **模型量化** (如果 4090 显存紧张):
   ```python
   model = StreamQwenModel.from_pretrained(
       "Qwen/Qwen2.5-VL-3B-Instruct",
       torch_dtype=torch.bfloat16,  # 已是 bf16
       # 进一步量化: load_in_4bit=True (需要 bitsandbytes)
   )
   ```

---

### Task 5.5: 更复杂的测试场景

**场景 4: 城市环境 (AirSimNH)**
```json
{
  "name": "neighborhood",
  "environment": "AirSimNH",
  "path": [[0,0,-8], [30,-10,-8], [50,0,-8], [80,15,-8], [100,0,-8]],
  "description": "在居民区中导航，需要绕过房屋和树木"
}
```

**场景 5: 动态高度 (爬升+下降)**
```json
{
  "name": "altitude_change",
  "path": [[0,0,-5], [20,0,-5], [30,0,-10], [40,0,-3], [50,0,-5]],
  "description": "需要调整飞行高度以越过或绕过障碍"
}
```

**场景 6: 多段路径 (长距离)**
```json
{
  "name": "long_range",
  "path": [[0,0,-8], [30,0,-8], [50,20,-8], [70,20,-8], [100,0,-8], [130,-10,-8]],
  "description": "长距离导航，测试 KV Cache 滑动窗口的效果"
}
```

---

## 验收标准

- [ ] (可选) LoRA 微调完成，JSON 输出成功率提升到 >90%
- [ ] 主动响应机制可工作 (或至少有原型)
- [ ] 路径可视化叠加到图像上
- [ ] 在 Blocks 环境中至少 3 个场景导航成功
- [ ] (如有 AirSimNH) 在更复杂环境中有基本避障能力
- [ ] 性能优化后帧编码延迟 < 50ms
- [ ] Demo 视频录制完成

---

## 后续路线图 (Demo 之后)

### 短期 (1-2 周)
- [ ] 在 AirSimNH 复杂场景中大量测试
- [ ] 收集更多训练数据 + 二次微调
- [ ] 实现完整的 Streaming EOS 主动响应 (参考 VideoLLM-online 训练)
- [ ] 添加更多传感器 (LiDAR, IMU)

### 中期 (1-2 月)
- [ ] 实现完整的全局路径规划 (带 3D 障碍物检测)
- [ ] 多无人机协同导航
- [ ] Sim2Real 迁移到真实无人机
- [ ] 论文撰写

### 长期
- [ ] 端到端可微训练 (类似 LMDrive)
- [ ] 在线学习 + 强化学习优化
- [ ] 大规模场景测试 (城市级)
