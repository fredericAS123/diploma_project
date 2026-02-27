# Copilot 总执行指令：流式视频理解系统完整构建

> **最后更新**: 2026-02-25
> **适用范围**: 本 prompt 是给 Copilot 的一次性完整执行指令，涵盖从 KV Cache 淘汰验证到 VLM Agent 导航闭环的全部工作。

---

## 〇、项目背景与当前状态（你必须先理解）

### 硬件与模型

- **GPU**: RTX 4090 24GB（单卡，不可更换）
- **模型**: Qwen2.5-VL-3B-Instruct (bf16)，模型本体 ~7.1 GB VRAM
- **运行环境**: AutoDL 远程服务器，Python 3.10+, transformers >= 4.50
- **模型路径**: `/root/autodl-tmp/Qwen/Qwen2___5-VL-3B-Instruct`

### ⚠️ 部署拓扑约束（极其重要）

```
┌──────────────────────────┐       HTTP API        ┌───────────────────────────┐
│   本地 Windows 机器        │  ◄────────────────►  │   AutoDL 云服务器 (无 GUI)   │
│                           │   帧 → 动作指令       │                            │
│  ● AirSim + Unreal Engine │                       │  ● Qwen2.5-VL-3B (4090)    │
│  ● 有显示器/GUI           │                       │  ● 流式推理引擎 + Agent     │
│  ● 截帧 + 执行导航指令     │                       │  ● FastAPI 推理服务         │
└──────────────────────────┘                       └───────────────────────────┘
```

- **AutoDL 是无头服务器**（无 GUI、无显示器），不能运行 AirSim/Unreal Engine
- **AirSim 必须在本地 Windows 机器上运行**（需要 GPU 渲染 + 显示器）
- 两者通过 **HTTP REST API** 通信（AutoDL 提供 FastAPI 推理服务，本地发帧+收指令）
- 这是机器人/具身智能研究的**标准部署模式**（感知+规划在云端，执行在边端）

### 已完成的核心代码（不要重写，在此基础上工作）

代码位于 `temporal_encoding/model/` 目录下：

| 文件 | 功能 | 关键 API |
|------|------|---------|
| `stream_qwen_model.py` | 3-branch 流式 M-RoPE 位置追踪 | `StreamQwenModel`, `stream_state`, `get_rope_index` 拦截 |
| `cache_manager.py` | KV Cache 生命周期管理 | `KVCacheManager`, `snapshot(model)`, `restore(model)`, `evict_if_needed()` |
| `video_stream_inference.py` | 完整流式推理引擎 | `VideoStreamingInference`, `append_frame()`, `ask()`, `ask_choice()` |
| `kv_cache_eviction.py` | KV Cache 淘汰策略（三级） | `KVCacheEvictor`, `EvictionConfig`, `TokenTypeTracker` |
| `__init__.py` | 统一导出 | `StreamQwenModel`, `VideoStreamingInference`, `KVCacheManager`, `EvictionConfig` |

### 关键实测数据（来自 test_step10，你的所有决策必须基于这些数据）

```
模型 VRAM (bf16):           ~7.1 GB
KV cache 每 token:          ~36 KB (across 36 layers)
1920×1080, 4帧/chunk:       ~5,389 tokens/chunk, ~0.185 GB/chunk
30 chunks (120帧):          cache 161,719 tokens, VRAM reserved 22.89 GB → 极限
40 chunks (160帧):          OOM
安全 cache 预算:             ~100,000 tokens (~3.4 GB cache)
激进 cache 预算:             ~150,000 tokens (~5.2 GB cache, peak ~155K)
```

### 架构核心设计

1. **Chunk-Local ViT**: ViT 只在 chunk 内建模，跨 chunk 时序由 LLM + KV Cache + 3D-RoPE 负责
2. **Snapshot/Restore**: `ask()` 前保存 KV Cache + `stream_state`，QA 完成后恢复，防止文本污染视频缓存
3. **Auto Sink Detection**: `EvictionConfig(sink_size=0)` → 首帧后自动以 cache 长度作为 sink（因分辨率/帧数不同，不可硬编码）
4. **Token Tracker**: `TokenTypeTracker` 追踪每个 token 的模态类型和 chunk 归属，用于 Level 2/3 淘汰

---

## 一、审阅者严苛修改意见（你必须严格遵守）

以下是来自领域专家的强制性修改意见，**优先级高于所有其他设计考量**：

### ❌ 1. 禁止在实时流式系统中使用基于注意力分数的淘汰

**原因**:
- 在 4090 上实时收集并排序 36 层 Transformer 注意力矩阵会造成灾难性延迟
- Token 级零散丢弃会破坏 Qwen2.5-VL 展平后的视觉 patch 空间结构，导致幻觉
- LOOK-M、H₂O 等工作都是离线/一次性输入场景，不适用于高频追加的流式系统

**行动**: 
- Level 1 (Sink + Sliding Window) 是**唯一需要在实验中充分验证**的淘汰策略
- Level 2 (均匀时序采样) 代码已有，可作为消融实验对比项，但**不是主力**
- Level 3 (帧级重要性) **完全砍掉**，不投入任何精力

### ❌ 2. Benchmark 评估不应成为时间黑洞

**原因**:
- OVO-Bench/OVBench 测试"回溯过去"/"等待未来"的能力，但最终落脚点是无人机实时导航
- 导航需要的是"当前空间通行性判断准确率"和"多步决策连贯性"，而非标准 VQA 指标

**行动**:
- OVO-Bench 只跑极小规模子集（证明流式架构不产生灾难性遗忘即可）
- 核心评估指标改为 AirSim 具身指标：碰撞率、任务成功率、平均决策延迟

### ❌ 3. Agent 控制流必须处理"思考耗时"

**原因**:
- Qwen2.5-VL 3B 一次 3-5 步 Agent 推理约需 2-5 秒
- 这段时间内无人机盲飞，遇突发障碍必坠机
- CodeAgent 要求模型生成 Python 代码，7B 以下不稳定，语法错误触发重试更拉长延迟

**行动**:
- Agent 进入 `generate()` 期间，底层必须下达 `hover()` 悬停指令
- 放弃 `CodeAgent`，使用 `ToolCallingAgent`（JSON 结构化输出，容错率高）

### 🔄 4. 执行顺序调整

**原方案**: KV Cache → Benchmark → Agent
**修改后**: KV Cache → Agent 闭环 → 针对性评估

**理由**: 先让无人机飞起来，才能发现 KV Cache 延迟对飞行的真实影响，从而指导评测设计。

---

## 二、理论风险（Position Gap 问题）

### ⚠️ 风险描述

淘汰 KV Cache 后，保留的 token 对应的 Position Index 不再连续（如 [0..5438, 95000..100000]），产生巨大的 position gap。原生 Qwen2.5-VL **未针对此场景训练**，其 3D-RoPE 在大 gap 下可能导致：

- 注意力分布异常（attention collapse）
- 历史帧信息丢失，长时依赖建模能力退化
- 问答/字幕提取等下游任务质量明显下降

### 🚦 应对指令

如果在实验中发现淘汰后质量明显下降，你必须**在不微调模型的前提下**，依次探索以下工程方案：

1. **Position Index 重映射**: 淘汰后将保留 token 的 position 连续化（sink: 0..S, window: S+1..S+W），消除大 gap
2. **降低分辨率/帧率**: 减少单帧 token 数，降低淘汰频率，减小 gap
3. **增大 max_cache_tokens**: 权衡显存与历史保留长度（150K 激进值）
4. **prompt 工程 / 采样参数调优**: 温度、top_p、max_new_tokens

如所有工程方案均无效，最后再建议微调/适配训练。

---

## 三、Phase 1: KV Cache 淘汰验证（预计 3-5 天）

### 目标

验证 Level 1 (Sink + Sliding Window) 在 4090 上能稳定运行 >300 帧视频流不 OOM，且 ask() 质量可接受。

### 实验 A: Sink 自动检测验证

在 `temporal_encoding/` 下创建 `test_eviction_exp_a.py`:

**验证点**:
1. `EvictionConfig(sink_size=0)` → 首 chunk 后 `effective_sink_size == cache_len`
2. 不同分辨率/帧数下 sink 值不同且合理
3. `update_chunk_stats()` 正确记录平均 token 数
4. `window_size` 自动计算 = `max_cache_tokens - sink_size`

**参数**:
- `CHUNK_FRAME_CONFIGS = [2, 4]`
- `NUM_CHUNKS = 5`（不触发淘汰，仅验证检测）
- `MAX_CACHE_TOKENS = 100_000`

**报告**: 输出到 `test_eviction_exp_a_report.txt`

### 实验 B: OOM-Free 长程测试

创建 `test_eviction_exp_b.py`:

**验证点**:
1. 使用 `EvictionConfig(max_cache_tokens=X)` 编码 >300 帧 (75+ chunks) 不 OOM
2. 每 10 chunks 记录 VRAM 和 cache_seq_length，确认稳定
3. 淘汰确实被触发（`total_evictions > 0`）
4. `torch.cuda.empty_cache()` 在淘汰后调用

**参数扫描** (逐个测试，不并行):
```
max_cache_tokens = [100_000, 130_000, 150_000]
```

**报告**: 输出到 `test_eviction_exp_b_report.txt`

### 实验 C: 淘汰后 Ask 质量验证

创建 `test_eviction_exp_c.py`:

**验证点**:
1. 编码真实视频（MV/字幕视频），每 N chunks 调用 `ask()` 提取字幕
2. 全程不 OOM
3. `ask()` 后 cache 正确恢复 (snapshot/restore 在淘汰后仍工作)
4. 输出非空且与视频内容相关（质量不崩溃）

**参数**:
- `MAX_CACHE_TOKENS = 100_000`（先用保守值）
- `ASK_INTERVAL = 20`（每 20 chunks 提问一次）
- `QUESTION = "Read all visible text, lyrics, or subtitles on screen. Output verbatim. If no text, say 'no text'."`

**报告**: 输出到 `test_eviction_exp_c_report.txt`

### 执行顺序与迭代

```
A → B → C (严格顺序)
```

每个实验如果失败：
1. 阅读 `_report.txt`，确认失败条件
2. 根据失败类型修改代码（参见下方文件定位表）
3. 重新运行同一实验直到通过
4. 进入下一实验

**文件定位表**:

| 失败类型 | 定位文件 | 定位函数 |
|---------|---------|---------|
| sink 检测不对 | `kv_cache_eviction.py` | `set_first_chunk_info()` |
| 淘汰未触发 | `kv_cache_eviction.py` | `should_evict()` + `evict()` |
| OOM | 调低 `max_cache_tokens` / 加 `torch.cuda.empty_cache()` | |
| snapshot/restore 失败 | `cache_manager.py` | `snapshot()` / `restore()` |
| 质量崩溃 | 见"理论风险"章节，执行 Position 重映射等方案 | |

---

## 四、Phase 2: VLM Agent 导航闭环（预计 2-3 周）

> **审阅者强制要求**: 本阶段提前到 Benchmark 之前。先让无人机飞起来。
> **部署约束**: AutoDL 无 GUI，AirSim 必须在本地运行。两者通过 HTTP API 通信。

### 4.0 三种可行方案（按推荐程度排序）

| 方案 | 架构 | 优点 | 缺点 | 推荐度 |
|------|------|------|------|-------|
| **A: API Bridge（实时闭环）** | 本地 AirSim ↔ HTTP ↔ AutoDL VLM | 真正的闭环演示，学术价值最高 | 需处理网络延迟 | ⭐⭐⭐⭐⭐ |
| **B: 离线回放** | 本地录视频 → 上传 AutoDL → VLM 离线推理 | 最简单，不依赖网络 | 非真闭环，无法展示实时决策 | ⭐⭐⭐ |
| **C: 本地全部运行** | 本地同时跑 AirSim + VLM | 零网络延迟 | 需要本地有高端 GPU (≥16GB) | ⭐⭐ |

**推荐路线**: 先用方案 B 验证 Agent 逻辑 → 再用方案 A 实现实时闭环演示

---

### 4.1 方案 A: API Bridge 实时闭环（核心方案）

#### 整体架构

```
本地 Windows                                    AutoDL 云服务器
┌────────────────────┐                          ┌────────────────────────────┐
│ AirSim Client      │   POST /append_frame     │ FastAPI Server             │
│                    │ ──────(帧 JPEG bytes)───► │                            │
│ 1. 截帧            │                          │ 1. 解码帧                   │
│ 2. 发送帧到云端     │   POST /decide           │ 2. engine.append_frame()   │
│ 3. 请求决策        │ ──────(请求决策)────────► │ 3. smolagents Agent 推理    │
│ 4. 收到动作指令     │ ◄─────(JSON 动作)──────── │ 4. 返回动作 JSON            │
│ 5. 执行 AirSim API │                          │                            │
│ 6. 重复            │   GET /status             │ 5. get_cache_info() 监控    │
│                    │ ──────(状态查询)────────► │                            │
└────────────────────┘                          └────────────────────────────┘
```

#### 4.1.1 AutoDL 端：FastAPI 推理服务器

在 AutoDL 的 `temporal_encoding/` 下创建 `server_api.py`:

```python
"""
流式 VLM 推理 API 服务器。
运行在 AutoDL 上，接收帧数据，返回 Agent 决策。

启动: uvicorn server_api:app --host 0.0.0.0 --port 6006
(AutoDL 默认开放 6006 端口，可通过「自定义服务」获取公网地址)
"""
import io
import base64
import torch
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from model import StreamQwenModel, VideoStreamingInference, EvictionConfig

app = FastAPI(title="Streaming VLM Navigation API")

# ── 全局状态 ──
engine = None  # VideoStreamingInference
processor = None

class FrameRequest(BaseModel):
    """帧数据请求"""
    frame_b64: str            # JPEG 帧的 base64 编码
    fps: float = 2.0
    chunk_frames: int = 1     # 本次发送的帧数（如果是多帧 chunk）

class DecideRequest(BaseModel):
    """决策请求"""
    question: str = "Observe the current UAV camera view. Describe obstacles, free paths, and recommend the next navigation action (forward/left/right/hover). Be concise."
    max_new_tokens: int = 128
    temperature: float = 0.3

class DecideResponse(BaseModel):
    """决策响应"""
    action: str               # 推荐动作: forward/left/right/hover
    reasoning: str            # VLM 原始推理文本
    ttft: float               # 首 token 延迟
    cache_len: int            # 当前 cache 长度
    chunks_encoded: int       # 已编码 chunk 数

@app.on_event("startup")
def load_model():
    global engine, processor
    from transformers import AutoProcessor
    import os
    
    model_path = os.environ.get(
        "QWEN_MODEL_PATH", "/root/autodl-tmp/Qwen/Qwen2___5-VL-3B-Instruct"
    )
    
    print("Loading model...")
    model = StreamQwenModel.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    processor = AutoProcessor.from_pretrained(model_path)
    
    eviction_config = EvictionConfig(max_cache_tokens=100_000)
    engine = VideoStreamingInference(
        model, processor, device="cuda", eviction_config=eviction_config
    )
    print("Model loaded. Server ready.")

@app.post("/append_frame")
def append_frame(req: FrameRequest):
    """接收一帧/多帧，追加到视频流。"""
    img_bytes = base64.b64decode(req.frame_b64)
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    
    result = engine.append_frame(
        image, as_video=False, fps=req.fps,
        text_content="UAV camera frame."
    )
    info = engine.get_cache_info()
    return {
        "status": "ok",
        "message": result,
        "cache_len": info["cache_seq_length"],
        "chunks": info["chunks_encoded"],
    }

@app.post("/append_chunk")
def append_chunk(req: FrameRequest):
    """接收多帧 chunk（base64 编码的多张 JPEG 拼接, 用 '|||' 分隔）。"""
    parts = req.frame_b64.split("|||")
    frames = []
    for part in parts:
        img_bytes = base64.b64decode(part.strip())
        frames.append(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
    
    result = engine.append_video_chunk(frames, fps=req.fps)
    info = engine.get_cache_info()
    return {
        "status": "ok",
        "message": result,
        "cache_len": info["cache_seq_length"],
        "chunks": info["chunks_encoded"],
    }

@app.post("/decide", response_model=DecideResponse)
def decide(req: DecideRequest):
    """基于已累积的视频记忆，做一次导航决策。"""
    answer, metrics = engine.ask(
        question=req.question,
        max_new_tokens=req.max_new_tokens,
        do_sample=True,
        temperature=req.temperature,
    )
    
    # 简单解析动作关键词
    answer_lower = answer.lower()
    if "left" in answer_lower:
        action = "turn_left"
    elif "right" in answer_lower:
        action = "turn_right"
    elif "forward" in answer_lower or "ahead" in answer_lower:
        action = "move_forward"
    else:
        action = "hover"
    
    info = engine.get_cache_info()
    return DecideResponse(
        action=action,
        reasoning=answer.strip(),
        ttft=metrics["ttft"],
        cache_len=info["cache_seq_length"],
        chunks_encoded=info["chunks_encoded"],
    )

@app.get("/status")
def status():
    """查询推理引擎状态。"""
    info = engine.get_cache_info()
    vram = {}
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        vram = {
            "allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 3),
            "reserved_gb": round(torch.cuda.memory_reserved() / 1e9, 3),
        }
    return {"engine": info, "vram": vram}

@app.post("/reset")
def reset():
    """重置引擎状态。"""
    engine.reset()
    return {"status": "ok", "message": "Engine reset."}
```

**启动方式**（AutoDL 上）:
```bash
cd /root/autodl-tmp/diploma_project/temporal_encoding
pip install fastapi uvicorn python-multipart
uvicorn server_api:app --host 0.0.0.0 --port 6006
```

**获取公网地址**: AutoDL 控制台 → 容器实例 → 自定义服务 → 获取公网访问地址
（形如 `https://u123456-6006.westX.autodl.pro`）

#### 4.1.2 本地端：AirSim 客户端

在本地 Windows 机器上创建 `airsim_nav_client.py`:

```python
"""
AirSim 导航客户端。
本地运行，连接 AirSim + 远程 VLM API。

前置条件:
  1. AirSim + Unreal 环境已在本地启动
  2. AutoDL 上 FastAPI 服务已启动
  3. pip install airsim requests
"""
import time
import io
import base64
import requests
import airsim
from PIL import Image

# ── 配置 ──
VLM_API_BASE = "https://u123456-6006.westX.autodl.pro"  # 替换为实际地址
DECISION_INTERVAL = 5.0   # 每 5 秒决策一次
FLIGHT_SPEED = 2.0        # m/s
FRAME_INTERVAL = 0.5      # 每 0.5 秒截一帧发给 VLM

def frame_to_b64(airsim_response) -> str:
    """AirSim 截帧 → base64 JPEG。"""
    img = Image.frombytes("RGB",
        (airsim_response.width, airsim_response.height),
        airsim_response.image_data_uint8
    )
    # 降分辨率以减少传输量和 token 数
    img = img.resize((640, 480))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()

def send_frame(b64_frame: str):
    """发送一帧到远程 VLM。"""
    resp = requests.post(f"{VLM_API_BASE}/append_frame", json={
        "frame_b64": b64_frame, "fps": 2.0
    }, timeout=30)
    return resp.json()

def request_decision() -> dict:
    """请求 VLM Agent 做导航决策。"""
    resp = requests.post(f"{VLM_API_BASE}/decide", json={
        "question": (
            "You are a UAV navigation agent. Based on the accumulated video "
            "memory, observe the current scene and decide: "
            "forward / turn_left / turn_right / hover. "
            "Explain briefly why."
        ),
        "max_new_tokens": 128,
        "temperature": 0.3,
    }, timeout=60)
    return resp.json()

def execute_action(client: airsim.MultirotorClient, action: str):
    """在 AirSim 中执行动作。"""
    if action == "move_forward":
        client.moveByVelocityAsync(FLIGHT_SPEED, 0, 0, duration=2).join()
    elif action == "turn_left":
        yaw = client.simGetVehiclePose().orientation
        client.rotateByYawRateAsync(-30, duration=1).join()
    elif action == "turn_right":
        client.rotateByYawRateAsync(30, duration=1).join()
    elif action == "hover":
        client.hoverAsync().join()
    print(f"  Executed: {action}")

def main():
    # 连接 AirSim
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    client.takeoffAsync().join()
    print("UAV ready.")
    
    frame_count = 0
    last_decision_time = 0
    
    try:
        while True:
            # 1. 截帧
            responses = client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
            ])
            if not responses or responses[0].width == 0:
                time.sleep(0.1)
                continue
            
            b64 = frame_to_b64(responses[0])
            frame_count += 1
            
            # 2. 发帧给 VLM（累积视频记忆）
            result = send_frame(b64)
            print(f"Frame {frame_count}: cache_len={result.get('cache_len', '?')}")
            
            # 3. 周期性决策
            now = time.time()
            if now - last_decision_time >= DECISION_INTERVAL:
                # ⚠️ 思考前悬停
                client.hoverAsync()
                print(f"\n--- Requesting decision (frame {frame_count}) ---")
                
                decision = request_decision()
                action = decision.get("action", "hover")
                reasoning = decision.get("reasoning", "")
                ttft = decision.get("ttft", 0)
                
                print(f"  Action: {action}")
                print(f"  Reasoning: {reasoning[:200]}")
                print(f"  TTFT: {ttft:.3f}s")
                
                execute_action(client, action)
                last_decision_time = now
            
            time.sleep(FRAME_INTERVAL)
    
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        client.hoverAsync().join()
        client.armDisarm(False)
        client.enableApiControl(False)

if __name__ == "__main__":
    main()
```

#### 4.1.3 网络延迟预算

```
AutoDL 公网延迟 (上海→北京):     ~30-50 ms RTT
帧传输 (640×480 JPEG ~50KB):    ~20 ms
VLM append_frame:                ~200-500 ms (GPU 编码)
VLM decide (ask):                ~500-2000 ms (prefill + decode)
────────────────────────────────
单次决策总延迟:                    ~1-3 秒
```

对于 2-5 m/s 的 UAV + 5 秒决策间隔，这完全可接受（决策前已悬停）。

---

### 4.2 方案 B: 离线回放验证（先行方案，用于调试 Agent 逻辑）

在写 API Bridge 之前，先用离线回放验证 VLM 对导航场景的理解能力：

#### 步骤

1. **本地录制**: 在 AirSim 中手动飞行，每 0.5 秒截帧保存为 JPEG 序列
   ```python
   # 本地运行: record_flight.py
   for i in range(600):  # 300 秒
       response = client.simGetImages([...])
       img.save(f"flight_frames/{i:05d}.jpg")
       time.sleep(0.5)
   ```

2. **上传到 AutoDL**: `scp -r flight_frames/ root@autodl:~/autodl-tmp/data/`

3. **AutoDL 上离线推理**: 逐帧 `append_frame()` + 周期性 `ask()` 决策
   ```python
   # AutoDL 运行: offline_nav_eval.py
   for i, frame_path in enumerate(sorted(glob("flight_frames/*.jpg"))):
       frame = Image.open(frame_path)
       engine.append_frame(frame, as_video=False)
       
       if i % 10 == 0:  # 每 10 帧决策
           answer, _ = engine.ask("Describe the scene and suggest next action.")
           log.append({"frame": i, "decision": answer})
   ```

4. **分析日志**: 检查 VLM 的场景描述是否准确、决策是否合理

**价值**: 即使不做实时闭环，离线回放数据也足以写进论文作为 "VLM Agent 导航能力验证"。

---

### 4.3 实现 StreamingVLMModel（smolagents 集成）

在 AutoDL 的 `temporal_encoding/model/` 下创建 `streaming_vlm_agent.py`:

```python
from smolagents.models import Model, ChatMessage, MessageRole

class StreamingVLMModel(Model):
    """
    包装 VideoStreamingInference 为 smolagents Agent 模型。
    
    核心设计:
    - 共享 VLM: Agent 推理复用已加载的模型和 KV Cache
    - 视频记忆: Agent 每次推理都能访问已累积的视频帧
    - Snapshot/Restore: Agent 推理不污染视频 KV Cache
    """
    
    def __init__(self, engine, processor):
        super().__init__(
            flatten_messages_as_text=False,
            model_id="StreamingQwen2.5-VL-3B",
        )
        self.engine = engine
        self.processor = processor
    
    def generate(self, messages, stop_sequences=None, 
                 response_format=None, tools_to_call_from=None, **kwargs):
        prompt = self._messages_to_prompt(messages)
        response, metrics = self.engine.ask(
            question=prompt,
            max_new_tokens=kwargs.get("max_new_tokens", 512),
            update_state=False,
        )
        if stop_sequences:
            for seq in stop_sequences:
                if seq in response:
                    response = response[:response.index(seq)]
        return ChatMessage(role=MessageRole.ASSISTANT, content=response)
    
    def _messages_to_prompt(self, messages):
        parts = []
        for msg in messages:
            role = msg.role if hasattr(msg, 'role') else msg.get('role', 'user')
            content = msg.content if hasattr(msg, 'content') else msg.get('content', '')
            if isinstance(content, str):
                parts.append(f"[{role}]: {content}")
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get('type') == 'text':
                        parts.append(f"[{role}]: {item['text']}")
        return "\n".join(parts)
```

**注意**: smolagents 的 `ToolCallingAgent` 在 `/decide` 端点内部使用。
本地客户端不需要 smolagents，只需 `requests` 调用 API。

### 4.4 验证步骤（修订版）

```
Step 1: 方案 B 离线回放
  ├── 本地 AirSim 录制飞行视频帧序列
  ├── 上传到 AutoDL
  ├── 运行离线推理，验证 VLM 场景理解 + 决策质量
  └── 确认 KV Cache 淘汰在导航场景下正常工作

Step 2: 方案 A 服务端
  ├── AutoDL 上启动 FastAPI 服务 (server_api.py)
  ├── 用 curl 或 Python 脚本测试 /append_frame + /decide
  └── 确认延迟可接受 (<3s 单次决策)

Step 3: 方案 A 实时闭环
  ├── 本地启动 AirSim + airsim_nav_client.py
  ├── 简单场景 (空旷 + 障碍物) 测试
  ├── 录屏（AirSim 画面 + 终端日志）
  └── 这就是答辩演示视频

Step 4: 性能优化 (如有需要)
  ├── 帧分辨率降低 (1920→640) 减少 token 数和网络传输
  ├── 多帧 chunk 批量发送 (/append_chunk) 减少 API 调用次数
  └── 调整决策间隔 vs 飞行速度
```

### 4.5 显存预算

```
模型本体:        ~7.1 GB
KV Cache (100K): ~3.4 GB
FastAPI 开销:    ~0.1 GB (极小)
Agent 推理开销:   ~0.5 GB (snapshot 共享模型)
剩余安全余量:     ~13 GB → 足够
```

### 4.6 本地机器要求

```
本地 Windows 机器:
  ● 能运行 AirSim + Unreal Engine (任意 GPU 均可，只需渲染)
  ● 安装: pip install airsim requests Pillow
  ● 网络: 能访问 AutoDL 公网地址
  ● 不需要 ML 推理能力
```

---

## 五、Phase 3: 针对性评估（预计 1-2 周）

> 在 Agent 闭环跑通后，基于实际观察到的问题设计评估。

### 5.1 具身评估指标（核心）

在 AirSim 中定义 3 个标准化场景：

| 场景 | 环境 | 评估指标 |
|------|------|---------|
| 简单避障 | 空旷 + 5 棵树 | 碰撞率, 成功率, 决策延迟 |
| 走廊穿越 | 城市街道 | 碰撞率, 成功率, 决策延迟 |
| 有/无记忆对比 | 同一场景 | 流式记忆 vs 无记忆 的 Agent 决策质量差异 |

### 5.2 OVO-Bench 最小子集（辅助）

- 只跑每类任务 20-50 个样本
- 目的：证明流式追加不产生灾难性遗忘
- 对比：原生全量输入 vs 流式追加 vs 流式+淘汰

### 5.3 KV Cache 效率分析

| 实验 | 指标 | 预计时间 |
|------|------|---------|
| 不同 max_cache_tokens 的显存曲线 | Peak VRAM (GB) | 1 天 |
| 淘汰前后 TTFT/推理延迟 | Latency (ms) | 1 天 |
| 长程稳定性 (>300帧) | 显存 + TTFT 曲线 | 1 天 |

### 5.4 消融实验

| 变量 | 可选值 | 观察指标 |
|------|-------|---------|
| max_cache_tokens | 100K / 130K / 150K | 质量 + 显存 |
| chunk_frames | 2 / 4 | 质量 + 延迟 |
| fps | 1 / 2 / 4 | 质量 |
| Level 1 vs Level 2 | sink+window vs +均匀采样 | 质量对比 |

---

## 六、文件变更清单

### AutoDL 远程服务器 (`temporal_encoding/`)

| 文件 | 状态 | 说明 |
|------|------|------|
| `model/kv_cache_eviction.py` | ✅ 已有 | Level 1 为核心，Level 2 可选消融，Level 3 不使用 |
| `model/cache_manager.py` | ✅ 已有 | snapshot/restore + eviction 集成 |
| `model/video_stream_inference.py` | ✅ 已有 | 首 chunk auto-detect + 淘汰触发 |
| `model/streaming_vlm_agent.py` | 🆕 待创建 | StreamingVLMModel for smolagents |
| `server_api.py` | 🆕 待创建 | FastAPI 推理服务器（方案 A 核心） |
| `offline_nav_eval.py` | 🆕 待创建 | 离线导航回放验证（方案 B） |
| `test_eviction_exp_a.py` | 🆕 待创建 | 实验 A: sink 检测 |
| `test_eviction_exp_b.py` | 🆕 待创建 | 实验 B: OOM-Free |
| `test_eviction_exp_c.py` | 🆕 待创建 | 实验 C: 淘汰后质量 |

### 本地 Windows 机器

| 文件 | 状态 | 说明 |
|------|------|------|
| `airsim_nav_client.py` | 🆕 待创建 | AirSim 截帧 + API 调用 + 执行动作 |
| `record_flight.py` | 🆕 待创建 | AirSim 飞行录制（方案 B 用） |

---

## 七、总体执行顺序

```
Phase 1 (3-5 天):
  实验 A → 实验 B → 实验 C
  ↓ 如果质量崩溃 → 执行 Position 重映射等工程方案
  ↓ 全部通过

Phase 2 (2-3 周):
  smolagents 工具链验证 → StreamingVLMModel → AirSim 集成 → 演示视频
  ↓ 闭环跑通

Phase 3 (1-2 周):
  AirSim 具身评估 → OVO-Bench 最小子集 → KV Cache 效率分析 → 消融实验
  ↓ 数据齐全

Phase 4 (2-3 周):
  论文撰写 + 答辩 PPT
```

---

## 八、关键约束总结（红线，不可违反）

1. **单卡 4090 24GB** — 所有方案必须在此硬件上可运行
2. **不微调模型** — 只做工程/推理层优化，不改模型权重
3. **不使用注意力分数淘汰** — Level 1 为主力，不引入 attention computation overhead
4. **Agent 思考期必须悬停** — 不允许盲飞
5. **使用 ToolCallingAgent** — 不使用 CodeAgent（3B 模型代码生成不稳定）
6. **所有参数基于实测数据** — 不硬编码魔法数字，参照 test_step10 数据
7. **分体部署** — AirSim 在本地 Windows（有 GUI），VLM 在 AutoDL（无 GUI），通过 HTTP API 通信
8. **AutoDL 端口 6006** — 使用 AutoDL「自定义服务」功能暴露 FastAPI，不要尝试在 AutoDL 上运行 AirSim

---

## 九、开始执行

请从 **Phase 1 实验 A** 开始。创建 `test_eviction_exp_a.py`，验证 sink 自动检测机制。完成后输出报告，我审阅后进入实验 B。
