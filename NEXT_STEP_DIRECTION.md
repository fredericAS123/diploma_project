# 流式视频理解系统：下一步技术方向（修订版 v3）

> **日期**: 2026-02-22 (v3 revised)  
> **现状**: 已完成 Qwen2.5-VL 流式视频理解改造（连续 3D-RoPE + 逐 Chunk KV Cache 累积）  
> **约束**: 单卡 4090 (24GB)，无集群微调能力，有无人机实机  
> **核心原则**: 一个 VLM 身兼多职——同一个模型做感知 + 推理 + 工具调用，通过 smolagents 框架实现

---

## 〇、设计哲学：一个 VLM 身兼多职

v1 过于宏大（端到端导航），v2 过于保守（VLM 只做传感器 + if-else 规则）。**v3 找到平衡点**：

```
❌ v1 思路：VLM 做端到端导航 → 需要训练，落地极难
❌ v2 思路：VLM 只做传感器 + if-else 规则 → 太死板，学术价值低
✅ v3 思路：VLM 既做感知又做推理，通过 smolagents 框架调用工具 → 灵活且可实现
```

**核心洞察**：Qwen2.5-VL 不只是"会看的模型"，它底座是 Qwen2.5——一个**代码生成能力很强**的语言模型。所以它能：
1. **看**：处理视频帧，理解场景（视觉能力）
2. **想**：推理出应该做什么（语言推理能力）
3. **做**：生成工具调用代码，驱动无人机（代码生成能力）

**只需要一个模型**，不需要 VLM + 额外 LLM，4090 完全吃得消。

**smolagents 框架**（HuggingFace 官方）原生支持 VLM 做 Agent 引擎，提供：
- `CodeAgent`：VLM 生成 Python 代码调用工具
- `ToolCallingAgent`：VLM 生成 JSON 格式工具调用（对模型能力要求更低）
- `agent.run(task, images=[frame])`：原生传图给 Agent

**贡献点**：
1. 流式 3D-RoPE 编码 + KV Cache 管理（核心技术）
2. 标准 benchmark 评估（实验验证）
3. VLM Agent 导航闭环演示（smolagents + AirSim，应用价值）

---

## 一、你已经完成了什么（价值回顾）

先明确你手上已有的东西，这些是**真正有技术含量**的：

| 已完成模块 | 技术贡献 | 代码位置 |
|-----------|---------|---------|
| **StreamQwenModel** | 3-branch mRoPE 位置追踪，支持连续 chunk 追加 | `stream_qwen_model.py` |
| **KVCacheManager** | KV Cache 生命周期管理 + snapshot/restore | `cache_manager.py` |
| **VideoStreamingInference** | 完整流式推理引擎：append_frame → ask | `video_stream_inference.py` |
| **Web Demo** | Gradio 可视化演示界面 | `web_demo/` |

**现有系统的瓶颈**（也是你接下来的发力点）：

- ❌ **没有 KV Cache 淘汰**：~600 帧即 OOM（4090 上约 5-10 分钟视频流）
- ❌ **没有质量评估**：不知道流式追加模式相比原生输入质量差多少
- ❌ **没有应用闭环**：纯技术验证，缺少下游任务演示
- ✅ **已有 Agent 基础**：airsim_agent 项目已有 smolagents + AirSim wrapper tools

---

## 二、接下来只做三件事

**不多不少，恰好三件**——每件都可实现、可评估、可写进论文：

```
┌─────────────────────────────────────────────────────────────┐
│                                                              │
│   任务 1：KV Cache 淘汰策略（核心技术贡献）                    │
│           → 解决 OOM，让流式系统跑得更久                       │
│                                                              │
│   任务 2：流式视频理解质量评估（实验验证）                      │
│           → 用标准 benchmark 证明流式系统有效                  │
│                                                              │
│   任务 3：VLM Agent 导航闭环（smolagents 集成）               │
│           → VLM 一个模型：看 + 想 + 做，驱动 AirSim 导航      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 三、任务 1：KV Cache 多模态淘汰策略

### 3.1 为什么这是核心贡献

- 这是**你的系统当前最紧迫的瓶颈**——不解决就无法长时间运行
- 现有 KV Cache 淘汰方法（StreamingLLM、H₂O）**都是纯文本设计**，没考虑多模态
- 多模态 KV Cache 淘汰是**2024-2026 的热点方向**，论文少、坑位多
- **实现难度适中**：不需要改模型架构，只需要在 `cache_manager.py` 加淘汰逻辑

### 3.2 文献定位（你要对标/超越的工作）

#### 核心文献 ⭐⭐⭐⭐⭐

**① LOOK-M: Look-Once Optimization in KV Cache for Efficient Multimodal Long-Context Inference**
> Wan et al., arXiv:2406.18139, 2024

- **核心发现**：多模态 LLM 在 prefill 阶段对文本 token 的注意力远高于图像 token
- **方法**：text-prior 策略压缩视觉 KV，KV pair merging 补偿信息损失
- **效果**：KV Cache 减少 80%，解码加速 1.5x，性能持平或更好
- **对你的意义**：这是**目前最有影响力**的多模态 KV Cache 论文，是你的直接对标

**② HAE: Hierarchical Adaptive Eviction for KV Cache in Multimodal Language Models**
> Ma et al., arXiv:2602.02197, 2026

- **核心方法**：双阶段——Prefill 阶段做 Dual-Attention Pruning（利用视觉 token 稀疏性 + 注意力方差），Decode 阶段做 Dynamic Eviction（灵感来自 OS 回收站）
- **效果**：KV Cache 减 41%，精度仅降 0.3%，推理加速 1.5x
- **对你的意义**：2026 年 2 月刚发表，**非常新**，说明这个方向正热

**③ StreamingLLM: Efficient Streaming Language Models with Attention Sinks**
> Xiao et al., arXiv:2309.17453, ICLR 2024

- **核心思想**：保留首 N 个 "attention sink" token + 最近 N 个 token 的滑动窗口
- **效果**：在无限长文本上稳定工作，无需微调
- **对你的意义**：最简单的 baseline，你的第一版淘汰策略应该基于它

#### 补充文献 ⭐⭐⭐⭐

| 论文 | arXiv | 核心贡献 | 对你的意义 |
|------|-------|---------|-----------|
| **H₂O** (Heavy-Hitter Oracle) | 2306.14048 (NeurIPS 2023) | 按注意力分数动态保留重要 token | 进阶淘汰策略 |
| **SnapKV** | 2404.14469 | 每个注意力头独立选择保留 KV | Query-Aware 压缩 |
| **FastV** | 2403.06764 | 第 2 层后剪枝 50% 视觉 token | 从源头减少 KV |
| **AccKV** | 2511.11106 | 音视频 LLM 的自适应 KV 压缩 | 多模态 KV 的另一个参考 |

### 3.3 你的淘汰策略设计（从简到难，三级递进）

#### Level 1：Attention Sink + Sliding Window（1 周可完成）

最简单的 baseline，直接可用：

```python
# 在 cache_manager.py 中新增

def evict_sliding_window(self, sink_size=128, window_size=4096):
    """
    保留:
      - 首 sink_size 个 token (system prompt + 首帧 = attention sink)
      - 最近 window_size 个 token (最新帧的上下文)
    淘汰:
      - 中间所有 token
    """
    seq_len = self.get_seq_length()
    if seq_len <= sink_size + window_size:
        return  # 未超预算
    
    new_cache = DynamicCache()
    for layer_idx in range(len(self._cache)):
        k, v = self._cache[layer_idx]
        # 拼接 sink 和 recent
        k_new = torch.cat([k[:, :, :sink_size, :], 
                           k[:, :, -window_size:, :]], dim=2)
        v_new = torch.cat([v[:, :, :sink_size, :], 
                           v[:, :, -window_size:, :]], dim=2)
        new_cache.update(k_new, v_new, layer_idx)
    self._cache = new_cache
```

**预算计算**：
- `sink_size=128` + `window_size=4096` = 4224 tokens → KV 约 240 MB
- **显存完全可控**，不再 OOM

#### Level 2：视觉-文本分离淘汰（参考 LOOK-M，2 周可完成）

关键洞察：视觉 token 和文本 token 的重要性分布不同，应分别处理。

```python
def evict_modality_aware(self, visual_budget=1024, text_budget=512):
    """
    分模态淘汰:
    - 视觉 token: 按注意力分数保留 Top-K (重要帧的 token 保留)
    - 文本 token: 保留 sink + recent window
    
    需要: token_type_mask 记录每个位置是视觉还是文本
    """
    # 1. 用 attention score 评估每个视觉 token 的重要性
    # 2. 保留 Top-K 视觉 token（重要帧自动被保留）
    # 3. 文本 token 用 sink + window 策略
    ...
```

**这个策略的创新点**：
- StreamingLLM / H₂O 都是纯文本设计，**不区分模态**
- LOOK-M 是离线场景，**不是流式场景**
- **你的系统是流式 + 多模态**，需要一个新的淘汰策略 → 这就是你的贡献

#### Level 3：帧级重要性评分淘汰（3 周可完成，若时间充裕）

比 token 级更高层——以"帧"为单位评估重要性：

```python
def evict_frame_level(self, frame_budget=50):
    """
    帧级淘汰:
    - 每帧的视觉 token 作为一组
    - 用累积注意力分数为每帧打分
    - 保留最重要的 N 帧 + 最近 M 帧
    - 淘汰低重要性的中间帧
    
    优点: 保持帧内 token 的完整性（空间关系不被破坏）
    """
    ...
```

### 3.4 实验设计（简单明确）

| 实验 | 指标 | 难度 | 预计时间 |
|------|------|------|---------|
| 不同 window_size 下的显存曲线 | Peak VRAM (GB) | ⭐ | 1 天 |
| 不同淘汰策略下的质量对比 | Accuracy on benchmark | ⭐⭐ | 2-3 天 |
| 淘汰前后的 TTFT/推理延迟 | Latency (ms) | ⭐ | 1 天 |
| 长程稳定性测试（>1000帧） | 显存/精度曲线 | ⭐⭐ | 1 天 |
| 消融：sink_size 和 window_size | Accuracy + VRAM | ⭐⭐ | 2 天 |

**核心图表**（写论文时直接用）：
1. **显存-帧数曲线**：无淘汰 vs Level 1 vs Level 2（证明淘汰有效）
2. **质量-压缩率曲线**：不同压缩比下的准确率变化（证明质量可控）
3. **模态分离 vs 统一淘汰**的对比表格（证明分模态有意义）

---

## 四、任务 2：流式视频理解质量评估

### 4.1 为什么需要评估

你目前只有效率指标（TTFT、显存），没有**质量指标**。论文需要回答：
- 流式追加帧模式 vs 原生全量输入，质量差多少？
- KV Cache 淘汰后质量损失几何？
- 帧数增长时质量如何衰减？

### 4.2 推荐 Benchmark（按易用性排序）

#### 首选：OVO-Bench ⭐⭐⭐⭐⭐

> **OVO-Bench: How Far is Your Video-LLMs from Real-World Online Video Understanding?**  
> Li et al., arXiv:2501.05510, **CVPR 2025**

- **设计理念**：专门评估**在线视频理解**中的时间感知能力
- **三类场景**：
  - Backward tracing（回溯过去事件）
  - Real-time understanding（理解当前发生的事）
  - Forward active responding（等待未来信息再回答）
- **规模**：644 个视频，~2800 人工标注 QA 对，12 个子任务
- **为什么适合你**：这就是为流式视频 LLM 设计的 benchmark，**CVPR 2025 刚发表**
- **适配方式**：逐帧 `append_frame()` 喂入，在指定时间戳调用 `ask_choice()`

#### 次选：OVBench ⭐⭐⭐⭐⭐

> **Online Video Understanding: OVBench and VideoChat-Online**  
> Huang et al., arXiv:2501.00584, **CVPR 2025**

- **6 类核心任务**覆盖过去/现在/未来三个时间维度，16 个子任务
- **提出 Pyramid Memory Bank (PMB)**：有效保留视频流中的关键时空信息
- **对你的意义**：OVBench 是另一个 CVPR 2025 的在线视频 benchmark，与 OVO-Bench 互补
- **实验价值**：可以报告在两个 benchmark 上的结果，增加说服力

#### 补充：VStream-QA ⭐⭐⭐⭐

> **Flash-VStream: Memory-Based Real-Time Understanding for Long Video Streams**  
> Zhang et al., arXiv:2406.08085, 2024

- 首个流式视频 QA benchmark
- 模拟异步问答：视频播放时，问题在任意时刻提出
- 可作为额外的评估点

### 4.3 评估实验设计

**对比实验（最重要的表格）**：

| 方法 | 模式 | OVO-Bench | OVBench | VRAM | TTFT |
|------|------|-----------|---------|------|------|
| Qwen2.5-VL (原生) | 全量输入 | baseline | baseline | 动态 | 高 |
| Ours (无淘汰) | 流式追加 | ? | ? | 线性增长 | 低 |
| Ours + Sink-Window | 流式+淘汰 | ? | ? | 恒定 | 低 |
| Ours + Modal-Aware | 流式+分模态淘汰 | ? | ? | 恒定 | 低 |

**消融实验**：

| 变量 | 可选值 | 观察指标 |
|------|-------|---------|
| chunk_size | 1 / 2 / 4 / 8 帧 | 质量 + 效率 |
| window_size | 1024 / 2048 / 4096 | 质量 + 显存 |
| sink_size | 64 / 128 / 256 | 质量 |
| 淘汰频率 | 每帧 / 每 10 帧 / 每 50 帧 | 质量 + 延迟 |
| fps | 0.5 / 1 / 2 | 质量 |

### 4.4 实施步骤（简单明确）

```
Step 1: 下载 OVO-Bench 数据集和评估脚本（GitHub 开源）
Step 2: 写一个 adapter 脚本将视频逐帧喂入你的 append_frame()
Step 3: 在指定时间戳调用 ask_choice()，收集结果
Step 4: 用 OVO-Bench 官方评估脚本计算分数
Step 5: 跑完原生/流式/淘汰 三组，画对比表
```

**预计时间**：1-2 周（大部分时间在等 GPU 跑完实验）

---

## 五、任务 3：VLM 一体化 Agent 导航（smolagents 集成）

### 5.1 核心思路：VLM = 感知 + 推理 + 规划，一个模型全搞定

上一版方案用 if-else 规则引擎做规划，太死板。更好的做法：

```
❌ 旧方案：VLM(看) → 文本描述 → if-else 规则 → 动作      （VLM 只做传感器）
✅ 新方案：VLM(看 + 想 + 做) → 直接生成工具调用代码 → 执行  （VLM 就是 Agent 大脑）
```

**为什么可行**：
- smolagents **原生支持 VLM 作为 Agent 引擎**（`TransformersModel` 自动识别 VLM）
- `agent.run(task, images=[pil_image])` 原生传图给 Agent
- Qwen2.5-VL 底座是 Qwen2.5，**代码生成能力足够**驱动工具调用
- **只有一个模型在 VRAM 中**，不需要 VLM + 额外 LLM
- 你已有的 `VideoStreamingInference` 提供视频记忆，VLM Agent 能"回忆"之前看到的场景

**参考文献**（这种"VLM 直接做 Agent"的范式有学术支撑）：

| 论文 | 核心思路 | 对你的意义 |
|------|---------|-----------|
| **SayNav** (arXiv:2309.04077) | LLM + 增量 Scene Graph → 工具调用 | Agent + 场景图的经典范式 |
| **USS-Nav** (arXiv:2602.00708) | UAV 场景图 + LLM 粗到细探索, 15Hz | VLM 可替代纯 LLM 做探索 |
| **Graph2Nav** (arXiv:2504.16782) | 3D 关系图 + SayNav 集成导航 | 感知→图→规划 仍是主流 |
| **smolagents 官方 VLM 示例** | CodeAgent + VLM 做网页浏览 | 证明 VLM 可驱动工具链 |

### 5.2 两层架构：流式视频记忆 + Agent 推理

```
┌──────────────────────────────────────────────────────────────┐
│              VLM 一体化 Agent 架构                             │
│                                                               │
│  ┌─────────────────────────────────────────────────┐         │
│  │  Layer 1: 流式视频记忆（你已有的系统）              │         │
│  │                                                   │         │
│  │  AirSim 相机 → append_frame() → KV Cache 累积     │         │
│  │  (StreamQwenModel + KVCacheManager)               │         │
│  │  VLM "记住"了飞行过程中看到的一切                   │         │
│  └───────────────────┬─────────────────────────────┘         │
│                      │ 共享同一个 VLM 模型                    │
│                      ↓                                        │
│  ┌─────────────────────────────────────────────────┐         │
│  │  Layer 2: Agent 推理（smolagents CodeAgent）       │         │
│  │                                                   │         │
│  │  StreamingVLMModel.generate(messages)             │         │
│  │    → 内部调用 engine.ask(agent_message)            │         │
│  │    → VLM 基于视频记忆 + 当前问题 生成代码          │         │
│  │    → 代码调用 tools: move(), turn(), detect()     │         │
│  │    → 执行结果反馈 → VLM 生成下一步                 │         │
│  └───────────────────┬─────────────────────────────┘         │
│                      │                                        │
│                      ↓                                        │
│  ┌─────────────────────────────────────────────────┐         │
│  │  Layer 3: 工具执行（AirSim API）                   │         │
│  │                                                   │         │
│  │  @tool move_forward(distance)                     │         │
│  │  @tool turn_left(degrees)                         │         │
│  │  @tool get_scene_description() → 调用 ask()       │         │
│  │  @tool detect_objects() → 调用 ask_choice()       │         │
│  │  @tool fly_to(x, y, z)                            │         │
│  └─────────────────────────────────────────────────┘         │
└──────────────────────────────────────────────────────────────┘
```

### 5.3 关键实现：自定义 StreamingVLMModel

smolagents 允许继承 `Model` 基类实现自定义引擎。**核心思路**：把你已有的 `VideoStreamingInference.ask()` 包装成 smolagents 的 `generate()` 接口。

```python
from smolagents.models import Model, ChatMessage, MessageRole

class StreamingVLMModel(Model):
    """
    将流式 VLM 推理引擎包装为 smolagents 的 Agent 模型。
    
    关键设计：
    - VLM 模型共享：Agent 推理复用 VideoStreamingInference 中已加载的模型
    - 视频记忆：Agent 每次推理都能访问已累积的 KV Cache（看过的所有帧）
    - Snapshot/Restore：Agent 推理不污染视频 KV Cache
    """
    
    def __init__(self, engine: VideoStreamingInference, processor):
        super().__init__(
            flatten_messages_as_text=False,  # 保留图片，不展平
            model_id="StreamingQwen2.5-VL",
        )
        self.engine = engine
        self.processor = processor
    
    def generate(
        self,
        messages: list,
        stop_sequences: list[str] | None = None,
        response_format=None,
        tools_to_call_from=None,
        **kwargs,
    ) -> ChatMessage:
        """
        Agent 推理的核心方法。
        
        将 smolagents 的 messages 转换为 engine.ask() 调用，
        VLM 基于累积的视频记忆 + 当前 Agent 上下文生成回复。
        """
        # 1. 提取 messages 中的文本内容（工具描述、历史对话等）
        prompt = self._messages_to_prompt(messages)
        
        # 2. 如果 messages 中有新图片，先 append 到视频记忆
        images = self._extract_images(messages)
        for img in images:
            self.engine.append_frame(img, as_video=False)
        
        # 3. 调用 engine.ask()——VLM 在完整视频记忆上推理
        #    snapshot/restore 自动保护视频 KV Cache
        response, metrics = self.engine.ask(
            question=prompt,
            max_new_tokens=kwargs.get("max_new_tokens", 1024),
            update_state=False,  # 不污染视频缓存
        )
        
        # 4. 处理 stop_sequences
        if stop_sequences:
            for seq in stop_sequences:
                if seq in response:
                    response = response[:response.index(seq)]
        
        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content=response,
        )
    
    def _messages_to_prompt(self, messages) -> str:
        """将 smolagents 的 message list 拼接为纯文本 prompt。"""
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
    
    def _extract_images(self, messages) -> list:
        """从 messages 中提取 PIL Image 对象。"""
        images = []
        for msg in messages:
            content = msg.content if hasattr(msg, 'content') else msg.get('content', '')
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get('type') == 'image':
                        img = item.get('image')
                        if img is not None:
                            images.append(img)
        return images
```

### 5.4 Agent 工具定义

```python
from smolagents import tool, CodeAgent

# ── 感知类工具（调用流式 VLM）──

@tool
def describe_scene() -> str:
    """
    观察当前无人机摄像头画面，描述场景中的障碍物、
    可通行方向、地面状况。基于持续积累的视频记忆。
    """
    response, _ = streaming_engine.ask(
        "Describe the current scene briefly: obstacles, free paths, terrain.",
        max_new_tokens=128, update_state=False
    )
    return response

@tool
def check_direction(direction: str) -> str:
    """
    检查指定方向（left/right/forward/above/below）是否安全可通行。
    Args:
        direction: 要检查的方向
    Returns:
        该方向的安全性描述
    """
    response, _ = streaming_engine.ask(
        f"Is the {direction} direction safe and clear for the UAV to fly? Answer YES or NO with brief reason.",
        max_new_tokens=64, update_state=False
    )
    return response

# ── 控制类工具（调用 AirSim API）──

@tool
def move_forward(distance_meters: float) -> str:
    """向前飞行指定距离（米）。"""
    client.moveByVelocityAsync(distance_meters, 0, 0, duration=distance_meters/2)
    return f"Moved forward {distance_meters}m"

@tool
def turn_left(degrees: float) -> str:
    """向左旋转指定角度。"""
    current_yaw = get_current_yaw()
    client.rotateToYawAsync(current_yaw - degrees)
    return f"Turned left {degrees}°"

@tool
def turn_right(degrees: float) -> str:
    """向右旋转指定角度。"""
    current_yaw = get_current_yaw()
    client.rotateToYawAsync(current_yaw + degrees)
    return f"Turned right {degrees}°"

@tool
def hover() -> str:
    """悬停在当前位置（不确定时使用）。"""
    client.hoverAsync()
    return "Hovering in place"
```

### 5.5 Agent 创建与运行

```python
# 初始化（只加载一次模型）
from temporal_encoding.model import StreamQwenModel, VideoStreamingInference

model = StreamQwenModel.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", ...)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
streaming_engine = VideoStreamingInference(model, processor)

# 包装为 smolagents Model
agent_model = StreamingVLMModel(engine=streaming_engine, processor=processor)

# 创建 Agent
agent = CodeAgent(
    tools=[describe_scene, check_direction, move_forward, 
           turn_left, turn_right, hover],
    model=agent_model,
    max_steps=10,  # 每轮最多 10 步推理
)

# 导航主循环
for frame in camera_stream:
    # 1. 持续积累视频记忆
    streaming_engine.append_frame(frame, as_video=True, fps=2.0)
    
    # 2. 每 N 帧让 Agent 做一次决策
    if frame_count % 15 == 0:  # 约每 7.5 秒
        result = agent.run(
            "You are a UAV navigation agent. Based on your video memory of the flight, "
            "observe the current scene and decide the next action to navigate safely. "
            "Avoid obstacles. Prefer open paths.",
            images=[frame],  # 当前帧作为额外视觉输入
        )
```

### 5.6 为什么这比 if-else 规则更好

| 对比项 | if-else 规则引擎 | VLM Agent (smolagents) |
|--------|-----------------|------------------------|
| 灵活性 | 固定规则，难扩展 | VLM 自主推理，能处理意外情况 |
| 场景理解 | 靠关键词匹配 | VLM 真正"看"到了画面 |
| 多步推理 | 不支持 | Agent 可多步：先观察→再检查→再行动 |
| 显存 | VLM + 规则（无额外开销） | VLM 一个模型（无额外开销） |
| 工程量 | 简单 | 中等（需写 tools + custom Model） |
| 学术价值 | 低（谁都能写 if-else） | 高（VLM Agent 是热门方向） |
| 可演示性 | 死板 | Agent 的推理过程可视化，演示效果好 |

### 5.7 实现注意事项

**显存管理**：
- 模型 ~14GB (7B FP16)，剩余 ~10GB 给 KV Cache
- Agent 推理通过 `snapshot/restore` 保护视频缓存，不额外消耗
- 配合 KV 淘汰策略（任务 1），显存完全可控

**推理延迟**：
- 每次 Agent step: `ask()` 约 0.5-1s（prefill + decode）
- 一个完整 Agent 决策（3-5 steps）: 约 2-5s
- 对于 UAV 导航（飞行速度 2-5m/s），每 5s 做一次决策完全够

**7B 模型的 Agent 能力**：
- Qwen2.5-VL-7B 基于 Qwen2.5-7B，代码生成能力中等
- 对于简单的 5-6 个工具调用足够
- 如果 7B 不够可用 `ToolCallingAgent`（比 `CodeAgent` 对模型要求低）
- 实在不行可以退回 3B 模型 + 更简单的工具集

### 5.8 备选方案：标准 TransformersModel（更简单但无流式记忆）

如果自定义 `StreamingVLMModel` 太麻烦，可以先用标准方案验证：

```python
from smolagents import TransformersModel, CodeAgent

# 标准方式——每次推理无状态，不共享 KV Cache
model = TransformersModel(
    model_id="Qwen/Qwen2.5-VL-7B-Instruct",
    device_map="auto",
    torch_dtype="bfloat16",
    trust_remote_code=True,
    max_new_tokens=1024,
)

agent = CodeAgent(tools=[...], model=model)
agent.run("Navigate safely", images=[current_frame])  # 每次只看当前帧
```

**优点**：零代码改造，开箱即用  
**缺点**：每次 Agent step 只看当前帧，没有"视频记忆"，无法回忆之前看到的场景

**推荐路线**：先用标准 `TransformersModel` 验证 Agent 工具链可用 → 再换 `StreamingVLMModel` 加入视频记忆

### 5.9 演示实验设计

| 场景 | AirSim 环境 | 预期效果 | 难度 |
|------|------------|---------|------|
| 简单避障 | 空旷场地 + 几棵树 | Agent: describe→check_direction→move | ⭐⭐ |
| 走廊穿越 | 城市街道 | Agent 多步推理：观察→转向→前进→再观察 | ⭐⭐⭐ |
| 对比演示 | 同一场景 | 有流式记忆 vs 无流式记忆 的 Agent 对比 | ⭐⭐⭐ |

**交付物**：
- 3-5 分钟演示视频（展示 Agent 的推理过程日志 + 飞行轨迹）
- Agent 每步推理的可视化（smolagents 自带日志，非常适合答辩展示）

---

## 六、实施路线图（总计 7-9 周）

```
========== Phase 1: KV Cache 淘汰（2 周）==========

Week 1:
  ├── 实现 Attention Sink + Sliding Window 淘汰
  │   └── 修改 cache_manager.py
  ├── 测试不同 window_size 的显存/质量
  └── 跑 >1000 帧的长程稳定性测试

Week 2:
  ├── 实现 视觉-文本分离淘汰（Level 2）
  ├── 对比 Level 1 vs Level 2 的质量差异
  └── 绘制 显存-帧数 / 质量-压缩率 图表

交付: 改进的 cache_manager.py + 淘汰策略对比报告

========== Phase 2: Benchmark 评估（2-3 周）==========

Week 3:
  ├── 下载 OVO-Bench 数据集 + 评估脚本
  ├── 写 adapter 脚本适配你的流式接口
  └── 跑 Qwen2.5-VL 原生模式 baseline

Week 4:
  ├── 跑 流式模式（无淘汰 / 有淘汰）
  ├── 完成对比表格
  └── 消融实验（chunk_size, window_size, fps）

Week 5 (可选):
  └── 在 OVBench 上补充实验

交付: 完整评估报告 + 对比表格 + 消融分析

========== Phase 3: VLM Agent 导航（2-3 周）==========

Week 6:
  ├── 实现 StreamingVLMModel(Model) 自定义类
  │   └── 包装 engine.ask() 为 smolagents generate()
  ├── 定义 AirSim 导航工具集 (5-6 个 @tool)
  └── 简单场景测试 Agent 推理能力

Week 7:
  ├── AirSim 视频流 → append_frame 桥接
  ├── 导航主循环：持续积帧 + 周期性 Agent 决策
  └── 录制 Agent 推理过程 + 飞行轨迹演示视频

交付: StreamingVLMModel + 导航工具集 + 演示视频

========== Phase 4: 论文撰写（2-3 周）==========

Week 8-10:
  ├── 整理实验数据
  ├── 撰写论文/毕设报告
  └── 制作答辩 PPT
```

---

## 七、论文结构建议

### 标题

> **基于流式 KV Cache 管理的视觉语言 Agent 用于无人机实时导航**

或英文：

> **Streaming VLM Agent with Efficient KV Cache Management for Real-Time UAV Navigation**

### 结构

```
1. 引言
   - 流式视频理解的需求（长程 UAV 导航需要持续感知）
   - 现有方法局限：离线输入、KV Cache 无限增长、纯文本淘汰策略不适用多模态
   - 本文贡献：(1) 流式 3D-RoPE 编码 (2) 多模态 KV Cache 淘汰 (3) VLM Agent 导航闭环

2. 相关工作
   - 2.1 流式视频理解 (Flash-VStream, VideoChat-Online, Event-VStream)
   - 2.2 KV Cache 压缩 (StreamingLLM, H₂O, LOOK-M, HAE)
   - 2.3 VLM Agent 导航 (SayNav, USS-Nav, Graph2Nav, smolagents)

3. 方法
   - 3.1 连续流式 3D-RoPE 编码（你已完成的核心）
   - 3.2 KV Cache 多模态感知淘汰策略（核心贡献）
      - 3.2.1 Attention Sink + Sliding Window (baseline)
      - 3.2.2 视觉-文本分离淘汰 (your contribution)
      - 3.2.3 帧级重要性评分 (if time permits)
   - 3.3 VLM-as-Agent: 流式视频记忆驱动的导航 Agent
      - 3.3.1 StreamingVLMModel 架构设计
      - 3.3.2 视频记忆与 Agent 推理的交互机制

4. 实验
   - 4.1 流式视频理解质量 (OVO-Bench, OVBench)
   - 4.2 KV Cache 效率分析 (显存、延迟、吞吐量)
   - 4.3 消融实验 (cache 策略、chunk 大小、窗口大小)
   - 4.4 VLM Agent 导航实验 (AirSim: 成功率、碰撞率、推理日志)

5. 结论
```

### 你的核心创新点（投稿/答辩时的卖点）

| # | 创新点 | 一句话说明 |
|---|--------|-----------|
| 1 | **流式 3D-RoPE 连续编码** | 首次实现 Qwen2.5-VL 的逐帧追加式 mRoPE 位置编码 |
| 2 | **多模态感知 KV Cache 淘汰** | 区分视觉/文本 token 的差异化淘汰策略（StreamingLLM 和 LOOK-M 都没做的组合） |
| 3 | **流式 VLM 在线视频理解评估** | 在 OVO-Bench/OVBench 上系统评估流式 vs 离线的质量差距 |
| 4 | **VLM Agent 导航闭环** | 同一个 VLM 既做流式感知又做 Agent 推理，通过 smolagents 调用导航工具 |
| 5 | **单卡 4090 长程部署** | 完整的工程方案：恒定显存 + 实时推理 + Agent 导航 |

**创新 2 为什么有价值**：
- StreamingLLM (ICLR 2024)：纯文本，不区分模态
- H₂O (NeurIPS 2023)：纯文本，按注意力淘汰
- LOOK-M (2024)：多模态但是离线（一次性输入所有图像）
- HAE (2026)：多模态但不是流式场景
- **你的工作**：**流式 + 多模态 + 持续追加**，这个组合目前没人做过

**创新 4 为什么有价值**：
- SayNav/Graph2Nav：用**独立 LLM**做规划，感知和规划是两个模型
- USS-Nav：用 LLM 做粗到细探索，VLM 只提供语义标签
- **你的工作**：**一个 VLM 同时做视频流感知 + Agent 推理**，通过 KV Cache 共享"视频记忆"，这是一个新的架构范式

---

## 八、关键参考文献总表

### A. KV Cache 管理（直接相关）

| # | 论文 | 年份/会议 | arXiv | 核心贡献 |
|---|------|---------|-------|---------|
| 1 | **LOOK-M** | 2024 | 2406.18139 | 多模态 KV Cache text-prior 压缩 + KV merging |
| 2 | **HAE** | 2026 | 2602.02197 | 多模态分层自适应淘汰 (Dual-Attention + OS Recycle) |
| 3 | **StreamingLLM** | ICLR 2024 | 2309.17453 | Attention Sink + 无限流式推理 |
| 4 | **H₂O** | NeurIPS 2023 | 2306.14048 | Heavy-Hitter 动态淘汰 |
| 5 | **SnapKV** | 2024 | 2404.14469 | 观测窗口 per-head KV 选择 |
| 6 | **FastV** | 2024 | 2403.06764 | 视觉 Token 早期层剪枝 |
| 7 | **AccKV** | 2025 | 2511.11106 | 音视频 LLM 自适应 KV 优化 |

### B. 流式视频理解（你的赛道）

| # | 论文 | 年份/会议 | arXiv | 核心贡献 |
|---|------|---------|-------|---------|
| 8 | **OVO-Bench** | CVPR 2025 | 2501.05510 | 在线视频时间感知 benchmark |
| 9 | **OVBench + VideoChat-Online** | CVPR 2025 | 2501.00584 | 在线视频 benchmark + Pyramid Memory Bank |
| 10 | **Flash-VStream** | 2024 | 2406.08085 | 记忆机制 + VStream-QA benchmark |
| 11 | **Event-VStream** | 2026 | 2601.15655 | 事件驱动帧选择 + 持久化记忆 |
| 12 | **VideoLLM-online (LIVE)** | CVPR 2024 | 2406.11816 | 流式视频 LLM 训练框架 |

### C. VLM Agent + 导航（应用参考）

| # | 论文 | 年份/会议 | arXiv | 核心贡献 |
|---|------|---------|-------|---------|
| 13 | **USS-Nav** | 2026 | 2602.00708 | UAV 场景图 + 15Hz 轻量导航 |
| 14 | **SayNav** | 2023 | 2309.04077 | LLM + 增量 Scene Graph + 低层规划 |
| 15 | **Graph2Nav** | 2025 | 2504.16782 | 3D 语义关系图 + SayNav 集成 |
| 16 | **ConceptGraphs** | 2023 | 2309.16650 | 开放词汇 3D 场景图 |
| 17 | **smolagents** | HuggingFace | [文档](https://huggingface.co/docs/smolagents) | VLM 原生支持的 Agent 框架 |

---

## 九、风险与缓解

| 风险 | 概率 | 缓解策略 |
|------|------|---------|
| OVO-Bench 数据集太大，跑不完 | 中 | 选子集（每类任务 50 个样本） |
| 分模态淘汰后质量下降明显 | 中 | 保守设置 budget；调整视觉/文本比例 |
| AirSim 场景搭建耗时 | 低 | 用默认 City/Mountain 环境，不自建 |
| 7B 模型 Agent 能力不足 | 中 | 退回 `ToolCallingAgent`（比 CodeAgent 要求低）或换 3B |
| VLM Agent 推理延迟过高 | 中 | 减少 max_steps；限制 max_new_tokens；增大决策间隔 |
| `StreamingVLMModel` 封装复杂 | 中 | 先用标准 `TransformersModel` 验证工具链，再加流式记忆 |
| 论文投稿被拒（创新不够） | 中 | 聚焦"流式+多模态+Agent"的三重 gap，这个组合确实没人做 |

---

## 十、总结

### 一句话定位

> **你在做"一个 VLM 搞定一切的流式导航系统"——同一个 Qwen2.5-VL 既做流式视频感知（KV Cache 累积），又做 Agent 推理（smolagents 工具调用），在单卡 4090 上完成从看到做的完整闭环。**

### 三个核心问题和答案

| 问题 | 答案 |
|------|------|
| 你做了什么？ | 把 Qwen2.5-VL 改成了流式视频理解模式 + 包装为 smolagents Agent 引擎 |
| 为什么有用？ | 一个 VLM 同时完成视频感知和导航推理，单卡 4090 即可长程运行 |
| 怎么证明有用？ | OVO-Bench/OVBench 跑分 + AirSim VLM Agent 导航演示 |

### 今天就可以开始做

1. 打开 `cache_manager.py`，写 `evict_sliding_window()` 方法
2. `pip install smolagents`，跑一个最简单的 `TransformersModel` + `CodeAgent` 示例
3. 去 OVO-Bench 的 GitHub 下载数据集
