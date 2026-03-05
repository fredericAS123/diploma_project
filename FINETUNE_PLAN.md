# StreamUAV 两阶段微调方案

> **主选模型**: Qwen2.5-VL-7B-Instruct (A800 微调) | Qwen2.5-VL-3B-Instruct (效率对比/资源受限部署)
> **目标**: 流式无人机视频事件感知 → 论文投稿 IEEE RAL / ACM MM 2026
> **微调硬件**: A800 (80 GB) × 1
> **推理硬件**: AutoDL RTX 4090 (24 GB) × 1
> **数据**: Stage 1 — VideoChatOnline-IT / OVO-Bench → Stage 2 — StreamUAV-QA v2
> **版本**: v2.0 | 2026-03-03

---

## 0  方案总览

```
┌─────────────────────────────────────────────────────────────────────┐
│  Pre-trained Qwen2.5-VL-7B-Instruct (主选)                         │
│  ViT: 32层, 1280d | LLM: 28层, 3584d, 28Q/4KV (GQA)              │
│  3D-mRoPE: section=[16,24,24], temporal_patch_size=2               │
│              OR                                                      │
│  Qwen2.5-VL-3B-Instruct (效率对比)                                  │
│  ViT: 32层, 1280d | LLM: 36层, 2048d, 16Q/2KV (GQA)              │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
         ┌─────────────────▼──────────────────┐
         │  Stage 1: 流式推理结构适配 SFT       │
         │  Data: VideoChatOnline-IT (96K)     │
         │        + OVO-Bench train split      │
         │  Method: LoRA r=64, bf16, A800      │
         │  Frozen: ViT 全冻结                  │
         │  目的: 适应 Sink+Window 时序空洞     │
         └─────────────────┬──────────────────┘
                           │
         ┌─────────────────▼──────────────────┐
         │  Stage 2: UAV 垂类流式能力增强       │
         │  Data: StreamUAV-QA v2 (1955 QA)    │
         │  Method: LoRA r=64, bf16 (续训)     │
         │  Frozen: ViT 全冻结                  │
         │  目的: UAV 俯视角领域适配            │
         └─────────────────┬──────────────────┘
                           │
         ┌─────────────────▼──────────────────┐
         │  评测 & 消融实验                     │
         │  StreamUAV-QA eval                  │
         │  (perception / memory / prediction) │
         │  + OVO-Bench test 泛化验证           │
         └────────────────────────────────────┘
```

---

## 1  微调动机 — 为什么需要微调，何时开始？

> **这是论文 Motivation 章节的核心论点，需要实验数据支撑。**

### 1.1 根本问题：预训练分布与流式推理的结构性错配

标准 Qwen2.5-VL 在**离线完整视频**上预训练：所有 token 相互可见。本项目的流式推理系统
（`temporal_encoding/model/kv_cache_eviction.py`）引入 **Attention Sink + Sliding Window
KV Cache Eviction**，制造了预训练从未见过的上下文模式：

```
预训练时 (离线):
  [全部视觉token_0 ... 视觉token_T][问题token]
          ↑————— 模型可全局 attention —————↑

流式推理时 (本系统):
  [Sink: 首chunk视觉token][▒▒ 被淘汰的中间 token ▒▒][Window: 最近N token][问题token]
       ↑ 可见 ↑                  ↑ 时序空洞 ↑                ↑ 可见 ↑
```

**时序空洞 (Temporal Void)** 是微调的核心动机：

- 预训练模型回答 L2（memory）问题时，会尝试"recall"中间已淘汰的 token，产生幻觉
- 模型无法识别 sink 与 window 之间存在时序跳跃，导致时序推理逻辑混乱
- 模型未被训练成"在当前窗口内容中做局部时序推理"的行为模式

### 1.2 四个具体动机

#### 动机 A：流式上下文理解缺失（最主要）

| 能力 | 预训练离线状态 | 本系统流式推理状态 | 微调动机强度 |
|------|-------------|-----------------|------------|
| 全局视频理解 | ✅ 优秀 | ❌ 不可用（上下文窗口限制）| ★★★★★ |
| Sink+Window 下 L2 问答 | ⚠️ 从未接触此模式 | ❌ 幻觉风险极高 | **★★★★★** |
| 时序跳跃感知 | ⚠️ 无此概念 | ❌ 表现不稳定 | ★★★★ |
| 当前帧 (L1) 感知 | ✅ 较好 | ✅ 相对稳定 | ★★ |

#### 动机 B：连续 3D-mRoPE 位置编码超出预训练分布

`StreamQwenModel`（`stream_qwen_model.py`）在流式模式下时序位置 ID 单调递增：

```python
# Branch 3 (Decode): _last_cache_position 不断增大
self._last_cache_position = cache_position[-1].item()
```

预训练时 temporal position ID 被约束在视频片段内（通常 <1000 时间步）。持续流式推理中，
该值可能增至数万，超出预训练分布，mRoPE 插值质量下降。微调让模型在流式数据上学习大
position ID 下的正确推理模式。

#### 动机 C：UAV 俯视角领域鸿沟（Stage 2 核心动机）

VisDrone 无人机俯视角场景与通用视频预训练语料差异巨大：

- **目标尺寸**: 俯视角行人约 8-20px，车辆约 20-50px，极小目标
- **视角**: 近乎垂直俯视，与第一视角/平视视频完全不同
- **场景语义**: 交通流分析、密度变化等 UAV 特有推理任务

无领域适配的直接推理：模型对小目标密度变化不敏感，L2/L3 问题准确率极低。

#### 动机 D：MCQ 格式与单字母输出遵循

StreamUAV-QA 使用 OVBench 对齐的 `(A)/(B)/(C)/(D)` MCQ 格式，要求模型输出单字母。
预训练模型倾向于生成完整句子；`[At X.Xs]` 串流时钟前缀的语义在无专项训练时不稳定。

### 1.3 决策触发实验：量化流式推理性能鸿沟

**在开始微调之前，先运行此对比实验，用数据证明微调的必要性（同时作为论文 Motivation 图）**：

```
实验 M1 — 离线 vs 流式基线对比（无微调）
  模型: Qwen2.5-VL-7B-Instruct (zero-shot)

  M1-A: 离线模式
    输入: 完整视频帧序列（不限 context 长度，无 KV 淘汰）
    评测: StreamUAV-QA test (195条)
    记录: Perception / Memory / Prediction / Overall Accuracy

  M1-B: 流式模式
    输入: 相同视频，使用 StreamQwenModel + Level-1 KV Eviction
    评测: 相同 195 条测试题
    记录: 同上

预期: M1-A > M1-B，Memory Accuracy 差距 ≥ 15 个百分点
      此差距即是微调的量化动机。

实验 M2 — 窗口大小 vs 准确率曲线
  在 max_cache_tokens = {20K, 50K, 100K, 150K} 下运行 M1-B
  → 绘制 "窗口大小 vs Memory Accuracy" 曲线，说明窗口约束的影响规律
```

**何时开始微调**：当 M1-A 与 M1-B 的 Memory Accuracy 差距 ≥ 10 个百分点时，微调具有
充分的动机。根据系统设计，预期差距远超此阈值。

---

## 2  微调方法选型

### 2.1 核心问题：7B LoRA 还是 3B LoRA？

用户拥有 **A800 (80GB)** 用于微调，**RTX 4090 (24GB)** 用于推理。

> **结论：推荐 Qwen2.5-VL-7B-Instruct 进行 LoRA (bf16) 微调。**

**纠正"QLoRA 将 7B 量化到 3B"的误解**：

QLoRA 不减少参数量。它将 7B 参数的存储从 14GB (bf16) 压缩到 ~3.7GB (NF4)，模型**仍有
7B 参数**，推理能力远高于等存储的 3B 模型。QLoRA 唯一优势是节省显存，适用于 A100
(40GB) 等受限场景。A800 拥有 80GB，7B bf16 LoRA 仅需约 25GB，**在 A800 上 QLoRA 引入
不必要的精度损失，没有任何显存收益**。

### 2.2 7B vs 3B 详细对比

| 维度 | 3B LoRA (bf16) on A800 | **7B LoRA (bf16) on A800** |
|------|------------------------|---------------------------|
| LLM 层数 / Hidden | 36L / 2048d | **28L / 3584d** |
| 模型权重 VRAM | 5.8 GB | **14.7 GB** |
| LoRA+优化器 VRAM | ~0.7 GB | **~1.5 GB** |
| 激活 (GC) | ~3-5 GB | **~5-10 GB** |
| **训练总 VRAM** | **~11 GB** | **~25 GB** ✅ A800 富余 |
| L2 时序推理能力 | ★★★☆ | **★★★★★** |
| L3 预测推理能力 | ★★☆ | **★★★★** |
| 4090 推理 (bf16) | ~11 GB ✅ | **~20 GB** ✅ (24GB 内) |
| 4090 推理 (int8) | ~7 GB ✅ | **~11 GB** ✅✅ |
| 训练速度 (A800) | 快 (~4h) | 中 (~10h) |
| 论文说服力 | 较低 | **高** |

**7B 在 RTX 4090 的推理可行性**：
- bf16: ~14.7GB 权重 + ~5GB KV cache (Level-1 淘汰后约 100K tokens) ≈ **20GB** ✅
- 如 VRAM 紧张：AWQ/GPTQ int8 量化 → ~11GB ✅

**论文双模型策略**：
- 主实验：7B LoRA（追求最优性能，论文主要数值）
- 效率对比：3B LoRA（可在 4090 上运行，形成性能-效率 tradeoff 分析）

### 2.3 方法选型：LoRA (bf16)，不用 QLoRA / RLHF

| 方案 | A800 显存 | 适用性 | 说明 |
|------|----------|--------|------|
| Full SFT (7B, bf16) | ~65-70 GB | ⚠️ A800 勉强 | 1955 样本下极易过拟合 |
| **LoRA (7B, bf16, r=64)** | **~25 GB** | **✅ 首选** | 低秩正则，两阶段可续训 |
| QLoRA (7B, NF4, r=64) | ~12 GB | ⚠️ A800 无需 | 引入精度损失，A800 上无优势 |
| RLHF / DPO | >50 GB | ❌ | MCQ 客观任务无需偏好对齐 |

**不用 QLoRA**：A800 80GB 完全能容纳 7B bf16 LoRA (~25GB)，引入 NF4 量化仅损害精度。

**不用 RLHF/DPO**：StreamUAV-QA 有唯一 ground-truth 答案，无偏好优化空间；且需要额外
的 Reward Model / 参考模型，显存需求翻倍。可作为 Future Work。

### 2.4 最终决策

| 阶段 | 硬件 | 模型 | 方法 | 目的 |
|------|------|------|------|------|
| Stage 1 | A800 | **7B**-Instruct | LoRA r=64, bf16 | 流式推理结构适配 |
| Stage 2 | A800 | 7B (续 S1) | LoRA r=64, bf16 | UAV 垂类增强 |
| 效率对比 | 4090 | 3B-Instruct | LoRA r=64, bf16 | 消融/效率分析 |

---

## 2.5  时间戳前缀设计辨析

### 问题：`[At X.Xs]` 对 3B/7B 小模型要求是否过高？

**先看数据集中的实际问题措辞**：

| 样例问题 | 时序关键词 |
|---------|----------|
| `[At 5.0s] Is there any bicycle visible in the **current** drone footage?` | current（相对） |
| `[At 10.0s] What type of object has **recently** disappeared?` | recently（相对） |
| `[At 12.0s] Based on your **observation of the video stream**, what is the overall **trend**?` | trend（全局统计） |

**关键结论：问题措辞全部使用相对时序语言，无需模型执行绝对时间定位。**

`[At X.Xs]` 的真实语义是**串流时钟标记**（streaming clock marker）：
> "这个问题是在视频流播放到 X.X 秒时提出的。"

这与离线模式下的 "请精确回忆 X.X 秒时刻的画面" 是两种完全不同的语义：

```
离线模式：模型输入完整视频 → [At 5.0s] 意味着 "跳到第 5 秒"（绝对定位）
流式模式：模型当前看到窗口内容 → [At 5.0s] 意味着 "你现在处于第 5 秒"（当前时钟）
```

**时间戳的隐性价值**：让模型感知视频的"进程"——

- `[At 2.0s]`：视频刚开始，历史极短，无需长期记忆
- `[At 120.0s]`：视频已播放 2 分钟，应有丰富窗口历史（但受 KV 淘汰限制）

Stage 1 训练的核心任务之一，就是让模型学会在**串流语境**下正确解读此前缀。

**结论**：保持 `[At X.Xs]` 格式，不修改数据集。理由：

1. OVBench 对齐（方便与 OVO-Bench 基线直接对比）
2. Stage 1 训练会教会模型串流时钟的正确语义
3. 数据问题本身已是相对语言，无绝对记忆依赖

如评审要求简化，可在论文中说明该前缀为"streaming timestamp indicator"而非"temporal
localization query"，并在消融实验中比较有/无此前缀对准确率的影响。

---

## 3  Stage 1: 流式推理结构适配

### 3.1 目标

使模型学会：
1. **在 Sink+Window KV Cache 的部分上下文下进行时序推理**（最核心目标）
2. **将 `[At X.Xs]` 理解为串流时钟，而非离线绝对定位**
3. **以单字母 `(A)/(B)/(C)/(D)` 格式输出 MCQ 答案**
4. **交错多轮对话（边看视频边回答）的上下文维护**

### 3.2 训练数据

| 数据集 | 样本数 | 采样权重 | 作用 |
|--------|--------|---------|------|
| VideoChatOnline-IT | ~96K | 1.0 | 通用在线视频推理，含交错时间戳对话 |
| OVO-Bench (train 80%) | ~1.6K | 3.0 (过采样) | MCQ 格式对齐，与 StreamUAV-QA 最接近 |
| **有效训练样本** | **~100.8K** | | |

### 3.3 关键设计：训练时模拟 KV Cache 时序空洞

**这是 Stage 1 区别于普通 SFT 的核心创新**。在训练时通过 DataCollator 随机在
token 序列中制造"中间视觉 token 缺失"模式，让模型学会在不完整上下文下推理：

```python
class StreamingAwareDataCollator:
    """
    模拟 Attention Sink + Sliding Window 的训练 DataCollator。
    
    以 drop_prob 的概率对序列中的视觉 token 进行截断：
    - 保留前 sink_ratio 的视觉 token（模拟 sink）
    - 保留后 window_ratio 的视觉 token（模拟 window）
    - 中间视觉 token 替换为 PAD（模拟淘汰）
    
    另 (1-drop_prob) 的概率保持完整序列（防止过拟合到残缺上下文）。
    """
    VISUAL_TOKEN_ID = 151656  # Qwen2.5-VL <|video_pad|>
    
    def __init__(self, processor, sink_ratio=0.15, window_ratio=0.35, drop_prob=0.5):
        self.processor = processor
        self.sink_ratio = sink_ratio    # 保留前 15% 视觉 token
        self.window_ratio = window_ratio  # 保留后 35% 视觉 token
        self.drop_prob = drop_prob      # 50% 概率触发截断

    def simulate_kv_eviction(self, input_ids: torch.Tensor) -> torch.Tensor:
        visual_pos = (input_ids == self.VISUAL_TOKEN_ID).nonzero(as_tuple=True)[0]
        n_visual = len(visual_pos)
        if n_visual < 10 or torch.rand(1).item() > self.drop_prob:
            return input_ids  # 不截断
        
        sink_end = int(n_visual * self.sink_ratio)
        window_start = int(n_visual * (1 - self.window_ratio))
        mid_pos = visual_pos[sink_end:window_start]
        
        input_ids = input_ids.clone()
        input_ids[mid_pos] = self.processor.tokenizer.pad_token_id
        return input_ids
```

### 3.4 训练超参 (A800, 7B LoRA)

```yaml
lora:
  r: 64
  lora_alpha: 128
  lora_dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj]

freeze:
  vision_encoder: true    # ViT 全冻结（视觉特征提取无需适配）
  embed_tokens: true

training:
  num_epochs: 1
  per_device_batch_size: 1      # A800 80GB, 7B bf16
  gradient_accumulation: 16
  effective_batch_size: 16
  learning_rate: 2e-4
  lr_scheduler: cosine
  warmup_ratio: 0.03
  gradient_checkpointing: true
  bf16: true

video:
  max_frames: 32
  resolution: 336
  fps: 2.0
```

### 3.5 A800 显存预算 (7B)

| 组件 | 显存 (GB) |
|------|----------|
| 7B 模型权重 (bf16, frozen) | 14.7 |
| LoRA 权重 + 梯度 + AdamW | ~1.5 |
| 激活值 (gradient checkpointing) | 5-8 |
| CUDA context + 框架 | 1.5 |
| **总计** | **~23-26 GB** ✅ A800 富余 54+ GB |

### 3.6 训练时间估算

```
有效样本: ~100.8K | Effective batch size: 16
Steps: 100,800 / 16 = 6,300
每 step (A800, 7B, seq~1K): ~4s
总时间: 6,300 × 4s ≈ 7h
```

---

## 4  Stage 2: UAV 垂类流式能力增强

### 4.1 目标

在 Stage 1 适配的流式推理基础上注入 UAV 领域知识：
- 无人机俯视角小目标分类与计数
- 空中场景密度变化感知
- UAV 特有时序事件推理（流量趋势、目标消失/出现、拥堵预测）

### 4.2 数据策略

| 数据集 | 样本数 | 划分 |
|--------|--------|------|
| StreamUAV-QA v2 (训练) | 1,564 (80%) | 主训练 |
| StreamUAV-QA v2 (验证) | 196 (10%) | Early stopping |
| StreamUAV-QA v2 (测试) | 195 (10%) | 最终报告（训练中不可见）|
| VideoChatOnline-IT (防遗忘) | ~2,000 (随机) | 防通用能力退化 |
| **有效训练样本** | **~3,564** | |

### 4.3 数据增强（防止 1955 样本过拟合）

1. **选项随机打乱**：每 epoch 随机重排 ABCD 顺序并更新 answer，增加格式多样性
2. **同义改写**：简单改写 question 措辞（保持语义），扩充约 1.5×
3. **帧级增强**：轻微色彩抖动（brightness/contrast ±0.1），不做翻转（UAV 方向敏感）

### 4.4 训练超参

```yaml
lora:
  r: 64
  lora_alpha: 128
  lora_dropout: 0.10          # 数据量小，增强正则化
  # 从 Stage 1 checkpoint 初始化，非随机

training:
  num_epochs: 5               # 小数据集，配合 early stopping
  per_device_batch_size: 4    # 序列短，可提高 batch
  gradient_accumulation: 4
  effective_batch_size: 16
  learning_rate: 5e-5         # 精调阶段，降为 Stage 1 的 1/4
  weight_decay: 0.05
  warmup_ratio: 0.05
  gradient_checkpointing: true
  early_stopping_patience: 3  # 连续 3 epoch 验证集无提升则停止

evaluation:
  eval_strategy: epoch
  metric_for_best_model: eval/memory_accuracy   # L2 记忆能力为核心指标
```

### 4.5 训练时间估算

```
有效训练样本: ~3,564 | Effective batch size: 16
Steps/epoch: 3,564 / 16 ≈ 223
Epochs: 5 (max) | 每 step (A800, 7B): ~3s
总时间: 223 × 5 × 3s ≈ 0.9h（含 early stopping 约 1.5h）
```

---

## 5  训练框架：自定义 PEFT（非 LLaMA-Factory）

### 5.1 为什么不用 LLaMA-Factory

1. **自定义模型类**：本项目使用 `StreamQwenModel`（继承并改造了 `Qwen2_5_VLForConditionalGeneration`），
   LLaMA-Factory 无法直接集成自定义模型类。
2. **训练时模拟 KV 淘汰**：Stage 1 的 `StreamingAwareDataCollator` 是自定义 Data Pipeline
   逻辑，框架级工具难以适配。
3. **3D-mRoPE 调试需求**：需要逐 batch 验证 `position_ids` 的三维格式，自定义 Trainer
   可方便插入 sanity check 逻辑。

**设计理念参考** [minimind-v](https://github.com/jingyaogong/minimind-v)：

minimind-v 以极简的 PyTorch 训练循环实现了完整的 VLM 微调，证明不依赖重型框架同样
可以高效训练。我们采用相同的设计哲学，但基于 HuggingFace `transformers.Trainer`（保证
与 Qwen2.5-VL 的兼容性），避免从零实现分布式训练。

### 5.2 技术栈

| 组件 | 选择 | 说明 |
|------|------|------|
| 基础模型 | `StreamQwenModel`（本项目） | 继承自 `Qwen2_5_VLForConditionalGeneration` |
| LoRA | `peft.LoraConfig` + `get_peft_model` | HuggingFace PEFT |
| 训练器 | `transformers.Trainer` | 支持 GC、bf16、callback |
| 注意力加速 | Flash Attention 2 | A800 原生支持 |
| DataCollator | 自定义 `StreamingAwareDataCollator` | 训练时模拟 KV 淘汰 |
| 混合精度 | bf16（非 NF4） | A800 无需量化 |

### 5.3 目录结构

```
temporal_encoding/
  train/
    train_stage1.py          ← Stage 1 主脚本
    train_stage2.py          ← Stage 2 主脚本
    data_collator.py         ← StreamingAwareDataCollator
    dataset.py               ← VideoSFTDataset（支持多数据集混合）
    lora_utils.py            ← LoRA 构建与冻结逻辑
    eval_callback.py         ← 自定义评测（按 answer_type 分项统计）
    merge_lora.py            ← LoRA 权重合并到 base model
    configs/
      stage1_7b.yaml
      stage2_7b.yaml
      stage1_3b.yaml         ← 效率对比用
```

### 5.4 核心代码骨架

**`lora_utils.py`**:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch
from peft import LoraConfig, TaskType, get_peft_model
from temporal_encoding.model.stream_qwen_model import StreamQwenModel

def build_lora_model(
    model_path: str,
    r: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.05,
    lora_checkpoint: str = None,    # Stage 2 传入 Stage 1 adapter 路径
) -> StreamQwenModel:
    """
    加载 StreamQwenModel 并应用 LoRA 适配器。
    
    冻结策略:
      - ViT (visual.*): 全冻结，视觉特征提取无需适配
      - LLM q/k/v/o: LoRA 适配，其余 frozen
      - embed_tokens / lm_head: 冻结（tie_word_embeddings=True）
    """
    model = StreamQwenModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="cuda",
    )
    
    # 冻结 Vision Encoder（ViT + PatchMerger）
    for name, param in model.named_parameters():
        if "visual" in name:
            param.requires_grad = False
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        # embed_tokens / lm_head 不在 target_modules 中，自动跳过
    )
    
    if lora_checkpoint is not None:
        # Stage 2: 从 Stage 1 adapter 初始化
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_checkpoint, is_trainable=True)
        # 更新 dropout（Stage 2 需要更高正则化）
        for module in model.modules():
            if hasattr(module, "lora_dropout") and hasattr(module.lora_dropout, "p"):
                module.lora_dropout.p = lora_dropout
    else:
        # Stage 1: 随机初始化 LoRA
        model = get_peft_model(model, lora_config)
    
    model.print_trainable_parameters()
    return model
```

**`train_stage1.py`** (核心骨架):

```python
import torch
from transformers import AutoProcessor, Trainer, TrainingArguments
from lora_utils import build_lora_model
from dataset import VideoSFTDataset
from data_collator import StreamingAwareDataCollator
from eval_callback import StreamingEvalCallback

def main():
    MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"
    
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model = build_lora_model(MODEL_PATH, r=64, lora_alpha=128, lora_dropout=0.05)
    
    train_dataset = VideoSFTDataset(
        data_files=["videochat_online_it.json", "ovobench_train.json"],
        processor=processor,
        max_seq_length=2048,
        oversample_weights={"ovobench_train.json": 3.0},
    )
    
    collator = StreamingAwareDataCollator(
        processor=processor,
        sink_ratio=0.15,
        window_ratio=0.35,
        drop_prob=0.5,
    )
    
    args = TrainingArguments(
        output_dir="./checkpoints/stage1_7b",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        gradient_checkpointing=True,
        dataloader_num_workers=4,
        logging_steps=10,
        save_steps=500,
        save_total_limit=3,
        report_to="tensorboard",
    )
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=collator,
    )
    trainer.train()
    model.save_pretrained("./checkpoints/stage1_7b/final")


if __name__ == "__main__":
    main()
```

**`train_stage2.py`** (Stage 2 关键差异):

```python
# 关键：加载 Stage 1 LoRA 作为起点
model = build_lora_model(
    MODEL_PATH,
    r=64,
    lora_alpha=128,
    lora_dropout=0.10,                           # 提高 dropout
    lora_checkpoint="./checkpoints/stage1_7b/final",  # Stage 1 adapter
)

train_dataset = VideoSFTDataset(
    data_files=["stream_uav_sft.json"],
    replay_files=["videochat_online_it_sample2k.json"],  # 防遗忘混合
    processor=processor,
    max_seq_length=1536,
)

# Stage 2 不使用 StreamingAwareDataCollator（UAV 视频较短，直接用完整序列）
```

### 5.5 3D-mRoPE Sanity Check（训练前必须执行）

```python
# 在训练开始前验证 position_ids 格式正确性
def sanity_check_rope(model, processor, sample_batch):
    """确保 StreamQwenModel 生成的 position_ids 形状为 (3, seq_len)。"""
    with torch.no_grad():
        outputs = model(**sample_batch, return_dict=True)
    pos_ids = sample_batch.get("position_ids")
    assert pos_ids is not None, "position_ids 未传入！"
    assert pos_ids.dim() == 3 and pos_ids.shape[0] == 3, \
        f"position_ids 形状应为 (3, B, S)，实际为 {pos_ids.shape}"
    print("[Sanity Check] 3D-mRoPE position_ids 格式正确 ✓")
```

### 5.6 LoRA 可训练参数计算 (7B)

```
7B 配置: hidden=3584, q_heads=28, kv_heads=4, head_dim=128

每层 LoRA 参数 (r=64):
  q_proj: (3584×64) + (64×3584)    = 229,376 + 229,376 = 458,752
  k_proj: (3584×64) + (64×512)     = 229,376 +  32,768 = 262,144  (kv_heads=4)
  v_proj: 同 k_proj                                    = 262,144
  o_proj: (3584×64) + (64×3584)    = 229,376 + 229,376 = 458,752

  每层合计: 1,441,792 ≈ 1.44M 参数
  28 层合计: 40.3M 参数 (bf16 ~0.08 GB)

AdamW 状态 (fp32 m + v + master copy):
  40.3M × 12 bytes ≈ 0.46 GB

LoRA 相关 VRAM 合计: ~0.54 GB
可训练参数占比: 40.3M / 7,600M ≈ 0.53%
```

---

## 6  评测方案

### 6.1 核心评测集

| 评测集 | 样本数 | 用途 |
|--------|--------|------|
| StreamUAV-QA test | 195 条 | 主要数值报告（训练中严格不可见）|
| StreamUAV-QA val | 196 条 | 训练过程监控 / early stopping |
| OVO-Bench test | ~400 条 | Stage 1 泛化能力验证 |

### 6.2 评测指标

| 指标 | 论文对应能力 |
|------|------------|
| Overall Accuracy | 整体 |
| Perception Accuracy | L1 实时感知 |
| **Memory Accuracy** | **L2 时序记忆（核心，与动机实验对应）** |
| Prediction Accuracy | L3 推理预测 |
| Per sub_answer_type | 30 个细粒度分项（能力诊断）|

### 6.3 实验对比矩阵

| 模型配置 | Perception | Memory | Prediction | Overall |
|---------|-----------|--------|-----------|---------|
| 7B base (离线, 无淘汰) | - | - | - | - |
| 7B base (流式, Level-1 淘汰) | - | - | - | - |
| 7B + Stage 1 only | - | - | - | - |
| **7B + Stage 1 + Stage 2** | - | - | - | - |
| 3B + Stage 1 + Stage 2 | - | - | - | - |
| 7B + Stage 2 only（消融）| - | - | - | - |

### 6.4 消融实验

| 实验 | 变量 | 目的 |
|------|------|------|
| **A1** | 离线 vs 流式推理（无微调）| **动机实验**，量化微调必要性 |
| A2 | Stage 1 有/无 | 验证流式结构适配的重要性 |
| A3 | Stage1→2 vs Stage2 only | 两阶段渐进的必要性 |
| A4 | 7B vs 3B（两阶段）| 性能-效率权衡 |
| A5 | 有/无防遗忘混合（Stage 2）| 遗忘分析 |
| A6 | LoRA r=32 / 64 / 128 | 最优 rank |
| **A7** | **有/无 StreamingAwareDataCollator** | **训练时 KV 模拟的效果** |

---

## 7  风险与应对

| 风险 | 概率 | 应对策略 |
|------|------|---------|
| Stage 2 过拟合（1955 样本）| 高 | early stopping + dropout=0.10 + 防遗忘混合 + 数据增强 |
| Stage 2 通用能力退化 | 中 | 混入 ~2K 条 Stage 1 数据；LoRA 低秩正则化 |
| 7B 在 4090 推理 OOM | 低 | 切换 int8 量化（~11GB）；降低 max_cache_tokens |
| 3D-mRoPE position_ids 格式错误 | 中 | 训练前运行 sanity check，验证 shape=(3,B,S) |
| A800 上视觉 token 太多 OOM | 低 | 降低训练分辨率至 224 / max_frames 至 16 |
| VideoChatOnline-IT 视频文件缺失 | 高 | 优先处理有视频文件的子集；或仅用文本对话部分 |

---

## 8  时间线

```
Week 1 (03/04 - 03/10):
  ├── [D1]     动机实验 M1 (7B base, 离线 vs 流式对比，195 条测试集)
  ├── [D2-D3]  编写训练代码骨架 (lora_utils + dataset + collator)
  ├── [D4]     下载 VideoChatOnline-IT，转换统一格式
  └── [D5-D7]  Stage 1 训练 (A800, ~7h)

Week 2 (03/11 - 03/17):
  ├── [D1]     Stage 1 评测: OVO-Bench test + StreamUAV-QA test
  ├── [D2]     Stage 2 数据准备 (划分 train/val/test + 数据增强)
  ├── [D3]     Stage 2 训练 (A800, ~1.5h) + 超参搜索
  ├── [D4-D5]  全量消融实验 A1-A7
  └── [D6-D7]  结果汇总、制表、绘图

Week 3 (03/18 - 03/24):
  ├── [D1-D3]  论文实验章节撰写
  └── [D4-D7]  Introduction + Related Work + Abstract
```

---

## 9  附录

### 9.1 LoRA 数学原理

对于预训练权重 $W_0 \in \mathbb{R}^{d \times k}$，LoRA 引入低秩分解：

$$W = W_0 + BA, \quad B \in \mathbb{R}^{d \times r},\ A \in \mathbb{R}^{r \times k},\ r \ll \min(d,k)$$

训练时冻结 $W_0$，仅更新 $A$ 和 $B$。对 7B 的 q_proj ($d=k=3584$, $r=64$)：

$$\text{压缩比} = \frac{64(3584 + 3584)}{3584^2} \approx 3.6\%$$

### 9.2 A800 显存分配总览 (7B LoRA bf16)

$$\text{VRAM} \approx \underbrace{2P_{\text{base}}}_{\text{模型权重}} + \underbrace{(2+8) P_L}_{\text{LoRA 梯度+AdamW}} + \underbrace{V_{\text{act}}}_{\text{激活(GC)}} + C$$

| 项目 | 值 |
|------|-----|
| $2P_{\text{base}}$（7B bf16）| 14.7 GB |
| $(2+8)P_L$（40M LoRA）| 0.5 GB |
| $V_{\text{act}}$（GC 后）| 5-8 GB |
| $C$（CUDA context）| 1.5 GB |
| **合计** | **~22-25 GB** ✅ A800 余量 55+ GB |

### 9.3 参考文献

1. Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models," ICLR 2022
2. Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs," NeurIPS 2023
3. Shi et al., "VideoChat-Online: Online Video Understanding via Pyramid Memory Bank," arXiv 2025
4. Han et al., "StreamingLLM: Efficient Streaming Language Models with Attention Sinks," ICLR 2024
5. Wang et al., "Qwen2.5-VL Technical Report," arXiv 2025
6. Jin & Gong, "minimind-v: A Minimal Vision-Language Model," GitHub 2024

---

> **版本 v2.0**: 更新微调动机（4 个具体动机 + 决策触发实验），将主选模型改为 7B LoRA
> on A800，阐明 QLoRA 在 A800 上的无效性，替换 LLaMA-Factory 为自定义 PEFT 方案（参考
> minimind-v 理念），澄清 `[At X.Xs]` 为串流时钟标记而非绝对记忆锚点。
