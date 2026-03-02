# 🛩️ StreamUAV: UAV 流式视频事件感知 — 完整实施指南

> **版本**: v2.0 · 2026-03-02  
> **目标**: 在 H800 服务器上按本文档顺序执行，完成从数据集构建到模型验证的全流程  
> **硬件**: RTX 4090 (24GB) × 1（开发调试） + H800 (80GB) × 1（数据构建+微调+评测）  
> **前置依赖**: `diploma_project/temporal_encoding/model/` 下的流式推理引擎已可用  
> **研究创新点**: 0 篇论文将流式 VLM 与 UAV 结合 — 本工作为首个

---

## 目录

- [0. 环境准备与代码同步](#0-环境准备与代码同步)
- [1. 数据集下载与预处理](#1-数据集下载与预处理)
- [2. 流式 QA 数据集构建（核心）](#2-流式-qa-数据集构建核心)
  - [2.1 总体方法论](#21-总体方法论)
  - [2.2 smolagents 驱动的自动化标注 Pipeline](#22-smolagents-驱动的自动化标注-pipeline)
  - [2.3 QA 类型设计与模板](#23-qa-类型设计与模板)
  - [2.4 流式对话格式转换](#24-流式对话格式转换)
  - [2.5 质量控制与过滤](#25-质量控制与过滤)
  - [2.6 主构建脚本](#26-主构建脚本)
- [3. 流式推理引擎适配与评测](#3-流式推理引擎适配与评测)
- [4. 评测与验证策略](#4-评测与验证策略)
- [5. 可选: LoRA 微调](#5-可选-lora-微调)
- [6. 论文写作框架](#6-论文写作框架)
- [7. 时间线与投稿计划](#7-时间线与投稿计划)
- [附录 A: VisDrone 标注格式规范](#附录-a-visdrone-标注格式规范)
- [附录 B: 关键技术参数（已验证）](#附录-b-关键技术参数已验证)
- [附录 C: 参考文献](#附录-c-参考文献)

---

## 0. 环境准备与代码同步

```bash
# ── H800 服务器上执行 ──

# 0.1 同步项目代码
cd /root/autodl-tmp/
git clone <your_repo_url> diploma_project
cd diploma_project

# 0.2 创建/激活 conda 环境
conda activate videollm
pip install smolagents[toolkit] datasets transformers accelerate
pip install opencv-python-headless decord Pillow
pip install peft bitsandbytes   # 为后续可能的 LoRA 微调准备

# 0.3 确认流式推理引擎可用
python -c "
import sys; sys.path.insert(0, 'temporal_encoding')
from model.stream_qwen_model import StreamQwenModel
print('✅ StreamQwenModel 加载成功')
"

# 0.4 模型路径约定
# Qwen2.5-VL-3B:  /root/autodl-tmp/Qwen/Qwen2___5-VL-3B-Instruct
# Qwen2.5-VL-7B:  /root/autodl-tmp/Qwen/Qwen2___5-VL-7B-Instruct
```

---

## 1. 数据集下载与预处理

### 1.1 VisDrone-VID 下载（主数据集）

> **VisDrone** 由天津大学 AISKYEYE 团队发布，14 个中国城市实拍，288 个视频片段，261,908 帧，260 万+ bbox。

```bash
mkdir -p /root/autodl-tmp/datasets/visdrone
cd /root/autodl-tmp/datasets/visdrone

# ─── trainset (7.53 GB) ───
# GoogleDrive: https://drive.google.com/file/d/1NSNapZQHar22OYzQYuXCugA3QlMndzvw
# 百度网盘:    https://pan.baidu.com/s/1kC3NTK6MPVv3D1CY9gXaCQ
unzip VisDrone2019-VID-train.zip -d VID-train/

# ─── valset (1.49 GB) ───
# GoogleDrive: https://drive.google.com/file/d/1xuG7Z3IhVfGGKMe3Yj6RnrFHqo_d2a1B
unzip VisDrone2019-VID-val.zip -d VID-val/

# ─── testset-dev (2.14 GB, GT 可用) ───
# GoogleDrive: https://drive.google.com/open?id=1-BEq--FcjshTF1UwUabby_LHhYj41os5
unzip VisDrone2019-VID-test-dev.zip -d VID-test-dev/
```

### 1.2 VisDrone 目录结构

```
VID-train/
├── sequences/              # 视频序列（每个子目录 = 一个视频）
│   ├── uav0000013_00000_v/
│   │   ├── 0000001.jpg
│   │   ├── 0000002.jpg
│   │   └── ...            # 原始 ~30fps
│   ├── uav0000013_01392_v/
│   └── ...
└── annotations/            # 逐帧 GT 标注
    ├── uav0000013_00000_v/
    │   ├── 0000001.txt     # 每行一个目标的 bbox+类别+遮挡
    │   ├── 0000002.txt
    │   └── ...
    └── ...
```

### 1.3 VisDrone 标注格式（详见附录 A）

每行: `bbox_left,bbox_top,bbox_width,bbox_height,score,category,truncation,occlusion`

```
# 示例 — uav0000013_00000_v/0000042.txt
684,8,56,26,0,4,0,0     # car, 无截断, 无遮挡
406,119,28,22,0,1,0,0   # pedestrian, 无截断, 无遮挡
200,308,18,24,0,1,0,1   # pedestrian, 无截断, 部分遮挡
```

**10 类有效目标**:

| ID | 类别 | 英文 | 典型场景 |
|----|------|------|----------|
| 1 | 行人 | pedestrian | 路口、广场 |
| 2 | 人 | people | 人群聚集 |
| 3 | 自行车 | bicycle | 非机动车道 |
| 4 | 汽车 | car | 道路、停车场 |
| 5 | 面包车 | van | 道路 |
| 6 | 卡车 | truck | 货运区域 |
| 7 | 三轮车 | tricycle | 城镇道路 |
| 8 | 遮阳三轮车 | awning-tricycle | 城镇道路 |
| 9 | 公共汽车 | bus | 公交站、主干道 |
| 10 | 摩托车 | motor | 混合交通 |

（ID=0 忽略区域, ID=11 其他 → 不参与评测）

### 1.4 数据加载器

```python
# ═══ 文件: temporal_encoding/dataset/visdrone_loader.py ═══

"""
VisDrone-VID 数据加载与预处理。
将逐帧图像序列 + 逐帧标注解析为结构化 Python 对象。
"""

import os
import glob
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from PIL import Image


VISDRONE_CATEGORIES = {
    0: "ignored",
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning-tricycle",
    9: "bus",
    10: "motor",
    11: "others",
}

VALID_CATEGORIES = {k: v for k, v in VISDRONE_CATEGORIES.items() if k not in (0, 11)}


@dataclass
class BBoxAnnotation:
    """单个目标标注"""
    bbox_left: int
    bbox_top: int
    bbox_width: int
    bbox_height: int
    score: int          # 1=参与评估, 0=忽略
    category_id: int
    truncation: int     # 0=无, 1=部分截断
    occlusion: int      # 0=无, 1=部分遮挡, 2=重度遮挡

    @property
    def category_name(self) -> str:
        return VISDRONE_CATEGORIES.get(self.category_id, "unknown")

    @property
    def center(self) -> Tuple[float, float]:
        return (self.bbox_left + self.bbox_width / 2,
                self.bbox_top + self.bbox_height / 2)

    @property
    def area(self) -> int:
        return self.bbox_width * self.bbox_height


@dataclass
class FrameAnnotation:
    """单帧标注"""
    frame_id: int
    objects: List[BBoxAnnotation] = field(default_factory=list)

    @property
    def valid_objects(self) -> List[BBoxAnnotation]:
        return [o for o in self.objects
                if o.category_id in VALID_CATEGORIES and o.score == 1]

    def count_by_category(self) -> Dict[str, int]:
        counts = {}
        for o in self.valid_objects:
            name = o.category_name
            counts[name] = counts.get(name, 0) + 1
        return counts


@dataclass
class VideoSequence:
    """一个 VisDrone 视频序列"""
    seq_name: str
    frame_dir: str
    anno_dir: str
    frames: List[FrameAnnotation] = field(default_factory=list)
    _frame_paths: List[str] = field(default_factory=list)

    @property
    def num_frames(self) -> int:
        return len(self._frame_paths)

    def load(self):
        """加载所有帧路径和标注"""
        self._frame_paths = sorted(glob.glob(os.path.join(self.frame_dir, "*.jpg")))
        if not self._frame_paths:
            self._frame_paths = sorted(glob.glob(os.path.join(self.frame_dir, "*.png")))

        self.frames = []
        for i, fp in enumerate(self._frame_paths):
            frame_name = Path(fp).stem
            anno_path = os.path.join(self.anno_dir, f"{frame_name}.txt")
            frame_anno = self._parse_annotation(i + 1, anno_path)
            self.frames.append(frame_anno)

    def get_frame_image(self, frame_idx: int) -> Image.Image:
        return Image.open(self._frame_paths[frame_idx])

    def get_frame_images(self, start: int, end: int) -> List[Image.Image]:
        return [Image.open(self._frame_paths[i])
                for i in range(start, min(end, len(self._frame_paths)))]

    @staticmethod
    def _parse_annotation(frame_id: int, anno_path: str) -> FrameAnnotation:
        frame = FrameAnnotation(frame_id=frame_id)
        if not os.path.exists(anno_path):
            return frame
        with open(anno_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(',')
                if len(parts) < 8:
                    continue
                bbox = BBoxAnnotation(
                    bbox_left=int(parts[0]),
                    bbox_top=int(parts[1]),
                    bbox_width=int(parts[2]),
                    bbox_height=int(parts[3]),
                    score=int(parts[4]),
                    category_id=int(parts[5]),
                    truncation=int(parts[6]),
                    occlusion=int(parts[7]),
                )
                frame.objects.append(bbox)
        return frame


def load_visdrone_vid(data_root: str) -> List[VideoSequence]:
    """
    加载 VisDrone-VID 数据集。

    Args:
        data_root: 数据集根目录 (如 /root/autodl-tmp/datasets/visdrone/VID-train)

    Returns:
        VideoSequence 列表
    """
    seq_dir = os.path.join(data_root, "sequences")
    anno_dir = os.path.join(data_root, "annotations")

    sequences = []
    for seq_name in sorted(os.listdir(seq_dir)):
        frame_path = os.path.join(seq_dir, seq_name)
        anno_path = os.path.join(anno_dir, seq_name)
        if os.path.isdir(frame_path):
            seq = VideoSequence(
                seq_name=seq_name,
                frame_dir=frame_path,
                anno_dir=anno_path,
            )
            seq.load()
            sequences.append(seq)
            print(f"  Loaded {seq_name}: {seq.num_frames} frames")

    print(f"✅ Loaded {len(sequences)} video sequences from {data_root}")
    return sequences
```

---

## 2. 流式 QA 数据集构建（核心）

### 2.1 总体方法论

**核心问题**: VisDrone 提供的是逐帧目标检测 GT 标注。我们需要将其转换为 **适合流式视频 VLM 训练/评测的 QA 数据集**。

**方法论来源**:

| 来源 | 核心做法 | 我们借鉴的部分 |
|------|----------|---------------|
| **VideoLLM-online** (CVPR 2024) | 离线标注 → LLM 生成流式对话 | 在时间线上随机插入提问 |
| **MVBench** (CVPR 2024) | Static-to-Dynamic: 从标注自动生成多选 QA | 基于标注生成客观可验证的 QA |
| **临床场景论文** | smolagents Tool 分解复杂场景理解 | 工具化原子分析 + Agent 综合推理 |
| **OVO-Bench** (CVPR 2025) | 3 类 QA: backward/realtime/forward | QA 类型分类体系 |

**三阶段数据构建流水线**:

```
┌─────────────────────────────────────────────────────────────────┐
│  Stage 1: 结构化场景描述生成                                       │
│    输入: VisDrone 原始标注 (bbox+类别+遮挡)                        │
│    工具: FrameSceneAnalyzerTool + SpatialRelationTool             │
│           + TrafficDensityTool                                    │
│    输出: 每帧结构化场景描述 JSON                                    │
│          (目标计数 / 类别分布 / 空间关系 / 密度级别 / 遮挡统计)      │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│  Stage 2: 时序事件链提取                                          │
│    输入: 多帧场景描述序列（按采样间隔排列）                          │
│    工具: TemporalDiffTool (相邻帧差分)                             │
│    输出: 事件链 JSON                                              │
│          (目标进出 / 数量变化 / 密度变化 / 类别切换 / 交通趋势)      │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│  Stage 3: QA 对生成 + 质量过滤                                     │
│    输入: 结构化场景 + 事件链                                        │
│    工具: StreamingQAGenerator (模板 + 规则)                        │
│    输出: 多选/开放式 QA 对 (L1/L2/L3 三级难度)                      │
│    过滤: 完整性 / 选项合理性 / 去重                                 │
└─────────────────────────────────────────────────────────────────┘
```

**关键设计决策**:

| 决策 | 选择 | 理由 |
|------|------|------|
| QA 格式 | **多选题为主 + 少量开放式** | 多选题评测客观自动化；与 OVO-Bench 对齐 |
| 构建方式 | **smolagents Tool Pipeline** | 可复现、可扩展、质量可控 |
| 时间粒度 | **chunk 级 (4帧/chunk ≈ 2秒)** | 匹配流式引擎的 chunk 粒度 |
| 难度分级 | **L1 事实 → L2 时序 → L3 推理** | 分层评估，L2 是核心差异化指标 |
| 答案来源 | **全部基于 GT 标注推导** | 答案客观确定，不依赖模型幻觉 |

### 2.2 smolagents 驱动的自动化标注 Pipeline

#### 2.2.1 设计理念（借鉴临床场景论文）

临床场景论文的核心思路：**用 smolagents 将复杂场景理解分解为多个结构化 Tool 调用，每个 Tool 负责一个原子感知任务，最后由 Agent 综合所有 Tool 输出进行推理。**

我们将此迁移到 UAV 场景：

```
┌─────────────────────────────────────────────────────────┐
│                   smolagents CodeAgent                   │
│  任务: "分析 VisDrone 视频的第 N 段，生成流式 QA"         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌───────────────┐  ┌───────────────┐  ┌──────────────┐│
│  │ FrameScene    │  │ TemporalDiff  │  │ Spatial      ││
│  │ AnalyzerTool  │  │    Tool       │  │ RelationTool ││
│  │               │  │               │  │              ││
│  │ 输入: 帧标注  │  │ 输入: 两帧描述│  │ 输入: 帧标注 ││
│  │ 输出: 场景JSON│  │ 输出: 变化事件│  │ 输出: 空间关系││
│  └───────────────┘  └───────────────┘  └──────────────┘│
│                                                         │
│  ┌───────────────┐  ┌───────────────┐                  │
│  │ TrafficDensity│  │ Streaming QA  │                  │
│  │    Tool       │  │   Generator   │                  │
│  │               │  │               │                  │
│  │ 车流密度分析  │  │ QA 生成+过滤  │                  │
│  └───────────────┘  └───────────────┘                  │
└─────────────────────────────────────────────────────────┘
```

#### 2.2.2 Tool 实现代码

```python
# ═══ 文件: temporal_encoding/dataset/scene_tools.py ═══

"""
smolagents Tool 实现 — Stage 1 场景分析工具集。

每个 Tool 遵循 smolagents 规范:
  - name: 工具唯一标识
  - description: Agent 用于理解何时调用此工具
  - inputs: 参数定义 dict
  - output_type: 输出类型
  - forward(): 执行逻辑
"""

import json
import math
from smolagents import Tool
from visdrone_loader import VALID_CATEGORIES, VISDRONE_CATEGORIES


class FrameSceneAnalyzerTool(Tool):
    """
    分析单帧场景：从 VisDrone 标注生成结构化描述。
    纯确定性分析，不调用任何模型。
    """
    name = "frame_scene_analyzer"
    description = (
        "Analyzes a single UAV video frame using ground-truth annotations. "
        "Returns: object counts by category, density level (empty/sparse/"
        "moderate/dense/very_dense), 3x3 spatial grid distribution, "
        "occlusion statistics, and dominant category."
    )
    inputs = {
        "frame_annotation_json": {
            "type": "string",
            "description": "JSON string: {frame_id, objects: [{bbox_left, bbox_top, bbox_width, bbox_height, score, category_id, truncation, occlusion}]}"
        },
        "image_width": {"type": "integer", "description": "Frame width in pixels"},
        "image_height": {"type": "integer", "description": "Frame height in pixels"},
    }
    output_type = "string"

    def forward(self, frame_annotation_json: str,
                image_width: int, image_height: int) -> str:
        anno = json.loads(frame_annotation_json)
        objects = anno.get("objects", [])

        # 过滤有效目标 (排除 ignored=0, others=11)
        valid = [o for o in objects
                 if o["category_id"] in VALID_CATEGORIES and o.get("score", 1) == 1]

        # 按类别统计
        counts = {}
        for o in valid:
            cat = VISDRONE_CATEGORIES[o["category_id"]]
            counts[cat] = counts.get(cat, 0) + 1

        # 3×3 空间网格分布
        grid = {f"r{r}c{c}": 0 for r in range(3) for c in range(3)}
        for o in valid:
            cx = o["bbox_left"] + o["bbox_width"] / 2
            cy = o["bbox_top"] + o["bbox_height"] / 2
            col = min(int(cx / image_width * 3), 2)
            row = min(int(cy / image_height * 3), 2)
            grid[f"r{row}c{col}"] += 1

        # 密度级别
        total = len(valid)
        if total == 0:
            density = "empty"
        elif total < 10:
            density = "sparse"
        elif total < 30:
            density = "moderate"
        elif total < 80:
            density = "dense"
        else:
            density = "very_dense"

        # 遮挡统计
        occluded = sum(1 for o in valid if o.get("occlusion", 0) >= 1)
        heavily_occluded = sum(1 for o in valid if o.get("occlusion", 0) >= 2)

        scene = {
            "total_objects": total,
            "category_counts": counts,
            "density_level": density,
            "spatial_distribution": grid,
            "occlusion_stats": {
                "partially_occluded": occluded,
                "heavily_occluded": heavily_occluded,
            },
            "dominant_category": max(counts, key=counts.get) if counts else "none",
        }
        return json.dumps(scene, ensure_ascii=False)


class TemporalDiffTool(Tool):
    """
    比较两个时间点的场景描述，提取时序变化事件。
    这是流式 QA 数据集的核心 — 只有跨帧对比才能生成时序推理问题。
    """
    name = "temporal_diff_analyzer"
    description = (
        "Compares scene descriptions at two time points. Extracts temporal "
        "changes: object entering/leaving, count changes per category, "
        "density level shifts, dominant category changes."
    )
    inputs = {
        "scene_t1_json": {"type": "string", "description": "Scene JSON at earlier time"},
        "scene_t2_json": {"type": "string", "description": "Scene JSON at later time"},
        "t1_seconds": {"type": "number", "description": "Timestamp of t1 (seconds)"},
        "t2_seconds": {"type": "number", "description": "Timestamp of t2 (seconds)"},
    }
    output_type = "string"

    def forward(self, scene_t1_json: str, scene_t2_json: str,
                t1_seconds: float, t2_seconds: float) -> str:
        s1 = json.loads(scene_t1_json)
        s2 = json.loads(scene_t2_json)
        events = []

        # 1. 目标总数变化
        diff = s2["total_objects"] - s1["total_objects"]
        if abs(diff) >= 2:
            direction = "increased" if diff > 0 else "decreased"
            events.append({
                "type": "count_change",
                "description": f"Total objects {direction} by {abs(diff)}",
                "from": s1["total_objects"],
                "to": s2["total_objects"],
                "magnitude": abs(diff),
            })

        # 2. 各类别变化
        all_cats = set(list(s1["category_counts"].keys()) +
                       list(s2["category_counts"].keys()))
        for cat in all_cats:
            c1 = s1["category_counts"].get(cat, 0)
            c2 = s2["category_counts"].get(cat, 0)
            if c1 == 0 and c2 > 0:
                events.append({"type": "object_appear", "category": cat,
                               "count": c2,
                               "description": f"{c2} {cat}(s) appeared"})
            elif c1 > 0 and c2 == 0:
                events.append({"type": "object_disappear", "category": cat,
                               "count": c1,
                               "description": f"{c1} {cat}(s) left"})
            elif abs(c2 - c1) >= 2:
                dir_ = "increased" if c2 > c1 else "decreased"
                events.append({"type": "category_change", "category": cat,
                               "from": c1, "to": c2,
                               "description": f"{cat} {dir_} {c1}→{c2}"})

        # 3. 密度变化
        if s1["density_level"] != s2["density_level"]:
            events.append({
                "type": "density_change",
                "from": s1["density_level"],
                "to": s2["density_level"],
                "description": f"Density: {s1['density_level']}→{s2['density_level']}",
            })

        # 4. 主导类别变化
        if s1["dominant_category"] != s2["dominant_category"]:
            events.append({
                "type": "dominant_change",
                "from": s1["dominant_category"],
                "to": s2["dominant_category"],
                "description": f"Dominant: {s1['dominant_category']}→{s2['dominant_category']}",
            })

        result = {
            "time_span": {"from": t1_seconds, "to": t2_seconds,
                          "duration": round(t2_seconds - t1_seconds, 2)},
            "events": events,
            "has_significant_change": len(events) >= 2,
        }
        return json.dumps(result, ensure_ascii=False)


class SpatialRelationTool(Tool):
    """分析帧内目标的空间关系（用于空间推理类 QA）。"""
    name = "spatial_relation_analyzer"
    description = (
        "Analyzes spatial relationships between objects in a UAV frame: "
        "pairwise directions, left/right/top/bottom distribution."
    )
    inputs = {
        "frame_annotation_json": {"type": "string", "description": "Frame annotation JSON"},
        "image_width": {"type": "integer", "description": "Frame width"},
        "image_height": {"type": "integer", "description": "Frame height"},
    }
    output_type = "string"

    def forward(self, frame_annotation_json: str,
                image_width: int, image_height: int) -> str:
        anno = json.loads(frame_annotation_json)
        objects = [o for o in anno.get("objects", [])
                   if o["category_id"] in VALID_CATEGORIES and o.get("score", 1) == 1]

        relations = []
        if len(objects) >= 2:
            # 取面积最大的 5 个目标做配对分析
            sorted_objs = sorted(objects,
                                 key=lambda o: o["bbox_width"] * o["bbox_height"],
                                 reverse=True)[:5]
            for i in range(len(sorted_objs)):
                for j in range(i + 1, len(sorted_objs)):
                    o1, o2 = sorted_objs[i], sorted_objs[j]
                    cx1 = o1["bbox_left"] + o1["bbox_width"] / 2
                    cy1 = o1["bbox_top"] + o1["bbox_height"] / 2
                    cx2 = o2["bbox_left"] + o2["bbox_width"] / 2
                    cy2 = o2["bbox_top"] + o2["bbox_height"] / 2
                    dx, dy = cx2 - cx1, cy2 - cy1
                    direction = ("right of" if dx > 0 else "left of") if abs(dx) > abs(dy) \
                                else ("below" if dy > 0 else "above")
                    cat1 = VISDRONE_CATEGORIES[o1["category_id"]]
                    cat2 = VISDRONE_CATEGORIES[o2["category_id"]]
                    relations.append({
                        "object1": cat1, "object2": cat2,
                        "relation": f"{cat2} is {direction} {cat1}",
                    })

        left = sum(1 for o in objects if o["bbox_left"] + o["bbox_width"]/2 < image_width/2)
        top = sum(1 for o in objects if o["bbox_top"] + o["bbox_height"]/2 < image_height/2)

        result = {
            "pairwise_relations": relations[:10],
            "area_distribution": {
                "left_half": left, "right_half": len(objects) - left,
                "top_half": top, "bottom_half": len(objects) - top,
            },
        }
        return json.dumps(result, ensure_ascii=False)


class TrafficDensityTool(Tool):
    """专门分析交通密度和拥堵程度（UAV 最常见场景）。"""
    name = "traffic_density_analyzer"
    description = (
        "Analyzes traffic density: vehicle count, pedestrian count, "
        "area coverage ratio, congestion level (no_traffic/free_flow/"
        "light_traffic/moderate_traffic/heavy_congestion)."
    )
    inputs = {
        "frame_annotation_json": {"type": "string", "description": "Frame annotation JSON"},
        "image_width": {"type": "integer", "description": "Frame width"},
        "image_height": {"type": "integer", "description": "Frame height"},
    }
    output_type = "string"

    VEHICLE_IDS = {4, 5, 6, 7, 8, 9, 10}

    def forward(self, frame_annotation_json: str,
                image_width: int, image_height: int) -> str:
        anno = json.loads(frame_annotation_json)
        objects = [o for o in anno.get("objects", [])
                   if o["category_id"] in VALID_CATEGORIES and o.get("score", 1) == 1]

        vehicles = [o for o in objects if o["category_id"] in self.VEHICLE_IDS]
        pedestrians = [o for o in objects if o["category_id"] in (1, 2)]

        total_area = image_width * image_height
        vehicle_area = sum(o["bbox_width"] * o["bbox_height"] for o in vehicles)
        coverage = vehicle_area / total_area if total_area > 0 else 0

        if len(vehicles) == 0:    congestion = "no_traffic"
        elif coverage < 0.02:     congestion = "free_flow"
        elif coverage < 0.08:     congestion = "light_traffic"
        elif coverage < 0.20:     congestion = "moderate_traffic"
        else:                     congestion = "heavy_congestion"

        vtypes = {}
        for v in vehicles:
            c = VISDRONE_CATEGORIES[v["category_id"]]
            vtypes[c] = vtypes.get(c, 0) + 1

        result = {
            "vehicle_count": len(vehicles),
            "pedestrian_count": len(pedestrians),
            "vehicle_coverage_ratio": round(coverage, 4),
            "congestion_level": congestion,
            "vehicle_types": vtypes,
        }
        return json.dumps(result, ensure_ascii=False)
```

#### 2.2.3 视频级事件链提取器

```python
# ═══ 文件: temporal_encoding/dataset/event_extractor.py ═══

"""
VideoEventChainExtractor — 从视频序列提取完整的时序事件链。
编排 Scene Tools 的输出，生成 Stage 2 的中间产物。
"""

import json
from typing import List, Dict
from visdrone_loader import VideoSequence, VALID_CATEGORIES, VISDRONE_CATEGORIES
from scene_tools import (
    FrameSceneAnalyzerTool, TemporalDiffTool,
    TrafficDensityTool, SpatialRelationTool,
)


class VideoEventChainExtractor:
    """从视频序列提取事件链（多帧差分分析）"""

    def __init__(self, fps: float = 2.0, chunk_size: int = 4):
        self.fps = fps
        self.chunk_size = chunk_size
        self.frame_analyzer = FrameSceneAnalyzerTool()
        self.temporal_diff = TemporalDiffTool()
        self.traffic_analyzer = TrafficDensityTool()

    def extract_from_sequence(self, seq: VideoSequence,
                              sample_interval: int = 30) -> Dict:
        """
        从一个 VisDrone 视频序列提取完整事件链。

        Args:
            seq: 视频序列
            sample_interval: 采样间隔(帧数), 默认30帧(~1秒@30fps)

        Returns:
            {video_name, num_frames, duration_seconds, scenes[], events[], traffic_timeline[]}
        """
        if not seq.frames:
            return {"scenes": [], "events": [], "traffic_timeline": []}

        first_img = seq.get_frame_image(0)
        img_w, img_h = first_img.size

        sampled_indices = list(range(0, len(seq.frames), sample_interval))

        # Stage 1: 每个采样帧 → 场景描述
        scenes = []
        for idx in sampled_indices:
            frame = seq.frames[idx]
            anno_json = json.dumps({
                "frame_id": frame.frame_id,
                "objects": [
                    {
                        "bbox_left": o.bbox_left, "bbox_top": o.bbox_top,
                        "bbox_width": o.bbox_width, "bbox_height": o.bbox_height,
                        "score": o.score, "category_id": o.category_id,
                        "truncation": o.truncation, "occlusion": o.occlusion,
                    }
                    for o in frame.objects
                ]
            })
            scene_json = self.frame_analyzer.forward(anno_json, img_w, img_h)
            traffic_json = self.traffic_analyzer.forward(anno_json, img_w, img_h)
            timestamp = idx / 30.0  # VisDrone 原始 ~30fps
            scenes.append({
                "frame_idx": idx,
                "timestamp": round(timestamp, 2),
                "scene": json.loads(scene_json),
                "traffic": json.loads(traffic_json),
            })

        # Stage 2: 相邻场景差分 → 事件链
        events = []
        for i in range(1, len(scenes)):
            diff_json = self.temporal_diff.forward(
                json.dumps(scenes[i-1]["scene"]),
                json.dumps(scenes[i]["scene"]),
                scenes[i-1]["timestamp"],
                scenes[i]["timestamp"],
            )
            diff = json.loads(diff_json)
            if diff["events"]:
                events.append(diff)

        # 交通密度时间线
        traffic_timeline = [
            {
                "timestamp": s["timestamp"],
                "congestion": s["traffic"]["congestion_level"],
                "vehicle_count": s["traffic"]["vehicle_count"],
                "pedestrian_count": s["traffic"]["pedestrian_count"],
            }
            for s in scenes
        ]

        return {
            "video_name": seq.seq_name,
            "num_frames": seq.num_frames,
            "duration_seconds": round(seq.num_frames / 30.0, 2),
            "image_size": [img_w, img_h],
            "scenes": scenes,
            "events": events,
            "traffic_timeline": traffic_timeline,
        }
```

#### 2.2.4 QA 生成器

```python
# ═══ 文件: temporal_encoding/dataset/qa_generator.py ═══

"""
StreamingQAGenerator — 从结构化场景+事件链生成流式 QA 对。

QA 类型三级体系:
  L1 (事实): 单帧可答 — 目标计数/类别存在/密度/主导类别
  L2 (时序): 跨帧才能答 — 数量变化/密度变化/目标进出/趋势 ⭐ 核心
  L3 (推理): 综合分析 — 事件排序/趋势预测

所有 QA 的答案都基于 GT 标注客观推导，不依赖模型幻觉。
"""

import json
import random
from typing import List, Dict
from visdrone_loader import VALID_CATEGORIES, VISDRONE_CATEGORIES


class StreamingQAGenerator:

    def __init__(self, seed: int = 42):
        random.seed(seed)

    def generate_for_video(self, video_data: Dict,
                           max_qa_per_video: int = 30) -> List[Dict]:
        """
        为一个视频生成完整的流式 QA 集。

        每条 QA 格式:
        {
            "video_name", "question", "options" (A/B/C/D),
            "answer" (字母), "answer_text", "query_timestamp" (秒),
            "qa_type" (realtime/backward/forward),
            "difficulty" (L1/L2/L3), "source_evidence"
        }
        """
        qa = []
        scenes = video_data["scenes"]
        events = video_data["events"]
        traffic = video_data["traffic_timeline"]

        if len(scenes) < 3:
            return qa

        # ── L1 ──
        qa.extend(self._l1_counting(scenes))
        qa.extend(self._l1_category_presence(scenes))
        qa.extend(self._l1_density(scenes))
        qa.extend(self._l1_dominant(scenes))

        # ── L2 ⭐ ──
        qa.extend(self._l2_count_change(events))
        qa.extend(self._l2_density_change(events))
        qa.extend(self._l2_object_appear_disappear(events))
        qa.extend(self._l2_traffic_trend(traffic))

        # ── L3 ──
        qa.extend(self._l3_event_sequence(events))
        qa.extend(self._l3_congestion_prediction(traffic))

        random.shuffle(qa)
        qa = qa[:max_qa_per_video]
        for q in qa:
            q["video_name"] = video_data["video_name"]
        return qa

    # ═══ L1: 事实性 (realtime) ═══════════════════════════════

    def _l1_counting(self, scenes):
        """当前帧有多少 X？"""
        qa = []
        for sd in random.sample(scenes, min(3, len(scenes))):
            s, ts = sd["scene"], sd["timestamp"]
            if not s["category_counts"]:
                continue
            cat = random.choice(list(s["category_counts"].keys()))
            correct = s["category_counts"][cat]
            wrong = self._count_distractors(correct)
            opts_raw = [correct] + wrong
            random.shuffle(opts_raw)
            labels = ["A", "B", "C", "D"]
            qa.append({
                "question": f"How many {cat}(s) are currently visible in the aerial view?",
                "options": [f"{labels[i]}. {v}" for i, v in enumerate(opts_raw)],
                "answer": labels[opts_raw.index(correct)],
                "answer_text": str(correct),
                "query_timestamp": ts,
                "qa_type": "realtime",
                "difficulty": "L1",
                "source_evidence": f"Frame@{ts}s: {cat}={correct}",
            })
        return qa

    def _l1_category_presence(self, scenes):
        """当前帧是否存在 X？"""
        qa = []
        for sd in random.sample(scenes, min(3, len(scenes))):
            s, ts = sd["scene"], sd["timestamp"]
            present = set(s["category_counts"].keys())
            absent = set(VALID_CATEGORIES.values()) - present
            if present and absent:
                cat_y = random.choice(list(present))
                qa.append({
                    "question": f"Is there any {cat_y} visible in the current aerial view?",
                    "options": ["A. Yes", "B. No"],
                    "answer": "A", "answer_text": "Yes",
                    "query_timestamp": ts, "qa_type": "realtime", "difficulty": "L1",
                    "source_evidence": f"Frame@{ts}s: {cat_y} present ({s['category_counts'][cat_y]})",
                })
                cat_n = random.choice(list(absent))
                qa.append({
                    "question": f"Is there any {cat_n} visible in the current aerial view?",
                    "options": ["A. Yes", "B. No"],
                    "answer": "B", "answer_text": "No",
                    "query_timestamp": ts, "qa_type": "realtime", "difficulty": "L1",
                    "source_evidence": f"Frame@{ts}s: {cat_n} absent",
                })
        return qa

    def _l1_density(self, scenes):
        """场景密度如何？"""
        density_text = {
            "empty": "Almost empty with few or no objects",
            "sparse": "Sparse with only a few objects",
            "moderate": "Moderately populated",
            "dense": "Densely populated with many objects",
            "very_dense": "Very densely crowded",
        }
        qa = []
        for sd in random.sample(scenes, min(2, len(scenes))):
            s, ts = sd["scene"], sd["timestamp"]
            correct = s["density_level"]
            wrong = [l for l in density_text if l != correct]
            wrong = random.sample(wrong, min(3, len(wrong)))
            opts = [density_text[correct]] + [density_text[w] for w in wrong]
            random.shuffle(opts)
            labels = ["A", "B", "C", "D"]
            qa.append({
                "question": "How would you describe the current scene density from the aerial perspective?",
                "options": [f"{labels[i]}. {v}" for i, v in enumerate(opts)],
                "answer": labels[opts.index(density_text[correct])],
                "answer_text": density_text[correct],
                "query_timestamp": ts, "qa_type": "realtime", "difficulty": "L1",
                "source_evidence": f"Frame@{ts}s: density={correct}, total={s['total_objects']}",
            })
        return qa

    def _l1_dominant(self, scenes):
        """最常见的目标类别？"""
        qa = []
        for sd in random.sample(scenes, min(2, len(scenes))):
            s, ts = sd["scene"], sd["timestamp"]
            if s["dominant_category"] == "none":
                continue
            correct = s["dominant_category"]
            wrong = [c for c in VALID_CATEGORIES.values() if c != correct]
            wrong = random.sample(wrong, min(3, len(wrong)))
            opts = [correct] + wrong
            random.shuffle(opts)
            labels = ["A", "B", "C", "D"]
            qa.append({
                "question": "What is the most common type of object currently visible from the drone's view?",
                "options": [f"{labels[i]}. {v}" for i, v in enumerate(opts)],
                "answer": labels[opts.index(correct)],
                "answer_text": correct,
                "query_timestamp": ts, "qa_type": "realtime", "difficulty": "L1",
                "source_evidence": f"Frame@{ts}s: dominant={correct}",
            })
        return qa

    # ═══ L2: 时序推理 (backward) ⭐ ═════════════════════════

    def _l2_count_change(self, events):
        """目标数量增减？"""
        qa = []
        for ed in events:
            for evt in ed["events"]:
                if evt["type"] == "count_change":
                    ts = ed["time_span"]["to"]
                    increased = evt["to"] > evt["from"]
                    qa.append({
                        "question": "Compared to earlier frames, has the total number of objects increased or decreased?",
                        "options": [
                            f"A. Increased (from {evt['from']} to {evt['to']})",
                            f"B. Decreased (from {evt['from']} to {max(0, evt['from'] - evt['magnitude'])})",
                            "C. Remained roughly the same",
                            "D. Cannot be determined",
                        ],
                        "answer": "A" if increased else "B",
                        "answer_text": f"{'Increased' if increased else 'Decreased'} {evt['from']}→{evt['to']}",
                        "query_timestamp": ts, "qa_type": "backward", "difficulty": "L2",
                        "source_evidence": f"Count: {evt['from']}→{evt['to']}",
                    })
        return qa[:5]

    def _l2_density_change(self, events):
        """密度如何变化？"""
        qa = []
        for ed in events:
            for evt in ed["events"]:
                if evt["type"] == "density_change":
                    ts = ed["time_span"]["to"]
                    qa.append({
                        "question": "How has the scene density changed over the recent video segment?",
                        "options": [
                            f"A. From {evt['from']} to {evt['to']}",
                            f"B. From {evt['to']} to {evt['from']}",
                            f"C. Remained at {evt['from']}",
                            "D. Fluctuated between multiple levels",
                        ],
                        "answer": "A",
                        "answer_text": f"{evt['from']}→{evt['to']}",
                        "query_timestamp": ts, "qa_type": "backward", "difficulty": "L2",
                        "source_evidence": f"Density: {evt['from']}→{evt['to']}",
                    })
        return qa[:3]

    def _l2_object_appear_disappear(self, events):
        """新出现/消失了什么？"""
        qa = []
        for ed in events:
            for evt in ed["events"]:
                if evt["type"] in ("object_appear", "object_disappear"):
                    ts = ed["time_span"]["to"]
                    action = "appeared" if evt["type"] == "object_appear" else "disappeared"
                    correct = evt["category"]
                    wrong = [c for c in VALID_CATEGORIES.values() if c != correct]
                    wrong = random.sample(wrong, min(3, len(wrong)))
                    opts = [correct] + wrong
                    random.shuffle(opts)
                    labels = ["A", "B", "C", "D"]
                    qa.append({
                        "question": f"What type of object has recently {action} in the scene?",
                        "options": [f"{labels[i]}. {v}" for i, v in enumerate(opts)],
                        "answer": labels[opts.index(correct)],
                        "answer_text": correct,
                        "query_timestamp": ts, "qa_type": "backward", "difficulty": "L2",
                        "source_evidence": f"{correct} {action} @{ts}s",
                    })
        return qa[:5]

    def _l2_traffic_trend(self, traffic):
        """交通趋势？"""
        if len(traffic) < 5:
            return []
        mid = len(traffic) // 2
        recent = traffic[mid:]
        ts = recent[-1]["timestamp"]
        counts = [t["vehicle_count"] for t in recent]
        diff = counts[-1] - counts[0]
        if diff > 3:    trend, ans = "increasing", "A"
        elif diff < -3: trend, ans = "decreasing", "B"
        else:           trend, ans = "stable", "C"
        return [{
            "question": "Based on your observation, what is the overall trend of traffic flow?",
            "options": [
                "A. Traffic is gradually increasing",
                "B. Traffic is gradually decreasing",
                "C. Traffic has remained relatively stable",
                "D. Traffic is fluctuating unpredictably",
            ],
            "answer": ans,
            "answer_text": f"Traffic is {trend}",
            "query_timestamp": ts, "qa_type": "backward", "difficulty": "L2",
            "source_evidence": f"Vehicle counts: {counts}",
        }]

    # ═══ L3: 复杂推理 ═══════════════════════════════════════

    def _l3_event_sequence(self, events):
        """事件排序"""
        all_evts = []
        for e in events:
            for evt in e["events"]:
                all_evts.append({"desc": evt["description"],
                                 "ts": e["time_span"]["to"]})
        if len(all_evts) < 3:
            return []
        selected = random.sample(all_evts, min(3, len(all_evts)))
        correct = sorted(selected, key=lambda x: x["ts"])
        correct_text = " → ".join(e["desc"] for e in correct)
        reversed_text = " → ".join(e["desc"] for e in reversed(correct))
        shuffled = correct.copy()
        random.shuffle(shuffled)
        shuffled_text = " → ".join(e["desc"] for e in shuffled)
        ts = max(e["ts"] for e in selected)
        return [{
            "question": "What is the correct chronological order of these observed events?",
            "options": [
                f"A. {correct_text}",
                f"B. {reversed_text}",
                f"C. {shuffled_text}",
                "D. They occurred simultaneously",
            ],
            "answer": "A",
            "answer_text": correct_text,
            "query_timestamp": ts, "qa_type": "backward", "difficulty": "L3",
            "source_evidence": f"Events: {json.dumps(correct)}",
        }]

    def _l3_congestion_prediction(self, traffic):
        """趋势预测"""
        if len(traffic) < 5:
            return []
        counts = [t["vehicle_count"] for t in traffic]
        recent = counts[-5:]
        slope = (recent[-1] - recent[0]) / 4
        ts = traffic[-1]["timestamp"]
        if slope > 1:     pred, ans = "worsen", "A"
        elif slope < -1:  pred, ans = "improve", "B"
        else:             pred, ans = "remain similar", "C"
        return [{
            "question": "Based on the traffic trend, what is your prediction for the near future?",
            "options": [
                "A. Congestion is likely to worsen",
                "B. Congestion is likely to improve",
                "C. Traffic will likely remain at a similar level",
                "D. Insufficient information to predict",
            ],
            "answer": ans,
            "answer_text": f"Likely to {pred}",
            "query_timestamp": ts, "qa_type": "forward", "difficulty": "L3",
            "source_evidence": f"Recent counts: {recent}, slope: {slope:.2f}",
        }]

    # ═══ 工具 ════════════════════════════════════════════════

    @staticmethod
    def _count_distractors(correct: int) -> list:
        d = set()
        for c in [correct+1, correct+2, correct+3,
                  max(0, correct-1), max(0, correct-2), correct*2]:
            if c != correct:
                d.add(c)
        return list(d)[:3]
```

### 2.3 QA 类型设计与模板

#### 完整 QA 类型体系

| 级别 | 类型 | qa_type | 描述 | 需要时序? | 数量/视频 |
|------|------|---------|------|-----------|----------|
| **L1** | 目标计数 | realtime | "当前画面有几辆车？" | ❌ | ~3 |
| **L1** | 类别存在 | realtime | "当前是否有行人？" | ❌ | ~3 |
| **L1** | 场景密度 | realtime | "当前场景密度？" | ❌ | ~2 |
| **L1** | 主导类别 | realtime | "最常见的目标类型？" | ❌ | ~2 |
| **L2** ⭐ | 数量变化 | backward | "目标数增加还是减少？" | ✅ | ~5 |
| **L2** ⭐ | 密度变化 | backward | "密度如何变化？" | ✅ | ~3 |
| **L2** ⭐ | 目标进出 | backward | "新出现了什么目标？" | ✅ | ~5 |
| **L2** ⭐ | 交通趋势 | backward | "交通流量趋势？" | ✅ | ~1 |
| **L3** | 事件排序 | backward | "按时间排序？" | ✅ | ~1 |
| **L3** | 趋势预测 | forward | "交通是否会恶化？" | ✅ | ~1 |

**预计总量**: 50 视频 × ~25 QA/视频 = **~1250 QA 对**

#### 为什么选多选题？

| 维度 | 多选题 | 开放式 |
|------|--------|--------|
| 评测客观性 | ✅ accuracy 精确可比 | ❌ 需 GPT-4 评分 |
| 3B 模型适配 | ✅ 只需选择 | ⚠️ 小模型生成不稳定 |
| 答案唯一性 | ✅ 确定 | ⚠️ 多种合理表述 |
| 与 OVO-Bench 对齐 | ✅ 同体系 | — |
| 统计意义 | ✅ ~200 样本够用 | ❌ 需更多 |

### 2.4 流式对话格式转换

```python
# ═══ 文件: temporal_encoding/dataset/format_streaming.py ═══

"""
将 QA 对转换为流式推理引擎的评测格式。

输出格式 — 每个视频一个条目:
{
    "video_name": str,
    "video_path": str,
    "fps": float,
    "chunk_size": int,
    "queries": [{
        "query_id", "timestamp", "chunk_index",
        "question", "options", "answer", "answer_text",
        "qa_type", "difficulty", "source_evidence"
    }]
}
"""

import json
from pathlib import Path
from typing import List, Dict


def convert_to_streaming_eval_format(
    qa_pairs: List[Dict],
    video_root: str,
    output_path: str,
    fps: float = 2.0,
    chunk_size: int = 4,
):
    by_video = {}
    for qa in qa_pairs:
        vname = qa["video_name"]
        by_video.setdefault(vname, []).append(qa)

    dataset = []
    for vname, qas in by_video.items():
        qas = sorted(qas, key=lambda x: x["query_timestamp"])
        queries = []
        for i, qa in enumerate(qas):
            ts = qa["query_timestamp"]
            queries.append({
                "query_id": f"{vname}_q{i+1:03d}",
                "timestamp": ts,
                "chunk_index": int(ts * fps / chunk_size),
                "question": qa["question"],
                "options": qa["options"],
                "answer": qa["answer"],
                "answer_text": qa["answer_text"],
                "qa_type": qa["qa_type"],
                "difficulty": qa["difficulty"],
                "source_evidence": qa["source_evidence"],
            })
        dataset.append({
            "video_name": vname,
            "video_path": str(Path(video_root) / "sequences" / vname),
            "fps": fps,
            "chunk_size": chunk_size,
            "queries": queries,
        })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    total = sum(len(d["queries"]) for d in dataset)
    by_type = {}
    by_diff = {}
    for d in dataset:
        for q in d["queries"]:
            by_type[q["qa_type"]] = by_type.get(q["qa_type"], 0) + 1
            by_diff[q["difficulty"]] = by_diff.get(q["difficulty"], 0) + 1

    print(f"✅ Dataset saved to {output_path}")
    print(f"   Videos: {len(dataset)} | Total QA: {total}")
    print(f"   By type: {by_type}")
    print(f"   By difficulty: {by_diff}")
```

### 2.5 质量控制与过滤

```python
# ═══ 文件: temporal_encoding/dataset/quality_filter.py ═══

def filter_qa_dataset(qa_pairs: list) -> list:
    """
    质量过滤:
      1. 字段完整性
      2. 选项数量 ≥ 2
      3. 答案在选项中
      4. 问题长度 10-500 字符
      5. 时间戳非负
      6. 问题去重
    """
    filtered = []
    for qa in qa_pairs:
        if not all(k in qa for k in ["question", "options", "answer", "query_timestamp"]):
            continue
        if len(qa["options"]) < 2:
            continue
        if qa["answer"] not in [o[0] for o in qa["options"]]:
            continue
        if not (10 <= len(qa["question"]) <= 500):
            continue
        if qa["query_timestamp"] < 0:
            continue
        filtered.append(qa)

    seen = set()
    deduped = []
    for qa in filtered:
        key = qa["question"].lower().strip()
        if key not in seen:
            seen.add(key)
            deduped.append(qa)

    print(f"[Quality] {len(qa_pairs)} → {len(filtered)} (valid) → {len(deduped)} (dedup)")
    return deduped
```

### 2.6 主构建脚本

```python
# ═══ 文件: temporal_encoding/dataset/build_dataset_main.py ═══

"""
一键构建 StreamUAV 流式 QA 数据集。

用法:
  python build_dataset_main.py \
    --visdrone-root /root/autodl-tmp/datasets/visdrone/VID-train \
    --output /root/autodl-tmp/datasets/stream_uav_qa/ \
    --max-videos 50
"""

import argparse, json, os
from visdrone_loader import load_visdrone_vid
from event_extractor import VideoEventChainExtractor
from qa_generator import StreamingQAGenerator
from format_streaming import convert_to_streaming_eval_format
from quality_filter import filter_qa_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--visdrone-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-videos", type=int, default=50)
    parser.add_argument("--max-qa-per-video", type=int, default=30)
    parser.add_argument("--sample-interval", type=int, default=30,
                        help="标注采样间隔(帧), 30=~1秒@30fps")
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Step 1: 加载 VisDrone
    print("=" * 60)
    print("Step 1: Loading VisDrone-VID...")
    sequences = load_visdrone_vid(args.visdrone_root)[:args.max_videos]
    print(f"Selected {len(sequences)} sequences")

    # Step 2: 提取事件链
    print("\n" + "=" * 60)
    print("Step 2: Extracting event chains...")
    extractor = VideoEventChainExtractor(fps=args.fps, chunk_size=args.chunk_size)
    all_video_data = []
    for i, seq in enumerate(sequences):
        print(f"  [{i+1}/{len(sequences)}] {seq.seq_name} ({seq.num_frames} frames)")
        vd = extractor.extract_from_sequence(seq, sample_interval=args.sample_interval)
        all_video_data.append(vd)

    with open(os.path.join(args.output, "video_events.json"), 'w') as f:
        json.dump(all_video_data, f, indent=2, ensure_ascii=False)

    # Step 3: 生成 QA
    print("\n" + "=" * 60)
    print("Step 3: Generating QA pairs...")
    gen = StreamingQAGenerator(seed=args.seed)
    all_qa = []
    for vd in all_video_data:
        qa = gen.generate_for_video(vd, max_qa_per_video=args.max_qa_per_video)
        all_qa.extend(qa)
        print(f"  {vd['video_name']}: {len(qa)} QA pairs")

    # Step 4: 过滤
    print("\n" + "=" * 60)
    print("Step 4: Quality filtering...")
    filtered = filter_qa_dataset(all_qa)

    # Step 5: 格式化
    print("\n" + "=" * 60)
    print("Step 5: Converting to eval format...")
    eval_path = os.path.join(args.output, "stream_uav_eval.json")
    convert_to_streaming_eval_format(
        filtered, video_root=args.visdrone_root, output_path=eval_path,
        fps=args.fps, chunk_size=args.chunk_size,
    )

    with open(os.path.join(args.output, "all_qa_pairs.json"), 'w') as f:
        json.dump(filtered, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f"✅ Done! {len(filtered)} QA pairs across {len(all_video_data)} videos")
    print(f"   Output: {args.output}")


if __name__ == "__main__":
    main()
```

---

## 3. 流式推理引擎适配与评测

### 3.1 评测脚本

```python
# ═══ 文件: temporal_encoding/evaluation/run_streaming_eval.py ═══

"""
在 StreamUAV 数据集上运行流式评测。

三组对比:
  A. Streaming (ours): 逐 chunk 喂帧 → 在时间戳 ask_choice()
  B. Offline baseline: 截取到提问时刻的完整帧 → 一次性推理
  C. Single-frame:     只用提问时刻的单帧 → 推理

核心预期:
  L1 (realtime):  Streaming ≈ Offline ≈ SingleFrame
  L2 (backward):  Streaming > SingleFrame, Streaming ≈ Offline ⭐
  L3 (complex):   Streaming ≥ Offline > SingleFrame
"""

import argparse, json, time, gc, glob
import torch
from pathlib import Path
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.stream_qwen_model import StreamQwenModel
from model.video_stream_inference import VideoStreamingInference
from model.kv_cache_eviction import EvictionConfig


def eval_streaming(model, processor, entry, eviction_config=None):
    """流式评测"""
    engine = VideoStreamingInference(
        model, processor, "cuda", eviction_config=eviction_config
    )
    fps = entry["fps"]
    chunk_size = entry["chunk_size"]
    frames = sorted(glob.glob(f"{entry['video_path']}/*.jpg"))
    if not frames:
        frames = sorted(glob.glob(f"{entry['video_path']}/*.png"))

    sample_interval = max(1, int(30.0 / fps))
    sampled = frames[::sample_interval]

    queries = sorted(entry["queries"], key=lambda q: q["timestamp"])
    qi = 0
    results = []

    for cs in range(0, len(sampled), chunk_size):
        chunk = sampled[cs:cs + chunk_size]
        if not chunk:
            break
        imgs = [Image.open(p).convert("RGB") for p in chunk]
        t_now = (cs + len(chunk)) / fps
        engine.append_video_chunk(imgs, as_video=True)

        while qi < len(queries) and queries[qi]["timestamp"] <= t_now:
            q = queries[qi]
            t0 = time.time()
            opts = [o[3:] for o in q["options"]]
            pred = engine.ask_choice(q["question"], opts)
            lat = (time.time() - t0) * 1000

            pred_label = "?"
            for o in q["options"]:
                if o[3:] == pred:
                    pred_label = o[0]
                    break

            results.append({
                "query_id": q["query_id"],
                "method": "streaming",
                "prediction": pred_label,
                "ground_truth": q["answer"],
                "correct": pred_label == q["answer"],
                "latency_ms": round(lat, 1),
                "qa_type": q["qa_type"],
                "difficulty": q["difficulty"],
            })
            qi += 1

    del engine; gc.collect(); torch.cuda.empty_cache()
    return results


def eval_offline(model, processor, entry):
    """离线评测: 截取到提问时刻的所有帧"""
    fps = entry["fps"]
    frames = sorted(glob.glob(f"{entry['video_path']}/*.jpg"))
    if not frames:
        frames = sorted(glob.glob(f"{entry['video_path']}/*.png"))
    sample_interval = max(1, int(30.0 / fps))
    sampled = frames[::sample_interval]
    results = []

    for q in entry["queries"]:
        end_idx = min(int(q["timestamp"] * fps), len(sampled))
        selected = sampled[:end_idx]
        if len(selected) > 32:
            indices = [int(i * (len(selected)-1) / 31) for i in range(32)]
            selected = [selected[i] for i in indices]

        imgs = [Image.open(p).convert("RGB") for p in selected]
        messages = [{
            "role": "user",
            "content": [
                {"type": "video", "video": imgs, "fps": fps},
                {"type": "text", "text": (
                    f"This is aerial drone footage. "
                    f"Question: {q['question']}\n"
                    + "\n".join(q['options'])
                    + "\nAnswer with the letter only (A/B/C/D)."
                )},
            ],
        }]

        t0 = time.time()
        text = processor.apply_chat_template(messages, tokenize=False,
                                              add_generation_prompt=True)
        inputs = processor(text=[text], images=None, videos=[imgs],
                           return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=10)
        resp = processor.batch_decode(out[:, inputs.input_ids.shape[1]:],
                                       skip_special_tokens=True)[0].strip()
        lat = (time.time() - t0) * 1000

        pred = "?"
        for letter in "ABCD":
            if letter in resp:
                pred = letter; break

        results.append({
            "query_id": q["query_id"], "method": "offline",
            "prediction": pred, "ground_truth": q["answer"],
            "correct": pred == q["answer"],
            "latency_ms": round(lat, 1),
            "qa_type": q["qa_type"], "difficulty": q["difficulty"],
        })
        del inputs; gc.collect(); torch.cuda.empty_cache()
    return results


def compute_metrics(results):
    if not results: return {}
    m = {
        "total": len(results),
        "overall_acc": sum(r["correct"] for r in results) / len(results),
        "avg_latency_ms": sum(r["latency_ms"] for r in results) / len(results),
    }
    for key_field, values in [("qa_type", ["realtime","backward","forward"]),
                               ("difficulty", ["L1","L2","L3"])]:
        for v in values:
            sub = [r for r in results if r[key_field] == v]
            if sub:
                m[f"{v}_acc"] = sum(r["correct"] for r in sub) / len(sub)
                m[f"{v}_count"] = len(sub)
    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--methods", nargs="+", default=["streaming","offline"])
    parser.add_argument("--max-videos", type=int, default=10)
    parser.add_argument("--max-cache-tokens", type=int, default=100000)
    parser.add_argument("--output", default="eval_results.json")
    args = parser.parse_args()

    from transformers import AutoProcessor
    print(f"Loading model: {args.model_path}")
    model = StreamQwenModel.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16).to("cuda").eval()
    processor = AutoProcessor.from_pretrained(args.model_path)

    with open(args.dataset) as f:
        dataset = json.load(f)[:args.max_videos]
    total_q = sum(len(d["queries"]) for d in dataset)
    print(f"Loaded {len(dataset)} videos, {total_q} queries")

    eviction_cfg = EvictionConfig(max_cache_tokens=args.max_cache_tokens)
    all_results = {}

    for method in args.methods:
        print(f"\n{'='*60}\n  Running: {method}\n{'='*60}")
        method_res = []
        for i, entry in enumerate(dataset):
            nq = len(entry["queries"])
            print(f"  [{i+1}/{len(dataset)}] {entry['video_name']} ({nq} queries)")

            if method == "streaming":
                res = eval_streaming(model, processor, entry, eviction_cfg)
            else:
                res = eval_offline(model, processor, entry)

            method_res.extend(res)
            acc = sum(r["correct"] for r in res) / len(res) if res else 0
            print(f"    Acc: {acc:.1%} ({sum(r['correct'] for r in res)}/{len(res)})")

        all_results[method] = method_res

    # 汇总
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    summary = {}
    for method, res in all_results.items():
        m = compute_metrics(res)
        summary[method] = m
        print(f"\n📊 {method.upper()}:")
        print(f"  Overall:  {m.get('overall_acc',0):.1%}")
        print(f"  L1:       {m.get('L1_acc',0):.1%} (n={m.get('L1_count',0)})")
        print(f"  L2 ⭐:    {m.get('L2_acc',0):.1%} (n={m.get('L2_count',0)})")
        print(f"  L3:       {m.get('L3_acc',0):.1%} (n={m.get('L3_count',0)})")
        print(f"  Latency:  {m.get('avg_latency_ms',0):.0f}ms")

    if "streaming" in summary and "offline" in summary:
        print(f"\n🎯 STREAMING GAIN:")
        for k in ["overall_acc", "L1_acc", "L2_acc", "L3_acc"]:
            s = summary["streaming"].get(k, 0)
            o = summary["offline"].get(k, 0)
            g = s - o
            e = "✅" if g >= 0 else "❌"
            print(f"  {e} {k}: {g:+.1%} (stream {s:.1%} vs offline {o:.1%})")

    with open(args.output, 'w') as f:
        json.dump({"config": vars(args), "summary": summary,
                   "detailed": all_results}, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Saved: {args.output}")


if __name__ == "__main__":
    main()
```

---

## 4. 评测与验证策略

### 4.1 为什么跳过 OVO-Bench，直接用 VisDrone

| 考量 | OVO-Bench | VisDrone 自建 |
|------|-----------|--------------|
| 下载量 | 44-144 GB | ~11 GB |
| 场景匹配 | 通用视频 ≠ UAV | ✅ UAV 原生 |
| 评测闭环 | 只能评测 | ✅ 评测 → 微调 → 再评 |
| 投稿说服力 | 通用指标 | ✅ 领域特定指标 |
| 时间成本 | 下载+适配 ~3天 | 构建 ~1天 |

**结论**: 直接在 VisDrone 上构建评测集，与最终应用场景完全一致。

### 4.2 执行命令

```bash
# ── H800 上按序执行 ──

# Step 1: 构建数据集 (~30min)
cd /root/autodl-tmp/diploma_project/temporal_encoding
python dataset/build_dataset_main.py \
    --visdrone-root /root/autodl-tmp/datasets/visdrone/VID-train \
    --output /root/autodl-tmp/datasets/stream_uav_qa/ \
    --max-videos 50

# Step 2: 流式 vs 离线评测 (~2-4h)
python evaluation/run_streaming_eval.py \
    --model-path /root/autodl-tmp/Qwen/Qwen2___5-VL-7B-Instruct \
    --dataset /root/autodl-tmp/datasets/stream_uav_qa/stream_uav_eval.json \
    --methods streaming offline \
    --max-videos 20 \
    --output eval_results_7b.json

# Step 3: 查看结果
python -c "
import json
with open('eval_results_7b.json') as f:
    d = json.load(f)
print(json.dumps(d['summary'], indent=2))
"
```

### 4.3 结果解读

```
✅ 改造成功的证据:
  1. L2 backward: streaming_acc > single_frame_acc (+5% 以上)
     → KV Cache 确实保留了历史时序信息
  2. L1 realtime: streaming_acc ≈ offline_acc (差距 <3%)
     → 改造未引入质量退化
  3. Overall: streaming_acc ≥ offline_acc × 0.9
     → 流式处理的质量损失可接受

❌ 需要微调的信号:
  1. L2 streaming < offline → KV Cache 历史信息不够
  2. L1 streaming << offline → 改造可能有 bug
  3. 大量重复/幻觉回答 → 需 LoRA 微调或换 7B
```

---

## 5. 可选: LoRA 微调

如果评测结果不理想（特别是 L2 时序推理低），在 H800 上进行 LoRA 微调：

```python
# ═══ 文件: temporal_encoding/training/convert_to_sft.py ═══

"""
将 stream_uav_eval.json 转换为 SFT 训练格式。
格式: messages list, 每条包含 system + user(含视频) + assistant(答案)。
"""

import json

def convert(input_path, output_path):
    with open(input_path) as f:
        dataset = json.load(f)

    sft_data = []
    for video in dataset:
        for q in video["queries"]:
            sft_data.append({
                "video_path": video["video_path"],
                "timestamp": q["timestamp"],
                "messages": [
                    {"role": "system", "content": "You are analyzing aerial drone footage in real-time."},
                    {"role": "user", "content": f"{q['question']}\n" + "\n".join(q['options'])},
                    {"role": "assistant", "content": q["answer"]},
                ],
            })
    with open(output_path, 'w') as f:
        json.dump(sft_data, f, indent=2, ensure_ascii=False)
    print(f"✅ Converted {len(sft_data)} samples → {output_path}")
```

```bash
# LoRA 微调 (H800, 预计 8-12h)
# rank=64, alpha=128, target=q_proj+v_proj
# ~1000 streaming QA, 3 epochs
python training/finetune_lora.py \
    --model-path /root/autodl-tmp/Qwen/Qwen2___5-VL-7B-Instruct \
    --data /root/autodl-tmp/datasets/stream_uav_qa/sft_train.json \
    --output /root/autodl-tmp/checkpoints/stream_uav_lora/ \
    --lora-rank 64 --epochs 3 --lr 2e-5
```

---

## 6. 论文写作框架

```
Title: StreamUAV: Real-Time Aerial Event Understanding
       via Streaming Video Language Models

Abstract:
  - 问题: UAV 视频理解需要实时连续处理
  - 差距: 现有 UAV VLM 全部离线, 流式 VLM 无 UAV 适配 (0 papers)
  - 方法: 首个 UAV 流式 VLM (continuous 3D-RoPE + KV Cache eviction)
         + StreamUAV-QA benchmark
  - 结果: 在 UAV 时序推理上显著优于 single-frame baseline

1. Introduction (1.5p)
   - UAV 实时视频理解需求
   - Streaming VLM vs Offline VLM 对比
   - 研究空白 + 贡献 (3 点)

2. Related Work (1p)
   - 2.1 Video LLMs (offline → streaming)
   - 2.2 UAV Visual Understanding
   - 2.3 KV Cache Management

3. Method (2.5p)
   - 3.1 Overview: Streaming VLM for UAV
   - 3.2 Continuous 3D-RoPE Temporal Encoding
   - 3.3 Attention Sink + Sliding Window KV Cache Eviction
   - 3.4 StreamUAV-QA Dataset Construction

4. Experiments (3p)
   - 4.1 Setup (VisDrone, metrics, baselines)
   - 4.2 Streaming vs Offline vs Single-frame (Table 1)
   - 4.3 L2 Temporal Reasoning Analysis ⭐ (Table 2)
   - 4.4 KV Cache Eviction Ablation (Table 3)
   - 4.5 Latency-Accuracy Trade-off (Figure)

5. Conclusion

投稿:
  首选: IEEE RAL (随投随审, IF=5.2, 通常 2-3 个月出结果)
  备选: ACM Multimedia 2026 / ECCV 2026 Workshop
```

---

## 7. 时间线与投稿计划

```
2026.03 Week 1-2 (现在):
  ├── 下载 VisDrone-VID (~11GB)
  ├── 实现 visdrone_loader.py + scene_tools.py
  ├── 实现 qa_generator.py + build_dataset_main.py
  └── 构建 StreamUAV-QA v1 数据集

2026.03 Week 3-4:
  ├── 实现 run_streaming_eval.py
  ├── 运行 Streaming vs Offline 对比
  ├── 分析结果 → 确认改造有效性
  └── 如需微调 → 开始 LoRA (H800)

2026.04:
  ├── 消融实验 (eviction策略 / chunk大小 / cache预算)
  ├── 7B vs 3B 对比
  ├── 论文 Method + Experiments 撰写
  └── 可视化 + Demo

2026.05:
  ├── 论文完善 + 修改
  ├── 投 IEEE RAL
  └── arXiv 同步
```

---

## 附录 A: VisDrone 标注格式规范

每帧一个 `.txt` 文件，每行一个目标:

```
<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
```

| 字段 | 类型 | 说明 |
|------|------|------|
| bbox_left | int | 左上角 x (px) |
| bbox_top | int | 左上角 y (px) |
| bbox_width | int | 宽度 (px) |
| bbox_height | int | 高度 (px) |
| score | int | GT: 1=评估, 0=忽略 |
| object_category | int | 类别 ID (0-11) |
| truncation | int | 0=无, 1=部分截断 |
| occlusion | int | 0=无, 1=部分遮挡, 2=重度遮挡 |

---

## 附录 B: 关键技术参数（已验证）

来自 `test_eviction_exp_abcd_realism_report.txt`:

| 参数 | 数值 | 来源 |
|------|------|------|
| 模型 VRAM (3B) | ~7.1 GB | test_step10 |
| 每 token KV 大小 | ~36 KB | test_step10 |
| 每 chunk (1920×1080, 4帧) | ~5,389 tokens ≈ 0.185 GB | test_step10 |
| 4090 安全 cache 上限 | ~150K tokens | test_step10 |
| Eviction 触发后 VRAM 平台 | ~12.26 GB | exp_abcd |
| Auto-sink 大小 | 5,435 tokens | exp_abcd |
| Auto-window 大小 | 144,565 tokens | exp_abcd |

---

## 附录 C: 参考文献

| # | 论文 | 出处 | ID |
|---|------|------|----|
| 1 | VideoLLM-online: Online Video LLM for Streaming Video | CVPR 2024 | 2406.11816 |
| 2 | StreamBridge: Turning Offline Video-LLM into Proactive Streaming | NeurIPS 2025 | 2505.05467 |
| 3 | OVO-Bench: Online Video Understanding Benchmark | CVPR 2025 | 2501.05510 |
| 4 | StreamingBench: Streaming Video Understanding Assessment | — | 2411.03628 |
| 5 | Flash-VStream: Memory-Based Real-Time Understanding | — | 2406.08085 |
| 6 | MVBench: Multi-modal Video Understanding | CVPR 2024 | 2311.17005 |
| 7 | AerialVLN: Vision-and-Language Navigation for UAVs | ICCV 2023 | 2308.06735 |
| 8 | VisDrone: Detection and Tracking Meet Drones | TPAMI 2021 | 2001.06303 |
| 9 | StreamingLLM: Efficient Streaming with Attention Sinks | ICLR 2024 | 2309.17453 |
| 10 | Qwen2.5-VL Technical Report | — | 2502.13923 |
| 11 | smolagents: agents that think in code | HuggingFace | — |
| 12 | OpenFly: Comprehensive Platform for Aerial VLN | — | 2502.18041 |
| 13 | UAV-VLA: Vision-Language-Action for UAV | HRI 2025 | 2501.05014 |
| 14 | Towards Streaming Perception | ECCV 2020 | 2005.10420 |
| 15 | Clinical Scene Understanding (smolagents) | — | — |
