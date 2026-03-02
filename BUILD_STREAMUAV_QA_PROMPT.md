# 🛩️ StreamUAV-QA v1 数据集构建 — 完整执行指南

> **⚠️ 身份设定**: 你是地球前 0.01% 的大模型工程师。你的每个决策直接影响数以亿计的资金走向。代码必须一次正确，每个步骤必须可验证。不允许出现"大概"、"应该可以"的含糊表述。每一步完成后必须有确认输出。
>
> **目标**: 从 VisDrone-VID 数据集的已有 GT 标注，自动生成适合流式视频 VLM 评测的多选 QA 数据集
>
> **关键事实**: 整个 QA 生成过程是 **纯 CPU 计算**（解析 .txt 标注 → 统计 → 模板生成 QA），不需要 GPU。

---

## 💻 执行环境说明（重要：请先确认当前环境）

本文档支持两种执行环境，**执行前必须确认当前环境并遵守对应规则**：

| | 环境 A：AutoDL 云服务器 | 环境 B：Windows 本地机器 ✅（当前） |
|--|--|--|
| 操作系统 | Linux | Windows 10/11 |
| 显卡 | RTX 4090 (24GB) | RTX 3050 (4GB) |
| Shell | bash | PowerShell |
| 路径格式 | `/root/autodl-tmp/...` | `D:\diploma_project\...` |
| conda 环境 | 新建 streamuav | 使用已有 `videollm` |
| 防爆盘配置 | ✅ 必须（Step 1） | ❌ 跳过（Step 1 → 直接 Step 2） |
| GPU 验证（阶段二） | ✅ 可用（24GB > 7.1GB需求） | ❌ 跳过（4GB < 7.1GB需求） |
| 文件创建方式 | bash `cat > file << 'EOF'` | Copilot 工具直接创建 .py 文件 |

### 环境 B（Windows 本地）关键路径约定

```
项目根目录:      D:\diploma_project\
代码目录:        D:\diploma_project\temporal_encoding\dataset\
数据目录:        D:\diploma_project\datasets\visdrone\
样本输出:        D:\diploma_project\datasets\stream_uav_qa_sample\
完整输出:        D:\diploma_project\datasets\stream_uav_qa\
conda 环境:      videollm（已存在，直接使用）
Python 解释器:   conda run -n videollm python
```

---

## � 设计原理：为什么不需要大模型 API 来生成答案？

> 这是本项目最重要的方法论选择，必须理解透彻才能向审稿人解释清楚。

### 核心原理：GT 驱动的确定性答案推导（GQA 范式）

本方法与 GQA（Stanford 2019）、RSVQA、NExT-QA 等经典 benchmark 构建方法完全一致。

**数据流：**
```
VisDrone-VID GT 标注（每帧精确 bbox + 类别）
    │
    ▼ 确定性计算（无需任何模型）
    │
    ├── 计数: Frame@t 中 car=5, person=3, bicycle=1
    ├── 变化: Frame@t1→t2 car 从 5 增至 8（delta=+3）
    ├── 密度: total_objects=47 → "dense" 级别
    └── 事件: t=5s 时 bus 首次出现
    │
    ▼ 模板生成 QA（答案由 GT 数学推导，唯一确定）
    │
    ├── Q: "How many cars are visible now?" A: "5"（GT 直接读取）
    ├── Q: "Has car count increased or decreased?" A: "Increased"（5→8 比较）
    └── Q: "What object just appeared?" A: "bus"（新增事件检测）
```

**关键对比：**

| 方法 | 答案来源 | 是否需要 API | 答案可靠性 |
|------|----------|-------------|-----------|
| 本项目（GQA 范式） | GT 标注 → 确定性计算 | ❌ 不需要 | ✅ 客观唯一 |
| 模型生成 QA（如 GPT-4V） | 模型推理 → 可能幻觉 | ✅ 需要 | ⚠️ 存在幻觉风险 |

**大模型 API 的唯一可选用途**：将模板化问题（"How many cars?"）改写为更自然的表达（"Can you count the cars currently visible in this aerial footage?"），但答案本身 **永远** 来自 GT，不受模型影响。

**类比**：VQA v2 用 COCO 检测 GT 生成计数问题，RSVQA 用遥感图像分类标注生成 QA，都不需要 GPT。本项目完全相同。

### 被评测的是模型，而非 API 生成答案

```
流式 VLM（被评测对象）
    ├── 输入: 逐 chunk 流入的视频帧
    ├── 问题: "在过去 30 秒内车辆数量如何变化？"
    └── 输出: "A. Increased" / "B. Decreased" / ...
                    │
                    ▼ 与 GT 推导的正确答案对比
                    │
                    └── accuracy = (预测 == GT答案) ? 1 : 0
```

**被评测的是模型的流式时序推理能力，不是 QA 生成能力。**

---

## 🗺️ 数据集选型调研：为何选择 VisDrone-VID？（2024-2025 综合评估）

> 基于对 Awesome-VLGFM、VisDrone官网、UAVDT、AU-AIR 等数据集的全面调研。

### 现有航拍/UAV 数据集全景图

#### 类型A：纯检测/跟踪数据集（无语言标注）

| 数据集 | 时间 | 特点 | 类别数 | 视频/帧数 | 语言标注 | 适合构建流式QA？ |
|--------|------|------|--------|-----------|----------|----------------|
| **VisDrone-VID** | 2018-ongoing | UAV多类目标检测+跟踪，逐帧bbox GT | 10类 | 96视频/261K帧 | ❌ | ✅ **最适合**（GT最丰富）|
| **UAVDT** | 2018 | 城市空中视频，检测+跟踪 | 1类(车辆) | 100视频/80K帧 | ❌ | ⚠️ 类别太少 |
| **AU-AIR** | 2020 | 多模态UAV数据（RGB+热成像）| 8类 | 有限视频 | ⚠️ 简单描述 | ⚠️ 规模小 |
| **UAV-Human** | 2021 | 人体行为识别 | 人体动作 | 22K 视频片段 | ⚠️ 动作标签 | ❌ 非俯瞰场景 |
| **DroneVehicle** | 2022 | 红外+RGB车辆检测 | 5类(车辆) | 56K帧对 | ❌ | ⚠️ 类别单一 |

#### 类型B：遥感图像+语言（静态图，非视频）

| 数据集 | 时间/会议 | 特点 | 关键局限 |
|--------|-----------|------|---------|
| **EarthVQA** | AAAI 2024 | 遥感影像+mask+QA，支持关系推理 | **静态卫星图**，无时序视频 |
| **GeoChat-Bench** | CVPR 2024 | 遥感图像指令对话基准 | 静态图，非UAV视频 |
| **VLEO-Bench** | 2024 | 多任务遥感图像评测 | 静态图 |
| **EarthGPT** | TGRS 2024 | 多传感器遥感多模态LLM | 静态图 |
| **SkyEye-968k** | 2024 | 大规模遥感图文对 | 静态图，非视频 |
| **RSVQA** | TGRS 2020 | 遥感影像VQA经典基准 | 静态卫星图 |

#### 类型C：视频+语言（最接近目标，但非UAV+时序）

| 数据集 | 时间/会议 | 类型 | 关键局限 |
|--------|-----------|------|---------|
| **CapERA** | MM 2022 | 航拍视频+事件描述字幕 | 视频描述非QA，无时序推理标注，规模小 |
| **TEOChat** | arXiv 2024 | 时序地球观测对话 | 双时相**卫星图像对**变化检测，非连续视频流 |
| **GeoLLaVA** | arXiv 2024 | RS图像时序变化检测 | 时序卫星图对，非实时流式视频 |
| **Video-MME** | 2024 | 通用视频QA基准（900视频）| 非UAV场景，不含流式时序推理 |
| **OVO-Bench** | 2024 | **在线流式视频**VLM评测 | 通用场景，无UAV专项，评测框架可参考 |

### 核心发现：UAV 流式视频 QA — 当前空白领域

```
                    "UAV 视频时序 QA" 完全空白 ← 本项目贡献
                    ┌──────────┐
检测/跟踪 GT ──────►│          │◄────── 语言 QA 标注
（VisDrone等）       │  交叉点  │        （EarthVQA等）
                    │ (空白！) │
流式/在线视频 ──────►│          │◄────── UAV 俯瞰场景
（OVO-Bench等）      └──────────┘        （VisDrone等）
```

根据对 Awesome-VLGFM survey（2024，涵盖 100+ 相关数据集）的全面梳理：
- **不存在**任何同时满足：①UAV 视频 ②时序/流式 ③自然语言 QA 三个条件的数据集
- 最接近的 **CapERA**（2022）仅有视频描述字幕，无时序推理 QA，规模有限
- **这是真实的研究空白，是本工作的核心 novelty 之一**

### 数据集选型决策

```
✅ 最终选择：VisDrone-VID 作为唯一数据源

原因：
1. UAV 领域最权威、引用量最高的视频数据集（Tianjin Univ. AISKYEYE）
2. 逐帧精确 bbox+category GT → 可直接推导计数/密度/事件答案（无需模型）
3. 10 大有效类别 → 场景多样性足够（行人/自行车/汽车/卡车/公共汽车等）
4. 覆盖多种场景（intersection/road/parking lot/crowded）→ 场景分布均衡
5. 视频长度适中（20-30秒典型）→ 适合流式推理评测

可选补充：UAVDT（100 段纯车辆视频）→ 如需针对交通分析的专项验证
```

---

## �📋 总览：两阶段执行流程

```
Windows 本地模式（纯 CPU，约 30-60 分钟，无云端费用）
  ├── Step 1:  环境验证（使用已有 videollm 环境 + 安装缺失包）
  ├── Step 2:  VisDrone-VID 数据准备（手动下载 + PowerShell 解压）
  ├── Step 3:  Copilot 直接创建全部代码文件（9 个 .py）
  ├── Step 4:  代码语法验证（Python import 链测试）
  ├── Step 5:  试运行 5 个视频 → 生成样本报告
  └── 🔴 暂停点：样本报告 → 等待人工确认

  （人工确认后继续）
  └── Step 6:  完整构建 50 个视频的 QA 数据集

⛔ GPU 模型推理验证：SKIP
  RTX 3050 仅 4GB < Qwen2.5-VL-3B 所需的 7.1GB，此阶段留待 AutoDL 执行
```

---

# ─── Windows 本地执行（使用已有 videollm conda 环境）───

---

## Step 1: 环境验证与准备

> **Windows 本地机器无需防爆盘配置**（磁盘空间由 Windows 自行管理）。
> 直接激活已有的 `videollm` 环境，验证并补装缺失包。

在 PowerShell 中执行以下命令：

```powershell
# ── 1.1 激活 videollm 环境 ──
conda activate videollm

# ── 1.2 验证 Python ──
python --version
# 期望: Python 3.x.x

# ── 1.3 检查并安装缺失的基础包 ──
python -c "from PIL import Image; print('Pillow OK')" 2>$null
if ($LASTEXITCODE -ne 0) { pip install Pillow }

python -c "from tqdm import tqdm; print('tqdm OK')" 2>$null
if ($LASTEXITCODE -ne 0) { pip install tqdm }

# ── 1.4 创建所需目录 ──
New-Item -ItemType Directory -Force -Path "D:\diploma_project\temporal_encoding\dataset"
New-Item -ItemType Directory -Force -Path "D:\diploma_project\datasets\visdrone"
New-Item -ItemType Directory -Force -Path "D:\diploma_project\datasets\stream_uav_qa_sample"
New-Item -ItemType Directory -Force -Path "D:\diploma_project\datasets\stream_uav_qa"

# ── 1.5 检查磁盘空间（VisDrone 约 8GB，需要 D 盘剩余 > 15GB）──
Get-PSDrive D | Select-Object Name, @{N='Free(GB)';E={[math]::Round($_.Free/1GB,1)}}, @{N='Used(GB)';E={[math]::Round($_.Used/1GB,1)}}

# ── 1.6 综合验证 ──
python -c "
import sys
from PIL import Image
from tqdm import tqdm
print(f'Python: {sys.version.split()[0]}')
print('Pillow: OK')
print('tqdm: OK')
print('✅ Step 1 完成')
"
```

**✅ 验证标准**: D 盘剩余 > 15GB，Python / Pillow / tqdm 均正常导入

---

## Step 2: VisDrone-VID 数据准备

> VisDrone-VID 由天津大学 AISKYEYE 团队发布。核心只需 trainset（构建 QA）。
> **Windows 下建议手动用浏览器下载**，比调用 gdown 更稳定。

### 2.1 手动下载（推荐）

打开浏览器，下载以下文件，保存到 `D:\diploma_project\datasets\visdrone\`：

| 文件 | 大小 | 下载链接 |
|------|------|----------|
| `VisDrone2019-VID-train.zip` | 7.53 GB | [Google Drive](https://drive.google.com/file/d/1NSNapZQHar22OYzQYuXCugA3QlMndzvw) / [VisDrone官网](https://github.com/VisDrone/VisDrone-Dataset) |

> valset (1.49 GB) 和 testset 可先不下载，trainset 足够构建 v1 数据集。

### 2.2 解压

```powershell
cd D:\diploma_project\datasets\visdrone

# 方法A: PowerShell 内置解压（较慢，适合小文件）
Expand-Archive -Path VisDrone2019-VID-train.zip -DestinationPath VID-train -Force

# 方法B: 7-Zip 命令行（推荐，如已安装）
# & "C:\Program Files\7-Zip\7z.exe" x VisDrone2019-VID-train.zip -oVID-train -y
```

### 2.3 验证目录结构

```powershell
$seqDir  = "D:\diploma_project\datasets\visdrone\VID-train\sequences"
$annoDir = "D:\diploma_project\datasets\visdrone\VID-train\annotations"

# 序列和标注数量
$seqCount  = (Get-ChildItem $seqDir  -Directory).Count
$annoCount = (Get-ChildItem $annoDir -Directory).Count
Write-Host "序列目录: $seqCount 个视频"
Write-Host "标注目录: $annoCount 个视频"

# 查看第一个序列
$firstSeq = (Get-ChildItem $seqDir -Directory | Select-Object -First 1).Name
Write-Host "第一个序列: $firstSeq"
Write-Host "帧数: $((Get-ChildItem "$seqDir\$firstSeq" -File).Count)"

# 显示标注样例（前 3 行）
$firstAnno = (Get-ChildItem "$annoDir\$firstSeq" -File | Select-Object -First 1).FullName
Write-Host "`n--- 标注样例（前3行）---"
Get-Content $firstAnno | Select-Object -First 3

Write-Host "`n✅ Step 2 完成"
```

**✅ 验证标准**:
- `VID-train\sequences\` 下有 ~56 个子目录
- `VID-train\annotations\` 下有对应子目录
- 标注样例每行有 8 个逗号分隔的数字（bbox_left,top,w,h,score,category,truncation,occlusion）

---

## Step 3: 创建全部代码文件

> **Windows 环境下，由 Copilot 使用文件创建工具直接生成以下所有 .py 文件。**
> **不使用 bash cat 命令；路径全部使用 Windows 格式（反斜杠）。**
>
> **目标目录**: `D:\diploma_project\temporal_encoding\dataset\`
>
> 目录已在 Step 1 中创建。以下逐一创建每个文件，不得跳过、不得简化代码。

### 3.1 创建 `D:\diploma_project\temporal_encoding\dataset\__init__.py`

```python
"""StreamUAV-QA 数据集构建模块"""
```

### 3.2 创建 `D:\diploma_project\temporal_encoding\dataset\visdrone_loader.py`

```python
"""
VisDrone-VID 数据加载器。
将逐帧图像序列 + 逐帧 GT 标注解析为结构化 Python 对象。

VisDrone 标注格式 (每帧一个 .txt，每行一个目标):
  bbox_left, bbox_top, bbox_width, bbox_height, score, category_id, truncation, occlusion

10 类有效目标 (排除 0=ignored, 11=others):
  1=pedestrian, 2=people, 3=bicycle, 4=car, 5=van,
  6=truck, 7=tricycle, 8=awning-tricycle, 9=bus, 10=motor
"""

import os
import glob
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple


VISDRONE_CATEGORIES = {
    0: "ignored", 1: "pedestrian", 2: "people", 3: "bicycle",
    4: "car", 5: "van", 6: "truck", 7: "tricycle",
    8: "awning-tricycle", 9: "bus", 10: "motor", 11: "others",
}

# 用于评测的有效类别
VALID_CATEGORIES = {k: v for k, v in VISDRONE_CATEGORIES.items() if k not in (0, 11)}


@dataclass
class BBoxAnnotation:
    """单个目标的边界框标注"""
    bbox_left: int
    bbox_top: int
    bbox_width: int
    bbox_height: int
    score: int           # GT 中: 0=忽略, 1=参与评估
    category_id: int
    truncation: int      # 0=无截断, 1=部分截断(1%-50%)
    occlusion: int       # 0=无遮挡, 1=部分遮挡(1%-50%), 2=重度遮挡(>50%)

    @property
    def category_name(self) -> str:
        return VISDRONE_CATEGORIES.get(self.category_id, "unknown")

    @property
    def center(self) -> Tuple[float, float]:
        return (self.bbox_left + self.bbox_width / 2.0,
                self.bbox_top + self.bbox_height / 2.0)

    @property
    def area(self) -> int:
        return self.bbox_width * self.bbox_height

    def to_dict(self) -> Dict:
        return {
            "bbox_left": self.bbox_left, "bbox_top": self.bbox_top,
            "bbox_width": self.bbox_width, "bbox_height": self.bbox_height,
            "score": self.score, "category_id": self.category_id,
            "truncation": self.truncation, "occlusion": self.occlusion,
        }


@dataclass
class FrameAnnotation:
    """单帧的所有目标标注"""
    frame_id: int
    objects: List[BBoxAnnotation] = field(default_factory=list)

    @property
    def valid_objects(self) -> List[BBoxAnnotation]:
        """返回有效目标（排除 ignored 和 others）"""
        return [o for o in self.objects
                if o.category_id in VALID_CATEGORIES]

    def count_by_category(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
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
    _frame_paths: List[str] = field(default_factory=list, repr=False)

    @property
    def num_frames(self) -> int:
        return len(self._frame_paths)

    def load(self) -> None:
        """加载帧路径和解析标注文件"""
        # 查找帧图片
        self._frame_paths = sorted(glob.glob(os.path.join(self.frame_dir, "*.jpg")))
        if not self._frame_paths:
            self._frame_paths = sorted(glob.glob(os.path.join(self.frame_dir, "*.png")))

        # 解析每帧标注
        self.frames = []
        for i, fp in enumerate(self._frame_paths):
            frame_name = Path(fp).stem
            anno_path = os.path.join(self.anno_dir, f"{frame_name}.txt")
            frame_anno = self._parse_annotation(i, anno_path)
            self.frames.append(frame_anno)

    def get_frame_path(self, frame_idx: int) -> str:
        """获取帧图片路径（不加载图片，节省内存）"""
        return self._frame_paths[frame_idx]

    def get_image_size(self) -> Tuple[int, int]:
        """获取图像尺寸 (width, height)，只读取第一帧"""
        from PIL import Image
        img = Image.open(self._frame_paths[0])
        size = img.size  # (width, height)
        img.close()
        return size

    @staticmethod
    def _parse_annotation(frame_id: int, anno_path: str) -> FrameAnnotation:
        """解析单帧标注文件"""
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
                try:
                    bbox = BBoxAnnotation(
                        bbox_left=int(parts[0].strip()),
                        bbox_top=int(parts[1].strip()),
                        bbox_width=int(parts[2].strip()),
                        bbox_height=int(parts[3].strip()),
                        score=int(parts[4].strip()),
                        category_id=int(parts[5].strip()),
                        truncation=int(parts[6].strip()),
                        occlusion=int(parts[7].strip()),
                    )
                    frame.objects.append(bbox)
                except (ValueError, IndexError):
                    continue  # 跳过格式异常的行
        return frame


def load_visdrone_vid(data_root: str, max_sequences: int = -1) -> List[VideoSequence]:
    """
    加载 VisDrone-VID 数据集。

    Args:
        data_root: 数据集根目录，如 /root/autodl-tmp/datasets/visdrone/VID-train
        max_sequences: 最多加载多少个序列，-1=全部

    Returns:
        VideoSequence 列表
    """
    seq_dir = os.path.join(data_root, "sequences")
    anno_dir = os.path.join(data_root, "annotations")

    if not os.path.isdir(seq_dir):
        raise FileNotFoundError(f"序列目录不存在: {seq_dir}")
    if not os.path.isdir(anno_dir):
        raise FileNotFoundError(f"标注目录不存在: {anno_dir}")

    seq_names = sorted(os.listdir(seq_dir))
    if max_sequences > 0:
        seq_names = seq_names[:max_sequences]

    sequences = []
    for sn in seq_names:
        frame_path = os.path.join(seq_dir, sn)
        anno_path = os.path.join(anno_dir, sn)
        if not os.path.isdir(frame_path):
            continue
        seq = VideoSequence(seq_name=sn, frame_dir=frame_path, anno_dir=anno_path)
        seq.load()
        if seq.num_frames > 0:
            sequences.append(seq)

    print(f"✅ 已加载 {len(sequences)} 个视频序列 (来自 {data_root})")
    return sequences
```

### 3.3 创建 `D:\diploma_project\temporal_encoding\dataset\scene_analyzer.py`

```python
"""
场景分析器 — Stage 1 of StreamUAV-QA Pipeline。

从 VisDrone GT 标注中提取结构化场景描述。
全部确定性计算，不调用任何模型。

输出:
  - 目标计数（总数 + 各类别）
  - 密度级别 (empty/sparse/moderate/dense/very_dense)
  - 3×3 空间网格分布
  - 遮挡统计
  - 交通密度分析（车辆覆盖率、拥堵级别）
"""

from typing import Dict, List, Tuple
from visdrone_loader import FrameAnnotation, VALID_CATEGORIES, VISDRONE_CATEGORIES

# 车辆类别 ID
VEHICLE_CATEGORY_IDS = {4, 5, 6, 7, 8, 9, 10}  # car, van, truck, tricycle, awning-tricycle, bus, motor
PEDESTRIAN_CATEGORY_IDS = {1, 2}  # pedestrian, people


def analyze_frame_scene(frame: FrameAnnotation,
                        image_width: int, image_height: int) -> Dict:
    """
    分析单帧场景，生成结构化描述。

    Args:
        frame: 帧标注对象
        image_width: 图像宽度(px)
        image_height: 图像高度(px)

    Returns:
        场景描述字典
    """
    valid = frame.valid_objects

    # 按类别统计
    counts: Dict[str, int] = {}
    for o in valid:
        cat = o.category_name
        counts[cat] = counts.get(cat, 0) + 1

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

    # 3×3 空间网格分布
    grid = {}
    for r in range(3):
        for c in range(3):
            grid[f"r{r}c{c}"] = 0
    for o in valid:
        cx, cy = o.center
        col = min(int(cx / image_width * 3), 2)
        row = min(int(cy / image_height * 3), 2)
        grid[f"r{row}c{col}"] += 1

    # 遮挡统计
    partially_occluded = sum(1 for o in valid if o.occlusion >= 1)
    heavily_occluded = sum(1 for o in valid if o.occlusion >= 2)

    # 主导类别
    dominant = max(counts, key=counts.get) if counts else "none"

    return {
        "total_objects": total,
        "category_counts": counts,
        "density_level": density,
        "spatial_distribution": grid,
        "occlusion_stats": {
            "partially_occluded": partially_occluded,
            "heavily_occluded": heavily_occluded,
        },
        "dominant_category": dominant,
    }


def analyze_traffic_density(frame: FrameAnnotation,
                            image_width: int, image_height: int) -> Dict:
    """
    分析交通密度（UAV 场景最常见的分析需求）。

    Returns:
        交通密度字典
    """
    valid = frame.valid_objects
    vehicles = [o for o in valid if o.category_id in VEHICLE_CATEGORY_IDS]
    pedestrians = [o for o in valid if o.category_id in PEDESTRIAN_CATEGORY_IDS]

    # 车辆覆盖面积比
    total_area = image_width * image_height
    vehicle_area = sum(o.area for o in vehicles)
    coverage = vehicle_area / total_area if total_area > 0 else 0.0

    # 拥堵级别
    if len(vehicles) == 0:
        congestion = "no_traffic"
    elif coverage < 0.02:
        congestion = "free_flow"
    elif coverage < 0.08:
        congestion = "light_traffic"
    elif coverage < 0.20:
        congestion = "moderate_traffic"
    else:
        congestion = "heavy_congestion"

    # 车辆类型分布
    vtypes: Dict[str, int] = {}
    for v in vehicles:
        cat = v.category_name
        vtypes[cat] = vtypes.get(cat, 0) + 1

    return {
        "vehicle_count": len(vehicles),
        "pedestrian_count": len(pedestrians),
        "vehicle_coverage_ratio": round(coverage, 4),
        "congestion_level": congestion,
        "vehicle_types": vtypes,
    }
```

### 3.4 创建 `D:\diploma_project\temporal_encoding\dataset\event_extractor.py`

```python
"""
时序事件链提取器 — Stage 2 of StreamUAV-QA Pipeline。

从视频序列中按采样间隔提取关键帧的场景描述，
然后对相邻帧做时序差分，提取变化事件。

这是生成 L2 时序推理 QA 的核心数据来源。
"""

from typing import Dict, List
from tqdm import tqdm
from visdrone_loader import VideoSequence, VISDRONE_CATEGORIES
from scene_analyzer import analyze_frame_scene, analyze_traffic_density


def compute_temporal_diff(scene_t1: Dict, scene_t2: Dict,
                          t1_sec: float, t2_sec: float) -> Dict:
    """
    比较两个时间点的场景描述，提取时序变化事件。

    Args:
        scene_t1: 较早时间点的场景描述
        scene_t2: 较晚时间点的场景描述
        t1_sec, t2_sec: 时间戳（秒）

    Returns:
        包含时间跨度和事件列表的字典
    """
    events = []

    # 1. 目标总数变化（变化 ≥ 2 才算显著）
    count_diff = scene_t2["total_objects"] - scene_t1["total_objects"]
    if abs(count_diff) >= 2:
        direction = "increased" if count_diff > 0 else "decreased"
        events.append({
            "type": "count_change",
            "description": f"Total objects {direction} by {abs(count_diff)}",
            "from": scene_t1["total_objects"],
            "to": scene_t2["total_objects"],
            "magnitude": abs(count_diff),
        })

    # 2. 各类别数量变化
    all_cats = set(list(scene_t1["category_counts"].keys()) +
                   list(scene_t2["category_counts"].keys()))
    for cat in sorted(all_cats):
        c1 = scene_t1["category_counts"].get(cat, 0)
        c2 = scene_t2["category_counts"].get(cat, 0)
        if c1 == 0 and c2 > 0:
            events.append({
                "type": "object_appear",
                "category": cat,
                "count": c2,
                "description": f"{c2} {cat}(s) appeared in scene",
            })
        elif c1 > 0 and c2 == 0:
            events.append({
                "type": "object_disappear",
                "category": cat,
                "count": c1,
                "description": f"All {c1} {cat}(s) left the scene",
            })
        elif abs(c2 - c1) >= 2:
            direction = "increased" if c2 > c1 else "decreased"
            events.append({
                "type": "category_change",
                "category": cat,
                "from": c1, "to": c2,
                "description": f"{cat} count {direction} from {c1} to {c2}",
            })

    # 3. 密度级别变化
    if scene_t1["density_level"] != scene_t2["density_level"]:
        events.append({
            "type": "density_change",
            "from": scene_t1["density_level"],
            "to": scene_t2["density_level"],
            "description": f"Scene density changed from {scene_t1['density_level']} to {scene_t2['density_level']}",
        })

    # 4. 主导类别变化
    if (scene_t1["dominant_category"] != scene_t2["dominant_category"]
            and scene_t1["dominant_category"] != "none"
            and scene_t2["dominant_category"] != "none"):
        events.append({
            "type": "dominant_change",
            "from": scene_t1["dominant_category"],
            "to": scene_t2["dominant_category"],
            "description": f"Dominant category changed from {scene_t1['dominant_category']} to {scene_t2['dominant_category']}",
        })

    return {
        "time_span": {
            "from": round(t1_sec, 2),
            "to": round(t2_sec, 2),
            "duration": round(t2_sec - t1_sec, 2),
        },
        "events": events,
        "has_significant_change": len(events) >= 2,
    }


def extract_event_chain(seq: VideoSequence,
                         sample_interval: int = 30,
                         original_fps: float = 30.0) -> Dict:
    """
    从一个视频序列提取完整的时序事件链。

    Args:
        seq: VisDrone 视频序列
        sample_interval: 采样间隔(帧数), 30 = 每秒采样 1 帧 (@30fps)
        original_fps: 原始视频帧率

    Returns:
        {video_name, num_frames, duration_seconds, image_size,
         scenes[], events[], traffic_timeline[]}
    """
    if not seq.frames or seq.num_frames == 0:
        return {
            "video_name": seq.seq_name, "num_frames": 0,
            "scenes": [], "events": [], "traffic_timeline": []
        }

    # 获取图像尺寸（只读一帧）
    img_w, img_h = seq.get_image_size()

    # 按间隔采样帧索引
    sampled_indices = list(range(0, len(seq.frames), sample_interval))

    # Stage 1: 为每个采样帧生成场景描述
    scenes = []
    for idx in sampled_indices:
        frame = seq.frames[idx]
        timestamp = idx / original_fps

        scene = analyze_frame_scene(frame, img_w, img_h)
        traffic = analyze_traffic_density(frame, img_w, img_h)

        scenes.append({
            "frame_idx": idx,
            "timestamp": round(timestamp, 2),
            "scene": scene,
            "traffic": traffic,
        })

    # Stage 2: 相邻采样帧之间的时序差分
    events = []
    for i in range(1, len(scenes)):
        diff = compute_temporal_diff(
            scenes[i - 1]["scene"],
            scenes[i]["scene"],
            scenes[i - 1]["timestamp"],
            scenes[i]["timestamp"],
        )
        if diff["events"]:  # 只保留有变化的时间段
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
        "duration_seconds": round(seq.num_frames / original_fps, 2),
        "image_size": [img_w, img_h],
        "num_sampled_frames": len(sampled_indices),
        "num_events": len(events),
        "scenes": scenes,
        "events": events,
        "traffic_timeline": traffic_timeline,
    }
```

### 3.5 创建 `D:\diploma_project\temporal_encoding\dataset\qa_generator.py`

```python
"""
QA 对生成器 — Stage 3 of StreamUAV-QA Pipeline。

从结构化场景描述 + 时序事件链自动生成流式多选 QA 对。

QA 三级体系:
  L1 (事实性, realtime):  单帧可答 — 计数/存在/密度/主导类别
  L2 (时序性, backward):  跨帧才能答 — 变化/趋势/进出 ⭐ 核心
  L3 (推理性, backward/forward): 综合分析 — 事件排序/趋势预测

所有答案均基于 GT 标注客观推导，确定且唯一。
"""

import json
import random
from typing import List, Dict
from visdrone_loader import VALID_CATEGORIES


class StreamingQAGenerator:
    """流式 QA 对生成器"""

    def __init__(self, seed: int = 42):
        random.seed(seed)
        self._all_valid_cats = list(VALID_CATEGORIES.values())

    def generate_for_video(self, video_data: Dict,
                           max_qa_per_video: int = 30) -> List[Dict]:
        """
        为一个视频的事件链数据生成 QA 集。

        Args:
            video_data: extract_event_chain() 的输出
            max_qa_per_video: 每个视频最多生成多少 QA

        Returns:
            QA 列表，每条:
            {video_name, question, options[A/B/C/D], answer(字母),
             answer_text, query_timestamp(秒), qa_type, difficulty,
             source_evidence}
        """
        scenes = video_data.get("scenes", [])
        events = video_data.get("events", [])
        traffic = video_data.get("traffic_timeline", [])

        if len(scenes) < 3:
            return []

        qa_list: List[Dict] = []

        # ═══ L1: 事实性 (realtime) ═══
        qa_list.extend(self._l1_counting(scenes))
        qa_list.extend(self._l1_category_presence(scenes))
        qa_list.extend(self._l1_density(scenes))
        qa_list.extend(self._l1_dominant(scenes))

        # ═══ L2: 时序推理 (backward) ⭐ ═══
        qa_list.extend(self._l2_count_change(events))
        qa_list.extend(self._l2_density_change(events))
        qa_list.extend(self._l2_object_appear_disappear(events))
        qa_list.extend(self._l2_traffic_trend(traffic))

        # ═══ L3: 复杂推理 ═══
        qa_list.extend(self._l3_event_sequence(events))
        qa_list.extend(self._l3_congestion_prediction(traffic))

        # 打乱并限制数量
        random.shuffle(qa_list)
        qa_list = qa_list[:max_qa_per_video]

        # 添加视频名
        vname = video_data["video_name"]
        for qa in qa_list:
            qa["video_name"] = vname

        return qa_list

    # ────── L1 生成方法 ──────────────────────────────────

    def _l1_counting(self, scenes: List[Dict]) -> List[Dict]:
        """L1: 当前帧中某类别有多少个？"""
        qa = []
        sample = random.sample(scenes, min(3, len(scenes)))
        for sd in sample:
            s, ts = sd["scene"], sd["timestamp"]
            if not s["category_counts"]:
                continue
            cat = random.choice(list(s["category_counts"].keys()))
            correct = s["category_counts"][cat]
            distractors = self._make_count_distractors(correct)
            opts_raw = [correct] + distractors
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

    def _l1_category_presence(self, scenes: List[Dict]) -> List[Dict]:
        """L1: 当前帧中是否存在某类别？"""
        qa = []
        sample = random.sample(scenes, min(3, len(scenes)))
        for sd in sample:
            s, ts = sd["scene"], sd["timestamp"]
            present = set(s["category_counts"].keys())
            absent = set(self._all_valid_cats) - present
            if present:
                cat_y = random.choice(list(present))
                qa.append({
                    "question": f"Is there any {cat_y} visible in the current drone footage?",
                    "options": ["A. Yes", "B. No"],
                    "answer": "A", "answer_text": "Yes",
                    "query_timestamp": ts, "qa_type": "realtime", "difficulty": "L1",
                    "source_evidence": f"Frame@{ts}s: {cat_y} present ({s['category_counts'][cat_y]})",
                })
            if absent:
                cat_n = random.choice(list(absent))
                qa.append({
                    "question": f"Is there any {cat_n} visible in the current drone footage?",
                    "options": ["A. Yes", "B. No"],
                    "answer": "B", "answer_text": "No",
                    "query_timestamp": ts, "qa_type": "realtime", "difficulty": "L1",
                    "source_evidence": f"Frame@{ts}s: {cat_n} absent",
                })
        return qa

    def _l1_density(self, scenes: List[Dict]) -> List[Dict]:
        """L1: 当前场景密度级别？"""
        DENSITY_TEXT = {
            "empty": "Almost empty with few or no objects",
            "sparse": "Sparse with only a few objects",
            "moderate": "Moderately populated",
            "dense": "Densely populated with many objects",
            "very_dense": "Very densely crowded",
        }
        qa = []
        sample = random.sample(scenes, min(2, len(scenes)))
        for sd in sample:
            s, ts = sd["scene"], sd["timestamp"]
            correct = s["density_level"]
            wrong = [l for l in DENSITY_TEXT if l != correct]
            wrong = random.sample(wrong, min(3, len(wrong)))
            opts = [DENSITY_TEXT[correct]] + [DENSITY_TEXT[w] for w in wrong]
            random.shuffle(opts)
            labels = ["A", "B", "C", "D"]
            qa.append({
                "question": "How would you describe the current scene density from the aerial perspective?",
                "options": [f"{labels[i]}. {v}" for i, v in enumerate(opts)],
                "answer": labels[opts.index(DENSITY_TEXT[correct])],
                "answer_text": DENSITY_TEXT[correct],
                "query_timestamp": ts, "qa_type": "realtime", "difficulty": "L1",
                "source_evidence": f"Frame@{ts}s: density={correct}, total={s['total_objects']}",
            })
        return qa

    def _l1_dominant(self, scenes: List[Dict]) -> List[Dict]:
        """L1: 最常见的目标类别是什么？"""
        qa = []
        sample = random.sample(scenes, min(2, len(scenes)))
        for sd in sample:
            s, ts = sd["scene"], sd["timestamp"]
            if s["dominant_category"] == "none":
                continue
            correct = s["dominant_category"]
            wrong = [c for c in self._all_valid_cats if c != correct]
            wrong = random.sample(wrong, min(3, len(wrong)))
            opts = [correct] + wrong
            random.shuffle(opts)
            labels = ["A", "B", "C", "D"]
            qa.append({
                "question": "What is the most common type of object currently visible from the drone?",
                "options": [f"{labels[i]}. {v}" for i, v in enumerate(opts)],
                "answer": labels[opts.index(correct)],
                "answer_text": correct,
                "query_timestamp": ts, "qa_type": "realtime", "difficulty": "L1",
                "source_evidence": f"Frame@{ts}s: dominant={correct}, counts={s['category_counts']}",
            })
        return qa

    # ────── L2 生成方法 ⭐ ──────────────────────────────

    def _l2_count_change(self, events: List[Dict]) -> List[Dict]:
        """L2: 目标数量增加还是减少了？"""
        qa = []
        for ed in events:
            for evt in ed["events"]:
                if evt["type"] != "count_change":
                    continue
                ts = ed["time_span"]["to"]
                increased = evt["to"] > evt["from"]
                mag = evt["magnitude"]
                qa.append({
                    "question": "Compared to earlier frames, has the total number of objects in the scene increased or decreased?",
                    "options": [
                        f"A. Increased by about {mag} (from {evt['from']} to {evt['to']})",
                        f"B. Decreased by about {mag} (from {evt['from']} to {max(0, evt['from'] - mag)})",
                        "C. Remained roughly the same",
                        "D. Cannot be determined from the video",
                    ],
                    "answer": "A" if increased else "B",
                    "answer_text": f"{'Increased' if increased else 'Decreased'} from {evt['from']} to {evt['to']}",
                    "query_timestamp": ts, "qa_type": "backward", "difficulty": "L2",
                    "source_evidence": f"Count change: {evt['from']}→{evt['to']} over {ed['time_span']['duration']}s",
                })
        return qa[:5]

    def _l2_density_change(self, events: List[Dict]) -> List[Dict]:
        """L2: 场景密度如何变化？"""
        qa = []
        for ed in events:
            for evt in ed["events"]:
                if evt["type"] != "density_change":
                    continue
                ts = ed["time_span"]["to"]
                qa.append({
                    "question": "How has the scene density changed over the recent video segment?",
                    "options": [
                        f"A. Changed from {evt['from']} to {evt['to']}",
                        f"B. Changed from {evt['to']} to {evt['from']}",
                        f"C. Remained at {evt['from']} level throughout",
                        "D. Fluctuated unpredictably",
                    ],
                    "answer": "A",
                    "answer_text": f"Density changed from {evt['from']} to {evt['to']}",
                    "query_timestamp": ts, "qa_type": "backward", "difficulty": "L2",
                    "source_evidence": f"Density: {evt['from']}→{evt['to']}",
                })
        return qa[:3]

    def _l2_object_appear_disappear(self, events: List[Dict]) -> List[Dict]:
        """L2: 最近出现/消失了什么类型的目标？"""
        qa = []
        for ed in events:
            for evt in ed["events"]:
                if evt["type"] not in ("object_appear", "object_disappear"):
                    continue
                ts = ed["time_span"]["to"]
                action = "appeared in" if evt["type"] == "object_appear" else "disappeared from"
                correct = evt["category"]
                wrong = [c for c in self._all_valid_cats if c != correct]
                wrong = random.sample(wrong, min(3, len(wrong)))
                opts = [correct] + wrong
                random.shuffle(opts)
                labels = ["A", "B", "C", "D"]
                qa.append({
                    "question": f"What type of object has recently {action} the scene?",
                    "options": [f"{labels[i]}. {v}" for i, v in enumerate(opts)],
                    "answer": labels[opts.index(correct)],
                    "answer_text": correct,
                    "query_timestamp": ts, "qa_type": "backward", "difficulty": "L2",
                    "source_evidence": f"{correct} {action} scene @{ts}s",
                })
        return qa[:5]

    def _l2_traffic_trend(self, traffic: List[Dict]) -> List[Dict]:
        """L2: 交通流量趋势？"""
        if len(traffic) < 5:
            return []
        mid = len(traffic) // 2
        recent = traffic[mid:]
        ts = recent[-1]["timestamp"]
        counts = [t["vehicle_count"] for t in recent]
        diff = counts[-1] - counts[0]
        if diff > 3:
            trend, ans = "increasing", "A"
        elif diff < -3:
            trend, ans = "decreasing", "B"
        else:
            trend, ans = "stable", "C"
        return [{
            "question": "Based on your observation of the video stream, what is the overall trend of traffic flow?",
            "options": [
                "A. Traffic is gradually increasing",
                "B. Traffic is gradually decreasing",
                "C. Traffic has remained relatively stable",
                "D. Traffic is fluctuating unpredictably",
            ],
            "answer": ans,
            "answer_text": f"Traffic is {trend}",
            "query_timestamp": ts, "qa_type": "backward", "difficulty": "L2",
            "source_evidence": f"Vehicle counts (recent half): {counts}",
        }]

    # ────── L3 生成方法 ──────────────────────────────────

    def _l3_event_sequence(self, events: List[Dict]) -> List[Dict]:
        """L3: 将观察到的事件按时间排序"""
        all_evts = []
        for e in events:
            for evt in e["events"]:
                all_evts.append({"desc": evt["description"], "ts": e["time_span"]["to"]})
        if len(all_evts) < 3:
            return []
        selected = random.sample(all_evts, min(3, len(all_evts)))
        correct = sorted(selected, key=lambda x: x["ts"])
        correct_text = " → ".join(e["desc"] for e in correct)
        reversed_order = list(reversed(correct))
        reversed_text = " → ".join(e["desc"] for e in reversed_order)
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
            "source_evidence": f"Events sorted by time: {json.dumps(correct, ensure_ascii=False)}",
        }]

    def _l3_congestion_prediction(self, traffic: List[Dict]) -> List[Dict]:
        """L3: 基于趋势预测交通是否恶化"""
        if len(traffic) < 5:
            return []
        counts = [t["vehicle_count"] for t in traffic]
        recent = counts[-5:]
        slope = (recent[-1] - recent[0]) / max(len(recent) - 1, 1)
        ts = traffic[-1]["timestamp"]
        if slope > 1:
            pred, ans = "worsen", "A"
        elif slope < -1:
            pred, ans = "improve", "B"
        else:
            pred, ans = "remain similar", "C"
        return [{
            "question": "Based on the traffic trend observed, what is your prediction for the near future?",
            "options": [
                "A. Traffic congestion is likely to worsen",
                "B. Traffic congestion is likely to improve",
                "C. Traffic will likely remain at a similar level",
                "D. Insufficient information to make a prediction",
            ],
            "answer": ans,
            "answer_text": f"Traffic is likely to {pred}",
            "query_timestamp": ts, "qa_type": "forward", "difficulty": "L3",
            "source_evidence": f"Recent 5 vehicle counts: {recent}, slope={slope:.2f}",
        }]

    # ────── 工具方法 ─────────────────────────────────────

    @staticmethod
    def _make_count_distractors(correct: int) -> List[int]:
        """生成计数类问题的干扰选项"""
        candidates = set()
        for delta in [1, 2, 3, -1, -2, -3]:
            val = correct + delta
            if val >= 0 and val != correct:
                candidates.add(val)
        # 再加一个 ×2 或 //2 的干扰
        if correct * 2 != correct:
            candidates.add(correct * 2)
        if correct // 2 != correct and correct > 0:
            candidates.add(max(0, correct // 2))
        candidates.discard(correct)
        return list(candidates)[:3]
```

### 3.6 创建 `D:\diploma_project\temporal_encoding\dataset\quality_filter.py`

```python
"""
QA 数据集质量控制与过滤。

过滤规则:
  1. 必要字段完整性检查
  2. 选项数量 ≥ 2
  3. 正确答案必须在选项中
  4. 问题长度 10-500 字符
  5. 时间戳非负
  6. 同一视频内问题去重
"""

from typing import List, Dict


def filter_qa_dataset(qa_pairs: List[Dict], verbose: bool = True) -> List[Dict]:
    """
    质量过滤 QA 数据集。

    Args:
        qa_pairs: 原始 QA 列表
        verbose: 是否打印过滤统计

    Returns:
        过滤后的 QA 列表
    """
    required_keys = {"question", "options", "answer", "query_timestamp",
                     "qa_type", "difficulty"}

    # Pass 1: 基本完整性
    valid = []
    reasons_dropped = {}
    for qa in qa_pairs:
        # 检查必要字段
        missing = required_keys - set(qa.keys())
        if missing:
            reasons_dropped["missing_fields"] = reasons_dropped.get("missing_fields", 0) + 1
            continue

        # 选项数量
        if len(qa["options"]) < 2:
            reasons_dropped["too_few_options"] = reasons_dropped.get("too_few_options", 0) + 1
            continue

        # 答案在选项中
        option_labels = [opt[0] for opt in qa["options"]]  # 取 "A", "B", "C", "D"
        if qa["answer"] not in option_labels:
            reasons_dropped["answer_not_in_options"] = reasons_dropped.get("answer_not_in_options", 0) + 1
            continue

        # 问题长度
        qlen = len(qa["question"])
        if qlen < 10 or qlen > 500:
            reasons_dropped["question_length"] = reasons_dropped.get("question_length", 0) + 1
            continue

        # 时间戳非负
        if qa["query_timestamp"] < 0:
            reasons_dropped["negative_timestamp"] = reasons_dropped.get("negative_timestamp", 0) + 1
            continue

        valid.append(qa)

    # Pass 2: 去重（同一视频内，相同问题只保留一个）
    seen_keys = set()
    deduped = []
    for qa in valid:
        key = (qa.get("video_name", ""), qa["question"].lower().strip())
        if key not in seen_keys:
            seen_keys.add(key)
            deduped.append(qa)
        else:
            reasons_dropped["duplicate"] = reasons_dropped.get("duplicate", 0) + 1

    if verbose:
        print(f"[质量过滤] 原始: {len(qa_pairs)} → 有效: {len(valid)} → 去重: {len(deduped)}")
        if reasons_dropped:
            print(f"  丢弃原因: {reasons_dropped}")

    return deduped
```

### 3.7 创建 `D:\diploma_project\temporal_encoding\dataset\format_streaming.py`

```python
"""
将 QA 对转换为流式推理引擎的评测格式。

输出 JSON 结构（每个视频一个条目）:
{
    "video_name": "uav0000013_00000_v",
    "video_path": "/path/to/sequences/uav0000013_00000_v/",
    "fps": 2.0,
    "chunk_size": 4,
    "queries": [
        {
            "query_id": "uav0000013_00000_v_q001",
            "timestamp": 15.0,
            "chunk_index": 7,
            "question": "...",
            "options": ["A. ...", "B. ...", "C. ...", "D. ..."],
            "answer": "A",
            "answer_text": "...",
            "qa_type": "backward",
            "difficulty": "L2",
            "source_evidence": "..."
        }
    ]
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
) -> Dict:
    """
    转换为流式评测格式并保存。

    Args:
        qa_pairs: 过滤后的 QA 列表
        video_root: VisDrone 数据根目录
        output_path: 输出 JSON 文件路径
        fps: 流式推理的采样帧率
        chunk_size: 每个 chunk 的帧数

    Returns:
        统计信息字典
    """
    # 按视频名分组
    by_video: Dict[str, List[Dict]] = {}
    for qa in qa_pairs:
        vname = qa.get("video_name", "unknown")
        by_video.setdefault(vname, []).append(qa)

    dataset = []
    for vname, qas in sorted(by_video.items()):
        qas = sorted(qas, key=lambda x: x["query_timestamp"])
        queries = []
        for i, qa in enumerate(qas):
            ts = qa["query_timestamp"]
            queries.append({
                "query_id": f"{vname}_q{i + 1:03d}",
                "timestamp": ts,
                "chunk_index": int(ts * fps / chunk_size),
                "question": qa["question"],
                "options": qa["options"],
                "answer": qa["answer"],
                "answer_text": qa.get("answer_text", ""),
                "qa_type": qa["qa_type"],
                "difficulty": qa["difficulty"],
                "source_evidence": qa.get("source_evidence", ""),
            })

        video_dir = str(Path(video_root) / "sequences" / vname)
        dataset.append({
            "video_name": vname,
            "video_path": video_dir,
            "fps": fps,
            "chunk_size": chunk_size,
            "num_queries": len(queries),
            "queries": queries,
        })

    # 保存
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    # 统计
    total_qa = sum(len(d["queries"]) for d in dataset)
    by_type: Dict[str, int] = {}
    by_diff: Dict[str, int] = {}
    for d in dataset:
        for q in d["queries"]:
            by_type[q["qa_type"]] = by_type.get(q["qa_type"], 0) + 1
            by_diff[q["difficulty"]] = by_diff.get(q["difficulty"], 0) + 1

    stats = {
        "num_videos": len(dataset),
        "total_qa": total_qa,
        "by_type": by_type,
        "by_difficulty": by_diff,
    }

    print(f"✅ 评测数据集已保存: {output_path}")
    print(f"   视频数: {stats['num_videos']} | QA 总数: {stats['total_qa']}")
    print(f"   按类型: {by_type}")
    print(f"   按难度: {by_diff}")

    return stats
```

### 3.8 创建 `D:\diploma_project\temporal_encoding\dataset\sample_reporter.py`

```python
"""
样本报告生成器。
将一部分 QA 对输出为 Markdown 格式的人工审查报告。
"""

import json
import os
from typing import List, Dict
from datetime import datetime


def generate_sample_report(
    qa_pairs: List[Dict],
    video_data_list: List[Dict],
    output_path: str,
    max_samples_per_level: int = 5,
) -> str:
    """
    生成人工审查用的样本报告（Markdown 格式）。

    Args:
        qa_pairs: QA 列表
        video_data_list: 视频事件链数据列表
        output_path: 输出报告路径
        max_samples_per_level: 每个难度级别最多展示几条

    Returns:
        报告文件路径
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        f"# StreamUAV-QA 样本审查报告",
        f"",
        f"> 生成时间: {now}",
        f"> QA 总数: {len(qa_pairs)}",
        f"> 视频数: {len(set(qa.get('video_name','') for qa in qa_pairs))}",
        f"",
        f"---",
        f"",
    ]

    # 总体统计
    by_type = {}
    by_diff = {}
    for qa in qa_pairs:
        by_type[qa["qa_type"]] = by_type.get(qa["qa_type"], 0) + 1
        by_diff[qa["difficulty"]] = by_diff.get(qa["difficulty"], 0) + 1

    lines.append("## 📊 数据集统计")
    lines.append("")
    lines.append("| 维度 | 类别 | 数量 | 占比 |")
    lines.append("|------|------|------|------|")
    for k, v in sorted(by_type.items()):
        pct = v / len(qa_pairs) * 100 if qa_pairs else 0
        lines.append(f"| 类型 | {k} | {v} | {pct:.1f}% |")
    for k, v in sorted(by_diff.items()):
        pct = v / len(qa_pairs) * 100 if qa_pairs else 0
        lines.append(f"| 难度 | {k} | {v} | {pct:.1f}% |")
    lines.append("")

    # 视频信息摘要
    lines.append("## 🎬 视频信息")
    lines.append("")
    for vd in video_data_list:
        lines.append(f"- **{vd['video_name']}**: {vd['num_frames']} 帧, "
                     f"{vd['duration_seconds']}s, "
                     f"尺寸 {vd.get('image_size', ['?','?'])}, "
                     f"采样 {vd.get('num_sampled_frames', '?')} 帧, "
                     f"事件 {vd.get('num_events', '?')} 个")
    lines.append("")

    # 按难度级别展示样本
    for level in ["L1", "L2", "L3"]:
        level_qa = [qa for qa in qa_pairs if qa.get("difficulty") == level]
        if not level_qa:
            continue

        emoji = {"L1": "🟢", "L2": "🟡⭐", "L3": "🔴"}
        lines.append(f"## {emoji.get(level, '')} {level} 样本 ({len(level_qa)} 条)")
        lines.append("")

        samples = level_qa[:max_samples_per_level]
        for i, qa in enumerate(samples, 1):
            lines.append(f"### {level}-{i}: [{qa['qa_type']}] {qa.get('video_name', '')}")
            lines.append(f"")
            lines.append(f"**⏱ 时间戳**: {qa['query_timestamp']}s")
            lines.append(f"")
            lines.append(f"**❓ 问题**: {qa['question']}")
            lines.append(f"")
            for opt in qa["options"]:
                marker = "✅" if opt[0] == qa["answer"] else "  "
                lines.append(f"  {marker} {opt}")
            lines.append(f"")
            lines.append(f"**📋 正确答案**: {qa['answer']}. {qa.get('answer_text', '')}")
            lines.append(f"")
            lines.append(f"**🔍 依据**: `{qa.get('source_evidence', 'N/A')}`")
            lines.append(f"")
            lines.append(f"---")
            lines.append(f"")

    # 审查指引
    lines.append("## ✅ 审查要点")
    lines.append("")
    lines.append("请检查以下方面：")
    lines.append("")
    lines.append("1. **问题是否清晰？** — 读者能否理解在问什么？")
    lines.append("2. **正确答案是否确实正确？** — 对照 source_evidence 验证")
    lines.append("3. **干扰选项是否合理？** — 不能太明显也不能太离谱")
    lines.append("4. **L2 时序问题是否需要跨帧信息？** — 这是核心差异化指标")
    lines.append("5. **时间戳是否合理？** — 不应超出视频时长")
    lines.append("")
    lines.append("如果发现系统性问题（如某类 QA 全部有误），请记录问题类型，")
    lines.append("后续可以针对性修复 qa_generator.py 中的对应方法。")

    report_text = "\n".join(lines)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"✅ 样本报告已生成: {output_path}")
    return output_path
PYEOF
echo "✅ sample_reporter.py 已创建"
```

### 4.9 创建 `build_dataset.py`（主构建脚本）

```bash
cat > /root/autodl-tmp/diploma_project/temporal_encoding/dataset/build_dataset.py << 'PYEOF'
"""
StreamUAV-QA v1 数据集一键构建脚本。

用法:
  # 试运行（5 个视频）:
  python build_dataset.py --mode sample --max-videos 5

  # 完整构建（50 个视频）:
  python build_dataset.py --mode full --max-videos 50

  # 自定义:
  python build_dataset.py --mode full --max-videos 30 --max-qa-per-video 25 --sample-interval 15
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# 确保当前目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from visdrone_loader import load_visdrone_vid
from event_extractor import extract_event_chain
from qa_generator import StreamingQAGenerator
from quality_filter import filter_qa_dataset
from format_streaming import convert_to_streaming_eval_format
from sample_reporter import generate_sample_report


def main():
    parser = argparse.ArgumentParser(
        description="StreamUAV-QA v1 数据集构建",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mode", choices=["sample", "full"], default="sample",
                        help="sample=试运行(5视频), full=完整构建")
    parser.add_argument("--visdrone-root",
                        default="/root/autodl-tmp/datasets/visdrone/VID-train",
                        help="VisDrone VID 数据集根目录")
    parser.add_argument("--output",
                        default="/root/autodl-tmp/datasets/stream_uav_qa",
                        help="输出目录")
    parser.add_argument("--max-videos", type=int, default=None,
                        help="最多处理视频数 (sample默认5, full默认50)")
    parser.add_argument("--max-qa-per-video", type=int, default=30,
                        help="每个视频最多生成 QA 数")
    parser.add_argument("--sample-interval", type=int, default=30,
                        help="标注采样间隔(帧数), 30=@30fps每秒采样1帧")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=float, default=2.0,
                        help="流式推理目标帧率")
    parser.add_argument("--chunk-size", type=int, default=4,
                        help="每个 chunk 的帧数")
    args = parser.parse_args()

    # 默认视频数
    if args.max_videos is None:
        args.max_videos = 5 if args.mode == "sample" else 50

    os.makedirs(args.output, exist_ok=True)
    start_time = time.time()

    # ═══════════════════════════════════════════════════════
    # Step 1: 加载 VisDrone 数据
    # ═══════════════════════════════════════════════════════
    print("=" * 70)
    print(f"  StreamUAV-QA v1 数据集构建 — 模式: {args.mode}")
    print(f"  视频数: {args.max_videos} | QA/视频: {args.max_qa_per_video}")
    print(f"  采样间隔: {args.sample_interval} 帧 | 种子: {args.seed}")
    print("=" * 70)
    print()
    print("[Step 1/5] 加载 VisDrone-VID 数据...")
    sequences = load_visdrone_vid(args.visdrone_root, max_sequences=args.max_videos)
    if not sequences:
        print("❌ 未找到任何视频序列！请检查 --visdrone-root 路径")
        sys.exit(1)
    print(f"  已加载 {len(sequences)} 个序列")
    print()

    # ═══════════════════════════════════════════════════════
    # Step 2: 提取事件链
    # ═══════════════════════════════════════════════════════
    print("[Step 2/5] 提取时序事件链...")
    all_video_data = []
    for i, seq in enumerate(sequences):
        print(f"  [{i+1}/{len(sequences)}] {seq.seq_name} "
              f"({seq.num_frames} 帧, {seq.num_frames/30:.1f}s)")
        vd = extract_event_chain(seq, sample_interval=args.sample_interval)
        all_video_data.append(vd)
        print(f"    → 采样 {vd['num_sampled_frames']} 帧, "
              f"提取 {vd['num_events']} 个事件")

    # 保存中间结果
    events_path = os.path.join(args.output, "video_events.json")
    with open(events_path, 'w', encoding='utf-8') as f:
        json.dump(all_video_data, f, indent=2, ensure_ascii=False)
    print(f"  ✅ 事件链已保存: {events_path}")
    print()

    # ═══════════════════════════════════════════════════════
    # Step 3: 生成 QA 对
    # ═══════════════════════════════════════════════════════
    print("[Step 3/5] 生成 QA 对...")
    gen = StreamingQAGenerator(seed=args.seed)
    all_qa = []
    for vd in all_video_data:
        qa = gen.generate_for_video(vd, max_qa_per_video=args.max_qa_per_video)
        all_qa.extend(qa)
        print(f"  {vd['video_name']}: 生成 {len(qa)} 条 QA")

    print(f"  总计: {len(all_qa)} 条 QA")
    print()

    # ═══════════════════════════════════════════════════════
    # Step 4: 质量过滤
    # ═══════════════════════════════════════════════════════
    print("[Step 4/5] 质量过滤...")
    filtered_qa = filter_qa_dataset(all_qa)
    print()

    # ═══════════════════════════════════════════════════════
    # Step 5: 格式化 + 报告
    # ═══════════════════════════════════════════════════════
    print("[Step 5/5] 格式化并保存...")

    # 5a. 保存评测格式
    eval_path = os.path.join(args.output, "stream_uav_eval.json")
    stats = convert_to_streaming_eval_format(
        filtered_qa, video_root=args.visdrone_root,
        output_path=eval_path, fps=args.fps, chunk_size=args.chunk_size,
    )

    # 5b. 保存完整 QA 列表（调试用）
    all_qa_path = os.path.join(args.output, "all_qa_pairs.json")
    with open(all_qa_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_qa, f, indent=2, ensure_ascii=False)

    # 5c. 生成样本报告
    report_path = os.path.join(args.output, "sample_review_report.md")
    generate_sample_report(filtered_qa, all_video_data, report_path)

    # 5d. 保存统计摘要
    elapsed = time.time() - start_time
    summary = {
        "build_mode": args.mode,
        "build_args": vars(args),
        "elapsed_seconds": round(elapsed, 1),
        "num_videos": len(sequences),
        "num_qa_raw": len(all_qa),
        "num_qa_filtered": len(filtered_qa),
        "stats": stats,
    }
    summary_path = os.path.join(args.output, "build_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 最终输出
    print()
    print("=" * 70)
    print(f"  ✅ StreamUAV-QA v1 数据集构建完成！")
    print(f"  模式: {args.mode} | 耗时: {elapsed:.1f}s")
    print(f"  视频: {len(sequences)} | QA: {len(filtered_qa)}")
    print(f"  输出目录: {args.output}")
    print(f"  ")
    print(f"  📁 产出文件:")
    print(f"    stream_uav_eval.json  — 流式评测格式（主文件）")
    print(f"    all_qa_pairs.json     — 完整 QA 列表")
    print(f"    video_events.json     — 视频事件链中间数据")
    print(f"    sample_review_report.md — 人工审查报告 ⬅️ 请先查看")
    print(f"    build_summary.json    — 构建统计摘要")
    print("=" * 70)

    if args.mode == "sample":
        print()
        print("🔴 当前为试运行模式。请先查看 sample_review_report.md，")
        print("   确认 QA 质量无问题后，使用以下命令进行完整构建:")
        print(f"   python build_dataset.py --mode full --max-videos 50")


if __name__ == "__main__":
    main()
```

---

## Step 4: 语法验证（import 链测试）

> 确保所有代码无语法错误、import 链正确。

```powershell
# 进入代码目录
cd D:\diploma_project\temporal_encoding\dataset

# 验证所有文件语法
$files = @("visdrone_loader.py","scene_analyzer.py","event_extractor.py",
           "qa_generator.py","quality_filter.py","format_streaming.py",
           "sample_reporter.py","build_dataset.py")
foreach ($f in $files) {
    python -c "import py_compile; py_compile.compile('$f', doraise=True)" 2>$null
    if ($LASTEXITCODE -eq 0) { Write-Host "✅ $f 语法正确" }
    else { Write-Host "❌ $f 语法错误！" }
}
```

```python
# import 链检查（在 D:\diploma_project\temporal_encoding\dataset\ 下运行）
import sys
sys.path.insert(0, '.')
from visdrone_loader import load_visdrone_vid, VALID_CATEGORIES
print(f'✅ visdrone_loader: {len(VALID_CATEGORIES)} 个有效类别')

from scene_analyzer import analyze_frame_scene, analyze_traffic_density
print('✅ scene_analyzer: 2 个分析函数')

from event_extractor import extract_event_chain, compute_temporal_diff
print('✅ event_extractor: 事件链提取器')

from qa_generator import StreamingQAGenerator
gen = StreamingQAGenerator(seed=42)
print('✅ qa_generator: QA 生成器初始化成功')

from quality_filter import filter_qa_dataset
print('✅ quality_filter: 过滤器')

from format_streaming import convert_to_streaming_eval_format
print('✅ format_streaming: 格式化器')

from sample_reporter import generate_sample_report
print('✅ sample_reporter: 报告生成器')

print()
print('✅ Step 4 完成: 所有模块可正常导入')
```

**✅ 验证标准**: 8 个文件全部语法正确，7 个模块全部可导入

---

## Step 5: 试运行 — 5 个视频的 QA 生成

> 纯 CPU 操作，用于快速验证 Pipeline 是否工作正常。

```powershell
cd D:\diploma_project\temporal_encoding\dataset

# 试运行（5 个视频，预计 < 2 分钟）
python build_dataset.py `
    --mode sample `
    --visdrone-root D:\diploma_project\datasets\visdrone\VID-train `
    --output D:\diploma_project\datasets\stream_uav_qa_sample `
    --max-videos 5
```

**✅ 验证标准**:
- 无报错退出
- 输出目录下有 5 个文件
- `sample_review_report.md` 非空
- QA 总数 > 50

```powershell
# 验证输出
Write-Host "=== 输出文件 ==="
Get-ChildItem D:\diploma_project\datasets\stream_uav_qa_sample\

Write-Host ""
Write-Host "=== 构建摘要 ==="
Get-Content D:\diploma_project\datasets\stream_uav_qa_sample\build_summary.json

Write-Host ""
Write-Host "=== 报告前 50 行 ==="
Get-Content D:\diploma_project\datasets\stream_uav_qa_sample\sample_review_report.md | Select-Object -First 50
```

---

## 🔴 暂停点 1：等待人工确认

> **⚠️ 在此处暂停！将以下文件内容发送给用户审查：**
>
> 1. 在 VS Code 中打开 `D:\diploma_project\datasets\stream_uav_qa_sample\sample_review_report.md`
> 2. 在 VS Code 中打开 `D:\diploma_project\datasets\stream_uav_qa_sample\build_summary.json`
>
> **告知用户**:
> - 请查看 `sample_review_report.md`，重点检查：
>   - L1 (事实性) QA 的答案是否合理
>   - L2 (时序性) QA 是否确实需要跨帧信息
>   - 干扰选项是否合理（不能太明显也不能太离谱）
>   - 时间戳是否在视频时长范围内
> - 如有问题，请告知具体哪类 QA 有什么问题
> - 确认无误后，回复 "确认" 继续完整构建

---

## Step 6: 完整数据集构建（人工确认后执行）

```powershell
cd D:\diploma_project\temporal_encoding\dataset

# 完整构建（50 个视频，预计 5-15 分钟，纯 CPU）
python build_dataset.py `
    --mode full `
    --visdrone-root D:\diploma_project\datasets\visdrone\VID-train `
    --output D:\diploma_project\datasets\stream_uav_qa `
    --max-videos 50 `
    --max-qa-per-video 30

# 验证完整数据集
Write-Host "=== 完整数据集验证 ==="
Get-ChildItem D:\diploma_project\datasets\stream_uav_qa\ | Format-Table Name, Length
```

```python
# 打印构建摘要（在 Python 中运行）
import json
with open(r'D:\diploma_project\datasets\stream_uav_qa\build_summary.json') as f:
    s = json.load(f)
print(f'视频数: {s["num_videos"]}')
print(f'QA 总数: {s["num_qa_filtered"]}')
print(f'按类型: {s["stats"]["by_type"]}')
print(f'按难度: {s["stats"]["by_difficulty"]}')
print(f'耗时: {s["elapsed_seconds"]}s')
print('✅ Step 6 完成')
```

**✅ 验证标准**:
- QA 总数 > 500
- L2 占比 > 30%（这是核心差异化指标）
- 无报错

---

## ✅ 构建完成通知

> **在此处通知用户**:
>
> ```
> ✅ StreamUAV-QA v1 数据集构建完成！
>
> 📊 统计: XX 个视频, XX 条 QA (L1:XX, L2:XX, L3:XX)
> 📁 输出: D:\diploma_project\datasets\stream_uav_qa\
>
> GPU 模型推理验证阶段（需要 Qwen2.5-VL-3B 运行 7.1GB VRAM）
> 已标记为 SKIP，留待 AutoDL 4090 环境执行。
>
> 如需立刻使用数据集，可将
> D:\diploma_project\datasets\stream_uav_qa\
> 内容备份并上传到 AutoDL 服务器。
> ```

---

# ⛔ 阶段二：GPU 模型推理验证 — **当前环境跳过**

> **跳过原因**: RTX 3050 仅 4GB VRAM < Qwen2.5-VL-3B 所需的 7.1GB
>
> **执行条件**: 需要在 AutoDL RTX 4090 (24GB) 环境下执行
>
> **全部 Step 9-11 内容保留在文档中供参考**，待切换到 AutoDL 时使用。

---

## Step 9: GPU 环境准备

```bash
# 激活环境
conda activate streamuav

# 安装 GPU 相关依赖（仅有卡模式需要）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate

# 验证 GPU
python -c "
import torch
print(f'CUDA: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
print('✅ Step 9 完成')
"
```

## Step 10: 模型推理验证

```bash
cat > /root/autodl-tmp/diploma_project/temporal_encoding/dataset/verify_with_model.py << 'PYEOF'
"""
用 Qwen2.5-VL 实际回答随机 QA，验证数据集合理性。
"""

import json
import random
import glob
import torch
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def main():
    # 加载数据集
    with open("/root/autodl-tmp/datasets/stream_uav_qa/stream_uav_eval.json") as f:
        dataset = json.load(f)

    # 随机选 10 条 QA（跨不同视频和难度级别）
    all_queries = []
    for video in dataset:
        for q in video["queries"]:
            q["_video_path"] = video["video_path"]
            all_queries.append(q)

    random.seed(42)
    samples = random.sample(all_queries, min(10, len(all_queries)))

    # 加载模型
    model_path = "/root/autodl-tmp/Qwen/Qwen2___5-VL-3B-Instruct"
    print(f"加载模型: {model_path}")
    processor = AutoProcessor.from_pretrained(model_path)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16
    ).to("cuda").eval()
    print("✅ 模型加载完成")

    # 逐条验证
    results = []
    for i, q in enumerate(samples):
        print(f"\n[{i+1}/10] {q['query_id']} ({q['difficulty']}/{q['qa_type']})")
        print(f"  Q: {q['question']}")

        # 取该时间戳附近的一帧
        video_path = q["_video_path"]
        frame_files = sorted(glob.glob(f"{video_path}/*.jpg"))
        if not frame_files:
            frame_files = sorted(glob.glob(f"{video_path}/*.png"))

        if not frame_files:
            print("  ⚠️ 未找到帧文件，跳过")
            continue

        # 取中间帧
        mid_idx = min(int(q["timestamp"] * 2), len(frame_files) - 1)  # 粗略估计
        mid_idx = max(0, min(mid_idx, len(frame_files) - 1))
        frame = Image.open(frame_files[mid_idx]).convert("RGB")

        # 构建 prompt
        prompt_text = (
            f"This is aerial drone footage. "
            f"Question: {q['question']}\n"
            + "\n".join(q["options"])
            + "\nAnswer with the letter only (A, B, C, or D)."
        )

        messages = [{"role": "user", "content": [
            {"type": "image", "image": frame},
            {"type": "text", "text": prompt_text},
        ]}]

        text = processor.apply_chat_template(messages, tokenize=False,
                                              add_generation_prompt=True)
        inputs = processor(text=[text], images=[frame],
                           return_tensors="pt", padding=True).to("cuda")

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=20)
        resp = processor.batch_decode(out[:, inputs.input_ids.shape[1]:],
                                       skip_special_tokens=True)[0].strip()

        # 解析答案
        pred = "?"
        for letter in "ABCD":
            if letter in resp:
                pred = letter
                break

        correct = pred == q["answer"]
        results.append({
            "query_id": q["query_id"],
            "difficulty": q["difficulty"],
            "qa_type": q["qa_type"],
            "question": q["question"][:80] + "...",
            "gt": q["answer"],
            "pred": pred,
            "correct": correct,
            "model_response": resp[:100],
        })

        emoji = "✅" if correct else "❌"
        print(f"  GT: {q['answer']} | Pred: {pred} {emoji}")
        print(f"  Model: {resp[:80]}")

        # 释放显存
        del inputs
        torch.cuda.empty_cache()

    # 汇总
    print("\n" + "=" * 60)
    print("模型验证汇总")
    print("=" * 60)
    total = len(results)
    correct_count = sum(r["correct"] for r in results)
    print(f"总计: {correct_count}/{total} = {correct_count/total*100:.0f}%")

    for diff in ["L1", "L2", "L3"]:
        sub = [r for r in results if r["difficulty"] == diff]
        if sub:
            acc = sum(r["correct"] for r in sub) / len(sub)
            print(f"  {diff}: {sum(r['correct'] for r in sub)}/{len(sub)} = {acc*100:.0f}%")

    # 保存结果
    verify_path = "/root/autodl-tmp/datasets/stream_uav_qa/model_verify_results.json"
    with open(verify_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n💾 验证结果: {verify_path}")

    # 解读
    print()
    if correct_count / total >= 0.3:
        print("📋 解读: 模型在单帧上有一定正答率，说明 QA 问题合理。")
        print("   注意: L2 时序问题用单帧回答正确率低是 *预期行为*，")
        print("   说明这些问题确实需要历史帧信息（流式推理的价值所在）。")
    else:
        print("⚠️ 正答率较低。请检查:")
        print("   1. 如果 L1 也低 → 模型可能无法理解 UAV 俯瞰视角")
        print("   2. 如果只有 L2/L3 低 → 正常，单帧无法回答时序问题")


if __name__ == "__main__":
    main()
PYEOF

# 运行验证
cd /root/autodl-tmp/diploma_project/temporal_encoding/dataset
python verify_with_model.py
```

---

## Step 11: 最终输出

### 11.1 最终项目结构

```bash
echo "=== 最终项目结构 ==="
find /root/autodl-tmp/diploma_project/temporal_encoding/dataset/ -type f | sort
echo ""
find /root/autodl-tmp/datasets/stream_uav_qa/ -type f | sort
```

预期结构:
```
/root/autodl-tmp/
├── diploma_project/
│   └── temporal_encoding/
│       └── dataset/                         # 代码目录
│           ├── __init__.py
│           ├── visdrone_loader.py            # VisDrone 数据加载器
│           ├── scene_analyzer.py             # 场景分析（Stage 1）
│           ├── event_extractor.py            # 事件链提取（Stage 2）
│           ├── qa_generator.py              # QA 生成（Stage 3）
│           ├── quality_filter.py             # 质量过滤
│           ├── format_streaming.py           # 格式转换
│           ├── sample_reporter.py            # 审查报告生成
│           ├── build_dataset.py             # 主构建脚本
│           └── verify_with_model.py         # 模型验证（可选）
│
├── datasets/
│   ├── visdrone/                            # 原始数据
│   │   ├── VID-train/
│   │   │   ├── sequences/                   # 视频帧
│   │   │   └── annotations/                 # GT 标注
│   │   └── VID-val/
│   │
│   ├── stream_uav_qa_sample/               # 试运行输出
│   │   ├── stream_uav_eval.json
│   │   ├── all_qa_pairs.json
│   │   ├── video_events.json
│   │   ├── sample_review_report.md
│   │   └── build_summary.json
│   │
│   └── stream_uav_qa/                      # 完整数据集 ⬅️ 主产出
│       ├── stream_uav_eval.json             # 流式评测格式（主文件）
│       ├── all_qa_pairs.json                # 完整 QA 列表
│       ├── video_events.json                # 事件链中间数据
│       ├── sample_review_report.md          # 人工审查报告
│       ├── build_summary.json               # 构建统计
│       └── model_verify_results.json        # 模型验证结果（可选）
```

### 11.2 使用手册

```
╔════════════════════════════════════════════════════════════╗
║        StreamUAV-QA v1 数据集 — 使用手册                    ║
╠════════════════════════════════════════════════════════════╣
║                                                            ║
║  📂 主文件: stream_uav_eval.json                           ║
║                                                            ║
║  格式: JSON 数组，每个元素是一个视频:                        ║
║  [                                                         ║
║    {                                                       ║
║      "video_name": "uav0000013_00000_v",                  ║
║      "video_path": "/path/to/sequences/...",              ║
║      "fps": 2.0,                                          ║
║      "chunk_size": 4,                                     ║
║      "queries": [                                          ║
║        {                                                   ║
║          "query_id": "uav0000013_00000_v_q001",           ║
║          "timestamp": 5.0,     ← 在此时刻提问              ║
║          "chunk_index": 2,     ← 第几个 chunk 后提         ║
║          "question": "How many cars...?",                  ║
║          "options": ["A. 3", "B. 5", ...],                ║
║          "answer": "A",        ← 正确答案字母              ║
║          "qa_type": "backward", ← realtime/backward/forward║
║          "difficulty": "L2"    ← L1/L2/L3                 ║
║        }                                                   ║
║      ]                                                     ║
║    }                                                       ║
║  ]                                                         ║
║                                                            ║
║  使用方式（在评测脚本中）:                                   ║
║    1. 逐 chunk 喂帧给流式引擎                               ║
║    2. 当 chunk_index 匹配时，调用 ask_choice()              ║
║    3. 比较模型回答与 answer → 计算 accuracy                  ║
║    4. 按 L1/L2/L3 分层统计 → L2 是核心指标                  ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
```

---

## 🛠️ 故障排除

| 问题 | 解决方案 |
|------|----------|
| `gdown` 下载失败 | 改用百度网盘下载，或 `aria2c` 多线程下载 |
| 系统盘爆满 | 执行 Step 1 的防爆配置；`pip cache purge && conda clean --all -y` |
| `No module named 'visdrone_loader'` | `cd` 到代码目录再运行；或 `sys.path.insert(0, '代码目录')` |
| 标注解析出错 | 检查 VisDrone 解压是否完整；某些行可能格式异常会自动跳过 |
| QA 数量太少 | 减小 `--sample-interval`（如 15），增加采样密度 |
| 有卡模式模型加载 OOM | 使用 3B 而非 7B；或加 `torch_dtype=torch.float16` |
| PIL 打开图片失败 | `pip install Pillow --force-reinstall` |
