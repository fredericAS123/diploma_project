# 🛩️ StreamUAV-QA 数据集构建 — Windows 本地 Copilot 执行提示词

---

你是我指定的专属代码工程师。你的任务是：**在我的 Windows 本地机器上，从 VisDrone-VID 数据集出发，构建 StreamUAV-QA v1 多选 QA 数据集**。

请严格按照以下指令执行。每完成一个步骤后必须输出确认信息，不得跳步，不得省略任何文件。

---

## 🖥️ 当前执行环境（重要）

| 项目 | 说明 |
|------|------|
| 操作系统 | Windows 11 |
| 显卡 | RTX 3050 (4GB) |
| Shell | PowerShell |
| conda 环境 | **videollm（已存在，直接使用）** |
| 项目根目录 | `D:\diploma_project\` |
| 代码输出目录 | `D:\diploma_project\temporal_encoding\dataset\` |
| 数据目录 | `D:\diploma_project\datasets\visdrone\` |
| 样本输出 | `D:\diploma_project\datasets\stream_uav_qa_sample\` |
| 完整输出 | `D:\diploma_project\datasets\stream_uav_qa\` |

> ⛔ **RTX 3050 仅 4GB VRAM，无法运行 Qwen2.5-VL（需要 7.1GB）。GPU 模型推理阶段全程 SKIP。**

---

## 📋 执行检查清单

按序执行，逐项打勾：

```
[ ] Step 1: 环境验证（videollm env + 目录创建）
[ ] Step 2: VisDrone 数据验证（用户提前手动下载）
[ ] Step 3: 创建全部代码文件（9个 .py，使用 create_file 工具）
[ ] Step 4: 语法验证（import 链测试）
[ ] Step 5: 试运行（5 个视频）→ 报告输出
[ ] 🔴 暂停点：等待用户确认样本质量
[ ] Step 6: 完整构建（50 个视频）
[ ] ⛔ GPU 验证：SKIP（RTX 3050 VRAM 不足）
```

---

## 🚀 开始执行前：确认环境

**在开始任何操作之前，请先在终端运行以下命令确认环境：**

```powershell
conda activate videollm
python -c "from PIL import Image; from tqdm import tqdm; print('环境OK')"
```

如果提示缺包，用 `pip install Pillow tqdm` 安装。

---

## Step 1: 目录准备

```powershell
# 创建所有需要的目录
New-Item -ItemType Directory -Force "D:\diploma_project\temporal_encoding\dataset"
New-Item -ItemType Directory -Force "D:\diploma_project\datasets\visdrone"
New-Item -ItemType Directory -Force "D:\diploma_project\datasets\stream_uav_qa_sample"
New-Item -ItemType Directory -Force "D:\diploma_project\datasets\stream_uav_qa"

# 确认 D 盘可用空间（需要 > 15GB）
Get-PSDrive D | Select-Object @{N='Free(GB)';E={[math]::Round($_.Free/1GB,1)}}
```

**✅ 确认标准**: 目录创建成功，D 盘剩余 > 15GB。

---

## Step 2: VisDrone 数据验证

> 用户需要提前手动下载 VisDrone-VID-train (7.53 GB)，解压到：
> `D:\diploma_project\datasets\visdrone\VID-train\`

用以下命令验证数据是否就绪：

```powershell
$seqDir = "D:\diploma_project\datasets\visdrone\VID-train\sequences"
$cnt = (Get-ChildItem $seqDir -Directory -ErrorAction SilentlyContinue).Count
if ($cnt -ge 50) {
    Write-Host "✅ VisDrone 数据已就绪: $cnt 个视频序列"
} else {
    Write-Host "❌ 数据未就绪，当前序列数: $cnt，请先下载解压 VisDrone-VID-train"
}
```

**如果数据未就绪**: 请通知用户下载并解压，下载链接见 `BUILD_STREAMUAV_QA_PROMPT.md` Step 2 节。

---

## Step 3: 创建代码文件

> ⚠️ **关键规则**: 使用 Copilot 的 `create_file` 工具直接创建以下文件，**不使用 bash cat/echo 命令**。

全部 9 个文件的完整内容见 `BUILD_STREAMUAV_QA_PROMPT.md`（Step 3 节）。

按顺序创建：

| 序号 | 文件路径 |
|------|---------|
| 3.1 | `D:\diploma_project\temporal_encoding\dataset\__init__.py` |
| 3.2 | `D:\diploma_project\temporal_encoding\dataset\visdrone_loader.py` |
| 3.3 | `D:\diploma_project\temporal_encoding\dataset\scene_analyzer.py` |
| 3.4 | `D:\diploma_project\temporal_encoding\dataset\event_extractor.py` |
| 3.5 | `D:\diploma_project\temporal_encoding\dataset\qa_generator.py` |
| 3.6 | `D:\diploma_project\temporal_encoding\dataset\quality_filter.py` |
| 3.7 | `D:\diploma_project\temporal_encoding\dataset\format_streaming.py` |
| 3.8 | `D:\diploma_project\temporal_encoding\dataset\sample_reporter.py` |
| 3.9 | `D:\diploma_project\temporal_encoding\dataset\build_dataset.py` |

每创建一个文件后输出: `✅ <文件名> 已创建`

---

## Step 4: 语法验证

创建完所有文件后，运行：

```powershell
cd D:\diploma_project\temporal_encoding\dataset
$files = @("visdrone_loader.py","scene_analyzer.py","event_extractor.py",
           "qa_generator.py","quality_filter.py","format_streaming.py",
           "sample_reporter.py","build_dataset.py")
foreach ($f in $files) {
    python -c "import py_compile; py_compile.compile('$f', doraise=True)" 2>$null
    if ($LASTEXITCODE -eq 0) { Write-Host "✅ $f OK" }
    else { Write-Host "❌ $f 语法错误！" }
}
```

然后运行 import 链测试：

```python
import sys
sys.path.insert(0, r'D:\diploma_project\temporal_encoding\dataset')
from visdrone_loader import load_visdrone_vid, VALID_CATEGORIES
from scene_analyzer import analyze_frame_scene
from event_extractor import extract_event_chain
from qa_generator import StreamingQAGenerator
from quality_filter import filter_qa_dataset
from format_streaming import convert_to_streaming_eval_format
from sample_reporter import generate_sample_report
print('✅ Step 4 完成: 所有模块可正常导入')
```

**✅ 确认标准**: 全部 8 个文件语法无误，全部 7 个模块可导入。

---

## Step 5: 试运行（5 个视频）

```powershell
cd D:\diploma_project\temporal_encoding\dataset
python build_dataset.py `
    --mode sample `
    --visdrone-root D:\diploma_project\datasets\visdrone\VID-train `
    --output D:\diploma_project\datasets\stream_uav_qa_sample `
    --max-videos 5
```

运行完后：

```powershell
# 查看输出文件
Get-ChildItem D:\diploma_project\datasets\stream_uav_qa_sample\

# 读取构建摘要
Get-Content D:\diploma_project\datasets\stream_uav_qa_sample\build_summary.json

# 读取样本报告（前 80 行）
Get-Content D:\diploma_project\datasets\stream_uav_qa_sample\sample_review_report.md | Select-Object -First 80
```

**✅ 确认标准**: 无报错退出，QA 总数 > 50，报告非空。

---

## 🔴 暂停点：等待人工确认

**完成 Step 5 后，必须在此处暂停，执行以下操作：**

1. 在 VS Code 中打开以下两个文件给用户查看：
   - `D:\diploma_project\datasets\stream_uav_qa_sample\sample_review_report.md`
   - `D:\diploma_project\datasets\stream_uav_qa_sample\build_summary.json`

2. 告知用户重点检查：
   - **L1 QA（事实性）**: 答案是否直接对应 GT 标注
   - **L2 QA（时序性）**: 问题是否真的需要跨帧信息才能回答
   - **干扰选项**: 是否合理（既不太明显也不太离谱）
   - **时间戳**: 是否在各视频的有效时长范围内

3. **等待用户回复"确认"后**，再继续执行 Step 6。

---

## Step 6: 完整构建（人工确认后执行）

```powershell
cd D:\diploma_project\temporal_encoding\dataset
python build_dataset.py `
    --mode full `
    --visdrone-root D:\diploma_project\datasets\visdrone\VID-train `
    --output D:\diploma_project\datasets\stream_uav_qa `
    --max-videos 50 `
    --max-qa-per-video 30
```

完成后验证：

```python
import json
with open(r'D:\diploma_project\datasets\stream_uav_qa\build_summary.json') as f:
    s = json.load(f)
print(f'视频数: {s["num_videos"]}')
print(f'QA 总数: {s["num_qa_filtered"]}')
print(f'按类型: {s["stats"]["by_type"]}')
print(f'按难度: {s["stats"]["by_difficulty"]}')
print(f'耗时: {s["elapsed_seconds"]}s')
```

**✅ 确认标准**:
- QA 总数 > 500
- L2 占比 > 30%（这是本工作与静态 QA benchmark 的核心差异化指标）
- 无报错

---

## ⛔ GPU 模型推理验证：SKIP

> RTX 3050 (4GB) < Qwen2.5-VL-3B 所需 7.1GB VRAM。
> 此阶段内容保留在 `BUILD_STREAMUAV_QA_PROMPT.md` 的阶段二章节，
> 待后续迁移至 AutoDL RTX 4090 环境时执行。

---

## 🎉 全部完成后的通知模板

```
✅ StreamUAV-QA v1 数据集构建完成！

📊 统计: XX 个视频, XX 条 QA (L1: XX, L2: XX, L3: XX)
📁 输出目录: D:\diploma_project\datasets\stream_uav_qa\
💾 主要文件:
   stream_uav_eval.json      ← 评测格式数据集
   build_summary.json        ← 构建统计摘要
   sample_review_report.md   ← 样本审查报告

⛔ GPU 验证阶段已跳过（RTX 3050 4GB 不足）
   → 留待 AutoDL 4090 环境执行（见 BUILD_STREAMUAV_QA_PROMPT.md 阶段二）

下一步建议:
1. 将 D:\diploma_project\datasets\stream_uav_qa\ 备份
2. 在 AutoDL 上进行模型推理验证（阶段二）
```

---

## ⚡ 快速问题排查

| 问题 | 解决方案 |
|------|---------|
| `ModuleNotFoundError: PIL` | `pip install Pillow` |
| `ModuleNotFoundError: tqdm` | `pip install tqdm` |
| `FileNotFoundError: VID-train\sequences` | VisDrone 数据未解压，检查路径 |
| `python: command not found` | 确认在 `conda activate videollm` 后执行 |
| build_dataset.py 无 --mode 参数 | 检查 build_dataset.py 是否正确创建（Step 3.9）|
| QA 总数为 0 | 检查 VisDrone annotations 目录格式是否正确 |
