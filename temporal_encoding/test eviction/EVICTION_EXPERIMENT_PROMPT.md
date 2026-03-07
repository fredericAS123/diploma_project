# KV Cache 淘汰策略实验验证 Prompt

> **用途**: 在 GPU 机器 (AutoDL, RTX 4090 24GB) 的 Copilot Agent 上逐步执行
> **前提**: 已部署 `/root/autodl-tmp/diploma_project/` 项目，模型权重在 `/root/autodl-tmp/Qwen/Qwen2___5-VL-3B-Instruct`
> **视频文件**: `/root/autodl-tmp/diploma_project/temporal_encoding/202208312002.mp4`

---

## 总体说明

本文件包含 **3 个实验**，分别验证本次 KV Cache 淘汰策略实现中的 3 个核心设计：

| 实验 | 验证目标 | 对应修正 |
|------|---------|---------|
| **实验 A** | sink_size 自动检测正确性 | 不硬编码 128，从首 chunk 实际长度推导 |
| **实验 B** | OOM-Free 长视频处理 | 无淘汰 120 帧即 OOM，启用淘汰后应无限 |
| **实验 C** | 淘汰后 ask() 质量不降级 | 滑窗 + 周期性提问，提取歌词/字幕 |

### 关键数据参考 (test_step10 实测, RTX 4090 24GB)

| 参数 | 值 |
|------|-----|
| 模型 VRAM (Qwen2.5-VL-3B, bf16) | 7.1 GB allocated, 7.33 GB reserved |
| KV cache 每 token | ~36 KB (across 36 layers) |
| 1920×1080, 4帧/chunk | ~5,389 tokens/chunk (~0.185 GB/chunk) |
| 首 chunk (含 system prompt) | ~5,438 tokens (多 ~49 个文本 token) |
| 30 chunks (120帧) 无淘汰 | cache 161,719 tokens, VRAM reserved 22.89 GB |
| 40 chunks (160帧) 无淘汰 | **OOM** |
| 安全 max_cache_tokens | ~100,000 tokens (~3.4 GB cache) |

### 执行指引

- **逐个实验执行**: A → B → C，每个实验独立
- **反复迭代**: 如果实验失败或结果不符合预期，请阅读报告输出、分析原因、修复代码，然后重新运行，直至通过
- **每个实验都有"通过标准"**: 见各实验末尾的 ✅ 判定条件
- **报告文件**: 每个实验会自动生成 `_report.txt`，务必查看完整内容

---

## 实验 A：sink_size 自动检测验证

### 目标

验证 `EvictionConfig(sink_size=0)` 的自动检测机制：
1. 首 chunk 追加后，`effective_sink_size` 等于实际 cache 长度
2. 不同分辨率/帧数组合下，sink 值不同且合理
3. 后续 chunk 的 `update_chunk_stats()` 正确记录平均 token 数

### 原理

sink_size 不能硬编码（如旧版的 128），因为：
- 首 chunk 包含 system prompt (~49 text tokens) + 首帧视觉 token
- 1920×1080 4帧/chunk ≈ 5,438 tokens; 2帧/chunk ≈ 2,750; 640×480 会更少
- 128 远小于任何合理的首 chunk 大小，会错误地淘汰首帧中的大部分视觉 token

### 步骤

请在 `/root/autodl-tmp/diploma_project/temporal_encoding/` 目录下创建 `test_eviction_exp_a.py`:

```python
"""
实验 A: sink_size 自动检测验证

验证:
  1) 首 chunk 后 effective_sink_size = 实际 cache 长度
  2) 不同 chunk 帧数下 sink 值变化合理
  3) update_chunk_stats() 正确记录后续 chunk 平均 token 数
  4) window_size 自动计算 = max_cache_tokens - sink_size
"""
import os
import sys
import gc
import time
import torch
from datetime import datetime
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import StreamQwenModel, VideoStreamingInference, EvictionConfig

MODEL_PATH = os.environ.get(
    "QWEN_MODEL_PATH",
    "/root/autodl-tmp/Qwen/Qwen2___5-VL-3B-Instruct",
)
REPORT_PATH = os.environ.get(
    "REPORT_PATH",
    "/root/autodl-tmp/diploma_project/temporal_encoding/test_eviction_exp_a_report.txt",
)

# 测试不同的 chunk 帧数
CHUNK_FRAME_CONFIGS = [2, 4]
# 追加 chunk 数 (足够验证自动检测, 不需要太多)
NUM_CHUNKS = 5
# 安全的 max_cache_tokens (足够大, 本实验不触发淘汰)
MAX_CACHE_TOKENS = 100_000


class TeeWriter:
    def __init__(self, *writers):
        self._writers = writers
    def write(self, text):
        for w in self._writers:
            w.write(text)
        self.flush()
    def flush(self):
        for w in self._writers:
            w.flush()


def get_vram_gb():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return {
            "allocated": round(torch.cuda.memory_allocated() / (1024 ** 3), 3),
            "reserved": round(torch.cuda.memory_reserved() / (1024 ** 3), 3),
        }
    return {}


def create_test_frames(n_frames, width=1920, height=1080):
    """创建测试帧 (纯色渐变, 模拟真实分辨率)。"""
    frames = []
    for i in range(n_frames):
        # 不同帧用不同颜色, 便于区分
        r = int(255 * i / max(n_frames - 1, 1))
        img = Image.new("RGB", (width, height), (r, 128, 255 - r))
        frames.append(img)
    return frames


def test_sink_detection(model, processor, device, chunk_frames, report_lines):
    """测试指定 chunk_frames 下的 sink 自动检测。"""
    report_lines.append(f"\n{'='*60}")
    report_lines.append(f"Testing: chunk_frames={chunk_frames}, 1920×1080")
    report_lines.append(f"{'='*60}")

    config = EvictionConfig(
        max_cache_tokens=MAX_CACHE_TOKENS,
        sink_size=0,     # 自动检测
        window_size=0,   # 自动计算
    )
    engine = VideoStreamingInference(
        model, processor, device, eviction_config=config
    )

    evictor = engine.cache_manager.evictor

    # 验证初始状态: sink 未检测
    assert not evictor._first_chunk_recorded, "首 chunk 前不应已记录"
    report_lines.append(f"  [Before] first_chunk_recorded = False ✅")

    cache_lens = []
    for i in range(NUM_CHUNKS):
        frames = create_test_frames(chunk_frames, 1920, 1080)
        result = engine.append_video_chunk(frames, fps=2.0)
        cache_len = engine.cache_manager.get_seq_length()
        cache_lens.append(cache_len)

        if i == 0:
            # 首 chunk 后验证
            assert evictor._first_chunk_recorded, "首 chunk 后应已记录"
            sink = evictor.effective_sink_size
            window = evictor.effective_window_size
            report_lines.append(f"  [Chunk 0] cache_len = {cache_len}")
            report_lines.append(f"  [Chunk 0] effective_sink_size = {sink}")
            report_lines.append(f"  [Chunk 0] effective_window_size = {window}")
            report_lines.append(f"  [Chunk 0] sink + window = {sink + window} (should ≤ {MAX_CACHE_TOKENS})")

            # 核心断言: sink = 首 chunk cache 长度
            assert sink == cache_len, f"sink ({sink}) != cache_len ({cache_len})"
            report_lines.append(f"  [Chunk 0] ✅ sink == cache_len")

            # window 自动计算
            assert window == MAX_CACHE_TOKENS - sink, \
                f"window ({window}) != max - sink ({MAX_CACHE_TOKENS - sink})"
            report_lines.append(f"  [Chunk 0] ✅ window == max_cache_tokens - sink")
        else:
            # 后续 chunk: 验证 chunk 统计
            avg = evictor._avg_chunk_tokens
            report_lines.append(
                f"  [Chunk {i}] cache_len = {cache_len}, "
                f"avg_chunk_tokens = {avg:.0f}"
            )

    # 计算实际每 chunk token 数 (非首 chunk)
    per_chunk = []
    for j in range(1, len(cache_lens)):
        per_chunk.append(cache_lens[j] - cache_lens[j - 1])

    if per_chunk:
        actual_avg = sum(per_chunk) / len(per_chunk)
        recorded_avg = evictor._avg_chunk_tokens
        report_lines.append(f"  Actual per-chunk tokens: {per_chunk}")
        report_lines.append(f"  Actual average: {actual_avg:.0f}")
        report_lines.append(f"  Recorded average: {recorded_avg:.0f}")
        # 允许小误差 (浮点运行平均)
        assert abs(recorded_avg - actual_avg) < 10, \
            f"avg mismatch: recorded={recorded_avg:.0f} vs actual={actual_avg:.0f}"
        report_lines.append(f"  ✅ Average chunk tokens match")

    # 清理
    del engine
    gc.collect()
    torch.cuda.empty_cache()

    return cache_lens[0]  # 返回首 chunk 的 sink 值


def main():
    report_dir = os.path.dirname(REPORT_PATH)
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        tee = TeeWriter(sys.stdout, f)
        original_stdout, original_stderr = sys.stdout, sys.stderr
        sys.stdout = tee
        sys.stderr = tee

        try:
            print("=" * 70)
            print("EXPERIMENT A: sink_size Auto-Detection Verification")
            print("=" * 70)
            print(f"Report time: {datetime.now().isoformat(timespec='seconds')}")
            print(f"max_cache_tokens = {MAX_CACHE_TOKENS}")
            print(f"Chunk frame configs to test: {CHUNK_FRAME_CONFIGS}")
            print()

            # 加载模型
            print("[1] Loading model...")
            from transformers import AutoProcessor
            device = "cuda"
            dtype = torch.bfloat16
            processor = AutoProcessor.from_pretrained(MODEL_PATH)
            model = StreamQwenModel.from_pretrained(
                MODEL_PATH, torch_dtype=dtype
            ).to(device)
            model.eval()
            print(f"  VRAM after load: {get_vram_gb()}")
            print()

            # 对每种 chunk 帧数测试
            results = {}
            report_lines = []
            for cf in CHUNK_FRAME_CONFIGS:
                sink_val = test_sink_detection(
                    model, processor, device, cf, report_lines
                )
                results[cf] = sink_val

            # 打印收集的报告
            for line in report_lines:
                print(line)

            # 总结
            print()
            print("=" * 70)
            print("SUMMARY")
            print("=" * 70)
            for cf, sink in results.items():
                print(f"  chunk_frames={cf}: sink_size = {sink} tokens")

            # 验证: 不同帧数 → 不同 sink
            sinks = list(results.values())
            if len(set(sinks)) == len(sinks):
                print(f"  ✅ 不同 chunk_frames 产生不同 sink_size")
            else:
                print(f"  ⚠️ 部分 chunk_frames 产生相同 sink_size (可能帧数差异不够大)")

            # 验证: sink 远大于旧版硬编码的 128
            for cf, sink in results.items():
                if sink > 128:
                    print(f"  ✅ chunk_frames={cf}: sink={sink} >> 128 (旧版硬编码值)")
                else:
                    print(f"  ❌ chunk_frames={cf}: sink={sink} ≤ 128, 不合理!")

            print()
            print("✅ EXPERIMENT A COMPLETE")

        except Exception as e:
            print(f"\n❌ EXPERIMENT A FAILED: {e}")
            import traceback
            traceback.print_exc()

        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr

    print(f"\nReport saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
```

### 运行命令

```bash
cd /root/autodl-tmp/diploma_project/temporal_encoding
python test_eviction_exp_a.py
```

### ✅ 通过标准

1. `effective_sink_size == 首 chunk cache_len` (首 chunk 后自动检测准确)
2. `effective_window_size == max_cache_tokens - sink_size` (自动计算正确)
3. `avg_chunk_tokens` 与实际增量吻合 (误差 < 10)
4. `chunk_frames=2` 和 `chunk_frames=4` 产生不同的 sink_size
5. 所有 sink_size 远大于 128 (旧版硬编码值)

### ❌ 如果失败

- **`sink != cache_len`**: 检查 `video_stream_inference.py` 中 `set_first_chunk_info()` 的调用时机是否在 forward 之后
- **`avg 不匹配`**: 检查 `update_chunk_stats()` 是否正确计算了 `cache_len_after - _prev_cache_len`
- **`sink ≤ 128`**: 测试帧可能分辨率过低, 或 ViT 编码异常 — 检查 ViT 输出 token 数
- **修复后重新运行**, 直至所有断言通过

---

## 实验 B：OOM-Free 长视频处理

### 目标

用 Level 1 淘汰策略处理完整 `1.mp4` 视频，验证：
1. 显存不持续增长, 不 OOM
2. `cache_len` 在达到 `max_cache_tokens` 后保持稳定
3. 淘汰统计数据正确 (总淘汰次数、token 数)
4. 最终 `ask()` 仍可正常回答

### 背景

test_step10 实测表明: **无淘汰时 1920×1080 最多 ~120 帧 (30 chunks) 即达到 22.89 GB reserved, 40 chunks OOM**。
1.mp4 时长约 200s, 以 fps=2 采样 → ~400 帧 → ~100 chunks (4帧/chunk) 或 ~200 chunks (2帧/chunk)。
无淘汰绝不可能处理完。

### 步骤

请在 `/root/autodl-tmp/diploma_project/temporal_encoding/` 目录下创建 `test_eviction_exp_b.py`:

```python
"""
实验 B: OOM-Free 长视频处理

验证:
  1) 启用 Level 1 KV Cache 淘汰 (Sink + Window, 全自动参数)
  2) 以 4 帧/chunk、fps=2 的方式逐段编码整个 1.mp4
  3) 显存保持稳定，不 OOM
  4) cache_len 在触发淘汰后保持 ≤ max_cache_tokens
  5) 最后提一个问题验证 cache 可用性
"""
import os
import sys
import gc
import time
import torch
from datetime import datetime
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import StreamQwenModel, VideoStreamingInference, EvictionConfig

MODEL_PATH = os.environ.get(
    "QWEN_MODEL_PATH",
    "/root/autodl-tmp/Qwen/Qwen2___5-VL-3B-Instruct",
)
VIDEO_PATH = os.environ.get(
    "VIDEO_PATH",
    "/root/autodl-tmp/diploma_project/temporal_encoding/1.mp4",
)
REPORT_PATH = os.environ.get(
    "REPORT_PATH",
    "/root/autodl-tmp/diploma_project/temporal_encoding/test_eviction_exp_b_report.txt",
)

# ── 淘汰参数 (全自动) ──
# ⬇️ 这是需要实验调优的核心超参数。
# 100K=保守(3.4GB cache), 130K=中等(4.5GB), 150K=激进(5.2GB, 接近极限)
# 峰值 cache = max + 1 chunk (~5.4K), 不可超 ~155K (4090 24GB)
# 过小→window不足→近期信息丢失→回答质量下降; 过大→OOM
# 建议从 130K 开始, 若稳定则尝试 150K
MAX_CACHE_TOKENS = 130_000  # 中等配置, ~4.5 GB cache, total ~11.6 GB

# ── 编码参数 ──
CHUNK_FRAMES = 4      # 每次追加 4 帧 (与 test_step10 一致)
SAMPLE_FPS = 2.0      # 采样帧率
PRINT_INTERVAL = 10   # 每 10 个 chunk 打印一次


class TeeWriter:
    def __init__(self, *writers):
        self._writers = writers
    def write(self, text):
        for w in self._writers:
            w.write(text)
        self.flush()
    def flush(self):
        for w in self._writers:
            w.flush()


def get_vram_gb():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return {
            "allocated": round(torch.cuda.memory_allocated() / (1024 ** 3), 3),
            "reserved": round(torch.cuda.memory_reserved() / (1024 ** 3), 3),
            "max_allocated": round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3),
        }
    return {}


def extract_frames_from_video(video_path, fps=2.0):
    """从视频文件中按指定 fps 采样帧。返回 PIL Image 列表。"""
    try:
        import decord
        from decord import VideoReader, cpu
        decord.bridge.set_bridge("native")
        vr = VideoReader(video_path, ctx=cpu(0))
        video_fps = vr.get_avg_fps()
        total_frames = len(vr)
        duration = total_frames / video_fps
        sample_interval = video_fps / fps
        indices = [int(i * sample_interval) for i in range(int(total_frames / sample_interval))]
        indices = [i for i in indices if i < total_frames]
        print(f"  Video: {video_path}")
        print(f"  Duration: {duration:.1f}s, FPS: {video_fps:.1f}, Total frames: {total_frames}")
        print(f"  Sampling at {fps} fps → {len(indices)} frames")
        frames = []
        for idx in indices:
            frame = vr[idx].asnumpy()
            frames.append(Image.fromarray(frame))
        return frames, duration
    except ImportError:
        import cv2
        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps if video_fps > 0 else 0
        sample_interval = video_fps / fps
        indices = [int(i * sample_interval) for i in range(int(total_frames / sample_interval))]
        indices = [i for i in indices if i < total_frames]
        print(f"  Video: {video_path}")
        print(f"  Duration: {duration:.1f}s, FPS: {video_fps:.1f}, Total frames: {total_frames}")
        print(f"  Sampling at {fps} fps → {len(indices)} frames")
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        cap.release()
        return frames, duration


def main():
    report_dir = os.path.dirname(REPORT_PATH)
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        tee = TeeWriter(sys.stdout, f)
        original_stdout, original_stderr = sys.stdout, sys.stderr
        sys.stdout = tee
        sys.stderr = tee

        try:
            print("=" * 70)
            print("EXPERIMENT B: OOM-Free Long Video Processing with KV Cache Eviction")
            print("=" * 70)
            print(f"Report time: {datetime.now().isoformat(timespec='seconds')}")
            print(f"Eviction config: max_cache_tokens={MAX_CACHE_TOKENS}, "
                  f"sink=auto, window=auto")
            print(f"Expected: test_step10 shows OOM at 40 chunks (160 frames) without eviction.")
            print(f"With eviction, should process ALL chunks without OOM.")
            print()

            # ── 0) 检查文件 ──
            if not os.path.exists(MODEL_PATH):
                print(f"❌ Model not found: {MODEL_PATH}")
                return
            if not os.path.exists(VIDEO_PATH):
                print(f"❌ Video not found: {VIDEO_PATH}")
                return

            # ── 1) 加载模型 ──
            print("[1] Loading model...")
            from transformers import AutoProcessor
            device = "cuda"
            dtype = torch.bfloat16
            processor = AutoProcessor.from_pretrained(MODEL_PATH)
            model = StreamQwenModel.from_pretrained(
                MODEL_PATH, torch_dtype=dtype
            ).to(device)
            model.eval()
            vram_model = get_vram_gb()
            print(f"  VRAM after model load: {vram_model}")
            print()

            # ── 2) 提取视频帧 ──
            print("[2] Extracting frames from video...")
            all_frames, duration = extract_frames_from_video(VIDEO_PATH, fps=SAMPLE_FPS)
            total_frame_count = len(all_frames)
            expected_chunks = (total_frame_count + CHUNK_FRAMES - 1) // CHUNK_FRAMES
            print(f"  Total frames extracted: {total_frame_count}")
            print(f"  Expected chunks (4 frames/chunk): {expected_chunks}")
            print(f"  ⚠️ Without eviction, OOM at ~40 chunks ({40*CHUNK_FRAMES} frames).")
            print(f"  With eviction (max={MAX_CACHE_TOKENS}), should handle all {expected_chunks} chunks.")
            print()

            # ── 3) 创建引擎 (启用 Level 1 淘汰, 全自动参数) ──
            print("[3] Creating streaming inference engine with eviction...")
            eviction_config = EvictionConfig(
                max_cache_tokens=MAX_CACHE_TOKENS,
                # sink_size=0  → 自动检测首 chunk
                # window_size=0 → 自动计算
            )
            engine = VideoStreamingInference(
                model, processor, device, eviction_config=eviction_config
            )
            print()

            # ── 4) 逐 chunk 编码 ──
            print("[4] Encoding video chunks...")
            t_start = time.time()
            vram_history = []
            cache_history = []
            chunk_count = 0
            first_eviction_chunk = None

            for i in range(0, total_frame_count, CHUNK_FRAMES):
                chunk = all_frames[i : i + CHUNK_FRAMES]
                if len(chunk) == 0:
                    continue
                # 补齐偶数帧 (temporal_patch_size=2)
                if len(chunk) % 2 != 0:
                    chunk.append(chunk[-1])

                result = engine.append_video_chunk(chunk, fps=SAMPLE_FPS)
                chunk_count += 1

                info = engine.get_cache_info()
                cache_len = info["cache_seq_length"]

                # 记录首次淘汰
                if "eviction_stats" in info:
                    es = info["eviction_stats"]
                    if es.get("total_evictions", 0) > 0 and first_eviction_chunk is None:
                        first_eviction_chunk = chunk_count

                if chunk_count % PRINT_INTERVAL == 0 or chunk_count == 1:
                    vram = get_vram_gb()
                    vram_history.append({
                        "chunk": chunk_count,
                        "cache_len": cache_len,
                        "vram_alloc": vram.get("allocated", 0),
                        "vram_reserved": vram.get("reserved", 0),
                    })
                    cache_history.append(cache_len)

                    eviction_str = ""
                    if "eviction_stats" in info:
                        es = info["eviction_stats"]
                        eviction_str = (
                            f", evictions={es.get('total_evictions', 0)}, "
                            f"evicted={es.get('total_tokens_evicted', 0)}"
                        )

                    print(
                        f"  Chunk {chunk_count:>4d}/{expected_chunks}: "
                        f"cache_len={cache_len:>6d}, "
                        f"mem={info.get('cache_memory_gb', 0):.3f} GB, "
                        f"VRAM={vram.get('allocated', 0):.2f}/{vram.get('reserved', 0):.2f} GB"
                        f"{eviction_str}"
                    )

            t_encode = time.time() - t_start

            # 汇总
            print(f"\n  ✅ Encoding completed: {chunk_count} chunks, "
                  f"{total_frame_count} frames in {t_encode:.1f}s")
            final_vram = get_vram_gb()
            print(f"  Final VRAM: {final_vram}")
            if first_eviction_chunk:
                print(f"  First eviction at chunk: {first_eviction_chunk}")

            # 获取 evictor 状态
            evictor = engine.cache_manager.evictor
            if evictor:
                print(f"  Effective sink_size: {evictor.effective_sink_size}")
                print(f"  Effective window_size: {evictor.effective_window_size}")
                print(f"  Avg chunk tokens: {evictor._avg_chunk_tokens:.0f}")
            print()

            # ── 5) 验证 ask 仍可用 ──
            print("[5] Verification: asking a question...")
            final_info = engine.get_cache_info()
            print(f"  Pre-ask cache: len={final_info['cache_seq_length']}, "
                  f"mem={final_info.get('cache_memory_gb', 0):.3f} GB")

            answer, metrics = engine.ask(
                "Briefly describe what you saw in the entire video.",
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
            )
            print(f"  Answer: {answer}")
            print(f"  TTFT: {metrics['ttft']:.3f}s")

            # 验证 ask 后 cache 恢复 (snapshot/restore)
            post_ask_info = engine.get_cache_info()
            print(f"  Post-ask cache: len={post_ask_info['cache_seq_length']}")
            assert post_ask_info['cache_seq_length'] == final_info['cache_seq_length'], \
                "ask() 后 cache 长度应恢复 (snapshot/restore)"
            print(f"  ✅ Cache restored after ask()")
            print()

            # ── 6) 总结 ──
            print("=" * 70)
            print("SUMMARY")
            print("=" * 70)
            print(f"  Video: {VIDEO_PATH} ({duration:.0f}s)")
            print(f"  Total frames: {total_frame_count}")
            print(f"  Total chunks: {chunk_count}")
            print(f"  Encoding time: {t_encode:.1f}s "
                  f"({total_frame_count / t_encode:.2f} frames/sec)")
            print(f"  Max cache tokens: {MAX_CACHE_TOKENS}")
            print(f"  Final cache_len: {final_info['cache_seq_length']}")
            print(f"  Final VRAM: allocated={final_vram.get('allocated', 0):.2f} GB, "
                  f"reserved={final_vram.get('reserved', 0):.2f} GB, "
                  f"max_allocated={final_vram.get('max_allocated', 0):.2f} GB")

            if "eviction_stats" in final_info:
                es = final_info["eviction_stats"]
                print(f"  Total evictions: {es.get('total_evictions', 0)}")
                print(f"  Total tokens evicted: {es.get('total_tokens_evicted', 0)}")

            # 通过判定
            print()
            print("─" * 70)
            print("PASS/FAIL CRITERIA:")
            all_pass = True

            # 1) 没 OOM (走到这里说明没 OOM)
            print(f"  ✅ [P1] No OOM — processed all {chunk_count} chunks "
                  f"(test_step10 OOM at 40 chunks without eviction)")

            # 2) cache_len ≤ max_cache_tokens
            if final_info['cache_seq_length'] <= MAX_CACHE_TOKENS:
                print(f"  ✅ [P2] cache_len ({final_info['cache_seq_length']}) "
                      f"≤ max ({MAX_CACHE_TOKENS})")
            else:
                print(f"  ❌ [P2] cache_len ({final_info['cache_seq_length']}) "
                      f"> max ({MAX_CACHE_TOKENS})")
                all_pass = False

            # 3) 有淘汰发生
            if "eviction_stats" in final_info:
                es = final_info["eviction_stats"]
                if es.get("total_evictions", 0) > 0:
                    print(f"  ✅ [P3] Eviction occurred "
                          f"({es['total_evictions']} times, "
                          f"{es['total_tokens_evicted']} tokens)")
                else:
                    print(f"  ❌ [P3] No eviction occurred — config may not have been applied")
                    all_pass = False

            # 4) VRAM 未超 23 GB
            max_alloc = final_vram.get("max_allocated", 0)
            if max_alloc < 23.0:
                print(f"  ✅ [P4] Max VRAM allocated ({max_alloc:.2f} GB) < 23 GB")
            else:
                print(f"  ⚠️ [P4] Max VRAM allocated ({max_alloc:.2f} GB) ≥ 23 GB")
                all_pass = False

            # 5) ask 正常
            if answer and len(answer) > 5:
                print(f"  ✅ [P5] ask() returned valid answer ({len(answer)} chars)")
            else:
                print(f"  ❌ [P5] ask() returned empty/short answer")
                all_pass = False

            print()
            if all_pass:
                print("🎉 EXPERIMENT B: ALL PASSED")
            else:
                print("⚠️ EXPERIMENT B: SOME CHECKS FAILED — see above")

        except torch.cuda.OutOfMemoryError:
            print(f"\n❌ EXPERIMENT B FAILED: CUDA OOM!")
            print(f"  This means eviction did not prevent OOM.")
            print(f"  Possible causes:")
            print(f"    1) Eviction not triggered — check EvictionConfig")
            print(f"    2) max_cache_tokens too large — try 50,000")
            print(f"    3) torch reserved memory fragmentation")
            import traceback
            traceback.print_exc()

        except Exception as e:
            print(f"\n❌ EXPERIMENT B FAILED: {e}")
            import traceback
            traceback.print_exc()

        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr

    print(f"\nReport saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
```

### 运行命令

```bash
cd /root/autodl-tmp/diploma_project/temporal_encoding
python test_eviction_exp_b.py
```

### ✅ 通过标准

1. **[P1]** 处理全部 chunk, 不 OOM (无淘汰时 40 chunks 就 OOM)
2. **[P2]** 最终 `cache_len ≤ max_cache_tokens`
3. **[P3]** 淘汰次数 > 0 且淘汰 token 数 > 0
4. **[P4]** VRAM max_allocated < 23 GB
5. **[P5]** `ask()` 返回有效回答

### ❌ 如果失败

- **CUDA OOM**: 降低 `MAX_CACHE_TOKENS` (如 100,000 或 50,000), 或降低 `SAMPLE_FPS` (如 1.0)。也可能是 `torch.cuda.memory_reserved` 碎片化 — 尝试每 N chunk 调用 `torch.cuda.empty_cache()`
- **cache_len 不下降**: 淘汰未触发 — 检查 `eviction_interval`、`should_evict()` 逻辑、`video_stream_inference.py` 中的 `_chunk_counter`
- **ask() 失败**: snapshot/restore 与淘汰不兼容 — 检查 `cache_manager.py` 的 snapshot 是否保存了 tracker
- **修复后重新运行**, 直至所有 P1-P5 通过

### 🔧 max_cache_tokens 调优流程 (实验 B 通过后执行)

实验 B 的完整目标不仅是“不 OOM”，还要找到 **充分利用 24GB 显存的最优 max_cache_tokens**。

window = max_cache_tokens - sink (≈ 5.4K)，所以 max 越大 → window 越大 → 保留更多近期视频帧 → 回答质量更好。
但 max 过大 → 峰值 cache (max + 1 chunk) 超过 CUDA 安全线 → OOM。

**步骤:**

1. 先用 `MAX_CACHE_TOKENS = 130_000` 跑完实验 B, 记录 VRAM max_reserved
2. 若 max_reserved < 21 GB: 提高到 `150_000` 重跑
3. 若 max_reserved 21~23 GB: 当前值即为最优
4. 若 OOM: 降低到 `100_000` 重跑
5. 用最终确定的值更新 `kv_cache_eviction.py` 中的 `max_cache_tokens` 默认值

```python
# 参考配置梯度:
MAX_CACHE_TOKENS = 100_000  # 保守: ~3.4 GB cache, total ~10.5 GB
MAX_CACHE_TOKENS = 130_000  # 中等: ~4.5 GB cache, total ~11.6 GB (推荐起点)
MAX_CACHE_TOKENS = 150_000  # 激进: ~5.2 GB cache, total ~12.3 GB (接近极限)
MAX_CACHE_TOKENS = 50_000   # 安全网: ~1.7 GB cache, total ~8.8 GB (仅在 OOM 时用)
```

**关键输出指标**:
- `Final VRAM reserved`: 尽量接近 22-23 GB (充分利用)
- `effective_window_size / avg_chunk_tokens`: = 窗口内能保留多少个 chunk, 越多越好
- `ask() 回答质量`: 在实验 C 中比较不同 max 值的歌词提取效果

---

## 实验 C：滑窗逐段处理 + 周期性自动提问

### 目标

将视频分段处理, 每编码 N 个 chunk 后自动提问一次, 验证:
1. 淘汰不影响 `ask()` 的 snapshot/restore 机制
2. 滑窗覆盖不同视频段, 每段都能提取有效信息
3. 最终能拼接出视频中的歌词/字幕内容
4. 全程不 OOM

### 步骤

请在 `/root/autodl-tmp/diploma_project/temporal_encoding/` 目录下创建 `test_eviction_exp_c.py`:

```python
"""
实验 C: 滑窗逐段 + 周期性提问，提取视频歌词/字幕

验证:
  1) 每编码 ASK_INTERVAL 个 chunk 后自动提问一次
  2) 淘汰不影响 ask() 的 snapshot/restore
  3) 收集所有回答，去重后拼接为完整歌词
  4) 全程不 OOM
"""
import os
import sys
import gc
import time
import torch
from datetime import datetime
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import StreamQwenModel, VideoStreamingInference, EvictionConfig

MODEL_PATH = os.environ.get(
    "QWEN_MODEL_PATH",
    "/root/autodl-tmp/Qwen/Qwen2___5-VL-3B-Instruct",
)
VIDEO_PATH = os.environ.get(
    "VIDEO_PATH",
    "/root/autodl-tmp/diploma_project/temporal_encoding/1.mp4",
)
REPORT_PATH = os.environ.get(
    "REPORT_PATH",
    "/root/autodl-tmp/diploma_project/temporal_encoding/test_eviction_exp_c_report.txt",
)

# ── 淘汰参数 (全自动) ──
# ⬇️ 应与实验 B 最终调优值一致; 更大 → 窗口更大 → 近期帧更多 → 回答更好
MAX_CACHE_TOKENS = 130_000  # 与实验 B 调优后保持一致

# ── 编码参数 ──
CHUNK_FRAMES = 4
SAMPLE_FPS = 2.0

# ── 提问参数 ──
ASK_INTERVAL = 25       # 每 25 个 chunk (~50 秒视频) 提问一次
MAX_NEW_TOKENS = 200

QUESTION = (
    "Read all text, lyrics, subtitles, or captions currently visible on screen. "
    "Output them verbatim. If there is no text, say 'No text visible'. "
    "Do NOT repeat previously mentioned text."
)


class TeeWriter:
    def __init__(self, *writers):
        self._writers = writers
    def write(self, text):
        for w in self._writers:
            w.write(text)
        self.flush()
    def flush(self):
        for w in self._writers:
            w.flush()


def get_vram_gb():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return {
            "allocated": round(torch.cuda.memory_allocated() / (1024 ** 3), 3),
            "reserved": round(torch.cuda.memory_reserved() / (1024 ** 3), 3),
        }
    return {}


def extract_frames_from_video(video_path, fps=2.0):
    """从视频中按指定 fps 采样帧。"""
    try:
        import decord
        from decord import VideoReader, cpu
        decord.bridge.set_bridge("native")
        vr = VideoReader(video_path, ctx=cpu(0))
        video_fps = vr.get_avg_fps()
        total_frames = len(vr)
        duration = total_frames / video_fps
        sample_interval = video_fps / fps
        indices = [int(i * sample_interval) for i in range(int(total_frames / sample_interval))]
        indices = [i for i in indices if i < total_frames]
        print(f"  Video: {video_path}")
        print(f"  Duration: {duration:.1f}s, Total: {total_frames} frames")
        print(f"  Sampling at {fps} fps → {len(indices)} frames")
        frames = []
        for idx in indices:
            frame = vr[idx].asnumpy()
            frames.append(Image.fromarray(frame))
        return frames, duration
    except ImportError:
        import cv2
        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps if video_fps > 0 else 0
        sample_interval = video_fps / fps
        indices = [int(i * sample_interval) for i in range(int(total_frames / sample_interval))]
        indices = [i for i in indices if i < total_frames]
        print(f"  Video: {video_path}")
        print(f"  Duration: {duration:.1f}s, Total: {total_frames} frames")
        print(f"  Sampling at {fps} fps → {len(indices)} frames")
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        cap.release()
        return frames, duration


def main():
    report_dir = os.path.dirname(REPORT_PATH)
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        tee = TeeWriter(sys.stdout, f)
        original_stdout, original_stderr = sys.stdout, sys.stderr
        sys.stdout = tee
        sys.stderr = tee

        try:
            print("=" * 70)
            print("EXPERIMENT C: Sliding Window + Periodic Auto-Questioning")
            print("=" * 70)
            print(f"Report time: {datetime.now().isoformat(timespec='seconds')}")
            print(f"Ask interval: every {ASK_INTERVAL} chunks (~{ASK_INTERVAL * CHUNK_FRAMES / SAMPLE_FPS:.0f}s)")
            print(f"Eviction: max_cache_tokens={MAX_CACHE_TOKENS}, sink/window=auto")
            print()

            # ── 0) 检查 ──
            if not os.path.exists(MODEL_PATH):
                print(f"❌ Model not found: {MODEL_PATH}")
                return
            if not os.path.exists(VIDEO_PATH):
                print(f"❌ Video not found: {VIDEO_PATH}")
                return

            # ── 1) 加载模型 ──
            print("[1] Loading model...")
            from transformers import AutoProcessor
            device = "cuda"
            dtype = torch.bfloat16
            processor = AutoProcessor.from_pretrained(MODEL_PATH)
            model = StreamQwenModel.from_pretrained(
                MODEL_PATH, torch_dtype=dtype
            ).to(device)
            model.eval()
            print(f"  VRAM: {get_vram_gb()}")
            print()

            # ── 2) 提取帧 ──
            print("[2] Extracting frames...")
            all_frames, duration = extract_frames_from_video(VIDEO_PATH, fps=SAMPLE_FPS)
            total_frame_count = len(all_frames)
            print()

            # ── 3) 创建引擎 ──
            print("[3] Creating engine with eviction...")
            eviction_config = EvictionConfig(
                max_cache_tokens=MAX_CACHE_TOKENS,
            )
            engine = VideoStreamingInference(
                model, processor, device, eviction_config=eviction_config
            )
            print()

            # ── 4) 编码 + 周期性提问 ──
            print("[4] Encoding with periodic questioning...")
            all_answers = []
            chunk_count = 0
            t_start = time.time()

            for i in range(0, total_frame_count, CHUNK_FRAMES):
                chunk = all_frames[i : i + CHUNK_FRAMES]
                if len(chunk) == 0:
                    continue
                if len(chunk) % 2 != 0:
                    chunk.append(chunk[-1])

                engine.append_video_chunk(chunk, fps=SAMPLE_FPS)
                chunk_count += 1

                # 周期性提问
                if chunk_count % ASK_INTERVAL == 0:
                    time_pos = (i + CHUNK_FRAMES) / SAMPLE_FPS
                    print(f"\n  ─── Ask at chunk {chunk_count} "
                          f"(video ~{time_pos:.0f}s / {duration:.0f}s) ───")

                    info = engine.get_cache_info()
                    vram = get_vram_gb()
                    eviction_str = ""
                    if "eviction_stats" in info:
                        es = info["eviction_stats"]
                        eviction_str = f", evictions={es.get('total_evictions', 0)}"
                    print(f"  Cache: len={info['cache_seq_length']}, "
                          f"mem={info.get('cache_memory_gb', 0):.3f} GB, "
                          f"VRAM={vram.get('allocated', 0):.2f} GB"
                          f"{eviction_str}")

                    # 记录 ask 前 cache 长度
                    pre_ask_len = info['cache_seq_length']

                    answer, metrics = engine.ask(
                        QUESTION,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=True,
                        temperature=0.3,
                    )

                    # 验证 snapshot/restore
                    post_ask_len = engine.cache_manager.get_seq_length()
                    restored = (post_ask_len == pre_ask_len)

                    all_answers.append({
                        "chunk": chunk_count,
                        "time_pos": f"~{time_pos:.0f}s",
                        "answer": answer.strip(),
                        "ttft": metrics["ttft"],
                        "cache_restored": restored,
                    })
                    print(f"  Answer: {answer.strip()[:150]}...")
                    print(f"  TTFT: {metrics['ttft']:.3f}s, "
                          f"Cache restored: {'✅' if restored else '❌'}")

            t_total = time.time() - t_start

            # 最后一段如果还没问过，补一次
            if chunk_count % ASK_INTERVAL != 0:
                print(f"\n  ─── Final ask at chunk {chunk_count} ───")
                pre_ask_len = engine.cache_manager.get_seq_length()
                answer, metrics = engine.ask(
                    QUESTION,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=0.3,
                )
                post_ask_len = engine.cache_manager.get_seq_length()
                restored = (post_ask_len == pre_ask_len)
                all_answers.append({
                    "chunk": chunk_count,
                    "time_pos": f"~{total_frame_count / SAMPLE_FPS:.0f}s",
                    "answer": answer.strip(),
                    "ttft": metrics["ttft"],
                    "cache_restored": restored,
                })
                print(f"  Answer: {answer.strip()[:150]}...")

            print(f"\n  ✅ Done: {chunk_count} chunks, {len(all_answers)} questions asked")
            print(f"  Total time: {t_total:.1f}s")
            print()

            # ── 5) 汇总所有歌词 ──
            print("=" * 70)
            print("ALL COLLECTED LYRICS / SUBTITLES")
            print("=" * 70)

            seen_lines = set()
            unique_lyrics = []

            for entry in all_answers:
                print(f"\n[{entry['time_pos']}] (chunk {entry['chunk']}):")
                print(f"  {entry['answer']}")

                lines = entry["answer"].split("\n")
                for line in lines:
                    line_clean = line.strip().lower()
                    if (
                        line_clean
                        and line_clean not in seen_lines
                        and "no text" not in line_clean
                        and "no lyrics" not in line_clean
                        and "no subtitle" not in line_clean
                        and "no caption" not in line_clean
                        and "no visible" not in line_clean
                    ):
                        seen_lines.add(line_clean)
                        unique_lyrics.append(line.strip())

            print()
            print("=" * 70)
            print("DEDUPLICATED LYRICS (all unique lines)")
            print("=" * 70)
            for line in unique_lyrics:
                print(f"  {line}")
            print(f"\n  Total unique lines: {len(unique_lyrics)}")

            # ── 6) 总结 + 通过判定 ──
            print()
            print("=" * 70)
            print("SUMMARY & PASS/FAIL")
            print("=" * 70)
            final_info = engine.get_cache_info()
            print(f"  Video duration: {duration:.0f}s")
            print(f"  Total chunks: {chunk_count}")
            print(f"  Questions asked: {len(all_answers)}")
            print(f"  Unique lyric lines: {len(unique_lyrics)}")
            print(f"  Final cache_len: {final_info['cache_seq_length']}")
            print(f"  Total time: {t_total:.1f}s")

            if "eviction_stats" in final_info:
                es = final_info["eviction_stats"]
                print(f"  Total evictions: {es.get('total_evictions', 0)}")
                print(f"  Total tokens evicted: {es.get('total_tokens_evicted', 0)}")

            avg_ttft = sum(a["ttft"] for a in all_answers) / max(len(all_answers), 1)
            print(f"  Average TTFT: {avg_ttft:.3f}s")

            # 通过判定
            print()
            all_pass = True

            # C1: 不 OOM
            print(f"  ✅ [C1] No OOM — processed all {chunk_count} chunks")

            # C2: 所有 ask 后 cache 恢复
            all_restored = all(a["cache_restored"] for a in all_answers)
            if all_restored:
                print(f"  ✅ [C2] All {len(all_answers)} ask() calls "
                      f"correctly restored cache (snapshot/restore)")
            else:
                failed = [a for a in all_answers if not a["cache_restored"]]
                print(f"  ❌ [C2] {len(failed)} ask() calls did not restore cache!")
                all_pass = False

            # C3: 至少 N 次提问有非空回答
            non_empty = [
                a for a in all_answers
                if a["answer"]
                and "no text" not in a["answer"].lower()
                and "no visible" not in a["answer"].lower()
            ]
            if len(non_empty) >= 1:
                print(f"  ✅ [C3] {len(non_empty)}/{len(all_answers)} answers "
                      f"contained text/lyrics")
            else:
                print(f"  ⚠️ [C3] All answers were empty/no text — "
                      f"video may not contain visible text")

            # C4: 提取到歌词行
            if len(unique_lyrics) >= 1:
                print(f"  ✅ [C4] Extracted {len(unique_lyrics)} unique lyric lines")
            else:
                print(f"  ⚠️ [C4] No lyrics extracted — may be expected if video has no text")

            # C5: TTFT 合理 (< 10s)
            if avg_ttft < 10.0:
                print(f"  ✅ [C5] Average TTFT ({avg_ttft:.3f}s) < 10s")
            else:
                print(f"  ⚠️ [C5] Average TTFT ({avg_ttft:.3f}s) ≥ 10s — may be slow")

            print()
            if all_pass:
                print("🎉 EXPERIMENT C: ALL PASSED")
            else:
                print("⚠️ EXPERIMENT C: SOME CHECKS FAILED — see above")

        except torch.cuda.OutOfMemoryError:
            print(f"\n❌ EXPERIMENT C FAILED: CUDA OOM!")
            import traceback
            traceback.print_exc()

        except Exception as e:
            print(f"\n❌ EXPERIMENT C FAILED: {e}")
            import traceback
            traceback.print_exc()

        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr

    print(f"\nReport saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
```

### 运行命令

```bash
cd /root/autodl-tmp/diploma_project/temporal_encoding
python test_eviction_exp_c.py
```

### ✅ 通过标准

1. **[C1]** 全程不 OOM
2. **[C2]** 所有 `ask()` 调用后 cache 正确恢复 (snapshot/restore 在淘汰后仍工作)
3. **[C3]** 至少有部分提问返回了有效文本内容 (取决于视频)
4. **[C4]** 去重后提取到歌词/字幕行 (取决于视频)
5. **[C5]** 平均 TTFT < 10s

### ❌ 如果失败

- **OOM**: 与实验 B 相同的排查步骤
- **Cache 不恢复**: `cache_manager.py` 的 `snapshot()` 可能未包含 tracker 的 eviction 相关状态
- **所有回答为空**: 可能视频确实无文本, 或 `QUESTION` prompt 需要调整, 或 `temperature` 太低
- **修复后重新运行**

### 参数调优建议

```python
# 更频繁提问
ASK_INTERVAL = 15      # 每 15 chunk (~30s)

# 更低温度, 更精确
temperature = 0.1

# 如果歌词为中文
QUESTION = "读取屏幕上当前可见的所有文字、歌词、字幕。逐字输出。如果没有文字，说'无文字'。"
```

---

## 理论风险与 Copilot 探索指令

### ⚠️ 理论风险说明

本实验采用“原生 Qwen2.5-VL”模型（未针对大跨度 position gap/淘汰场景微调），其注意力机制在遇到大规模 KV Cache 淘汰（即 position index 不连续，gap 很大）时，理论上可能出现：

- 注意力分布异常（如 attention collapse、历史帧信息丢失）
- 长时依赖建模能力下降，导致问答/字幕提取等下游任务质量变差

此风险属于模型预训练分布外（OOD）行为，工程上可运行，但推理质量需实验验证。

### 🚦 Copilot 自动探索指令

如实验 C 发现淘汰后模型输出质量明显下降（如字幕/问答内容丢失、无关、重复、极端简化等），Copilot 必须：

1. 自动分析日志与输出，确认是否为 position gap 导致的注意力异常
2. 在**不微调模型**前提下，探索以下工程/推理层解决思路，并逐一实验：
    - **位置重映射**：将新帧 position index 连续化（如重置为 sink+1…sink+N），避免大 gap
    - **降低分辨率/帧率**：减少单帧 token 数，减缓淘汰频率
    - **增大 max_cache_tokens**：权衡显存与历史保留长度
    - **其他推理层技巧**：如分块问答、滑动窗口融合、prompt 工程等
3. 记录每种方案的实验结果，优先推荐无需模型改动即可显著提升质量的方案

如所有工程方案均无效，再建议微调/适配训练。

---

## 三个实验的执行顺序与迭代指引

### 推荐执行顺序

```
实验 A (sink 检测) → 实验 B (OOM-Free) → 实验 C (ask 质量)
```

A 是基础: 如果 sink 检测不对, B/C 的淘汰行为都会异常。
B 是核心: 如果 OOM, C 自然也会 OOM。
C 是最终验证: 淘汰后模型仍然有用。

### 迭代策略

每个实验如果失败, 按以下流程:

1. **阅读报告**: 完整阅读 `_report.txt`, 确认哪个判定条件失败
2. **定位代码**: 根据失败类型, 查看对应文件:
   - sink 相关 → `kv_cache_eviction.py` 的 `set_first_chunk_info()` + `video_stream_inference.py` 的调用时机
   - 淘汰未触发 → `kv_cache_eviction.py` 的 `should_evict()` + `evict()` + `video_stream_inference.py` 的 `_chunk_counter`
   - OOM → `max_cache_tokens` 值, 或 `torch.cuda.empty_cache()` 缺失
   - snapshot/restore → `cache_manager.py` 的 `take_snapshot()` / `restore_snapshot()`
3. **修改代码**: 只改必要的部分
4. **重新运行**: 同一实验, 直至通过
5. **进入下一个实验**

### 调试建议

```python
# 在 video_stream_inference.py 的 append_frame 中加入 verbose 输出:
print(f"  [DEBUG] chunk={self.frame_count}, "
      f"cache_before={prev_len}, cache_after={cache_len_after}, "
      f"sink={evictor.effective_sink_size if evictor else 'N/A'}")
```

### 文件变更清单

以下文件已新增/修改, 请确认均已同步到远程机器:

| 文件 | 状态 | 说明 |
|------|------|------|
| `model/kv_cache_eviction.py` | 🆕 重写 | sink 自动检测, 均匀时序采样, 帧级重要性 |
| `model/cache_manager.py` | ✏️ 修改 | 集成 `set_first_chunk_info()`, `track_tokens(is_new_chunk=)` |
| `model/video_stream_inference.py` | ✏️ 修改 | 首 chunk auto-detect, chunk stats 更新 |
| `model/__init__.py` | ✏️ 已有 | 导出 EvictionConfig, KVCacheEvictor 等 |
| `PROJECT_STRUCTURE_V2.md` | ✏️ 重写 | 修正容量表、参数说明、策略描述 |
| `EVICTION_EXPERIMENT_PROMPT.md` | 🆕 重写 | 本文件: 3 个实验 + 迭代指引 |
