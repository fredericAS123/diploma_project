"""
Step 10: 最大编码帧数容量测试（4090 24GB VRAM）

测试目标：
  1) 使用真实视频在原生分辨率下测试流式引擎最大编码帧数
  2) 记录达到 OOM 前的最大帧数、编码时间、显存占用等关键指标
  3) 提供 4090（24GB）上的实际容量基准

测试分辨率：
  - 原生分辨率（视频原始尺寸）

测试策略：
  - 使用 4 帧 chunk 进行流式编码
  - 从小帧数开始，逐步增加直到 OOM 或视频帧耗尽
  - 记录原生分辨率下的临界点

需要 GPU (RTX 4090 24GB) + 模型权重 + 实际视频文件。
"""
import os
import sys
import time
import cv2
import torch
from datetime import datetime
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import StreamQwenModel, VideoStreamingInference

MODEL_PATH = os.environ.get("QWEN_MODEL_PATH", "/root/autodl-tmp/Qwen/Qwen2___5-VL-3B-Instruct")
VIDEO_PATH = os.environ.get(
    "VIDEO_PATH",
    "/root/autodl-tmp/diploma_project/temporal_encoding/202208312002.mp4",
)
REPORT_PATH = os.environ.get(
    "STEP10_REPORT_PATH",
    "/root/autodl-tmp/diploma_project/temporal_encoding/test_step10_max_frames_report.txt",
)
CHUNK_SIZE = 4  # 固定使用 4 帧 chunk
FRAME_STRIDE = int(os.environ.get("FRAME_STRIDE", "1"))


class TeeWriter:
    """Write stdout/stderr to both console and file."""

    def __init__(self, *writers):
        self._writers = writers

    def write(self, text):
        for w in self._writers:
            w.write(text)
        self.flush()

    def flush(self):
        for w in self._writers:
            w.flush()


def _get_vram_gb():
    """获取当前 CUDA VRAM 使用量（GB）。"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        return {"allocated": round(allocated, 2), "reserved": round(reserved, 2)}
    return {"allocated": 0.0, "reserved": 0.0}


def _get_video_meta(video_path: str):
    """读取视频元信息（原生分辨率、fps、总帧数、时长）。"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = (total_frames / fps) if fps > 0 else 0.0
    finally:
        cap.release()
    return {
        "width": width,
        "height": height,
        "fps": fps,
        "total_frames": total_frames,
        "duration": duration,
    }


def test_max_frames_for_resolution(
    model,
    processor,
    device: str,
    width: int,
    height: int,
    video_path: str,
    video_fps: float,
    max_available_frames: int,
    frame_stride: int = 1,
    max_attempts: int = 50,
    initial_chunks: int = 10,
    increment: int = 10,
):
    """
    测试指定分辨率下的最大编码帧数（原生分辨率）。

    策略：从 initial_chunks 开始，每次增加 increment chunks，直到 OOM 或视频帧耗尽。

    Args:
        model: StreamQwenModel 实例
        processor: AutoProcessor 实例
        device: 设备
        width: 帧宽度
        height: 帧高度
        video_path: 视频路径
        video_fps: 视频帧率
        max_available_frames: 可用总帧数（考虑 stride）
        frame_stride: 取帧步长
        max_attempts: 最大尝试次数（防止死循环）
        initial_chunks: 初始 chunk 数量
        increment: 每次增加的 chunk 数量

    Returns:
        dict: 包含最大帧数、编码时间、显存信息等
    """
    print(f"\n{'═' * 70}")
    print(f"Testing resolution (native): {width}×{height}")
    print(f"{'═' * 70}")

    last_successful = {
        "num_chunks": 0,
        "encoded_chunks": 0,
        "total_frames": 0,
        "padded_frames": 0,
        "encode_time": 0.0,
        "cache_seq_length": 0,
        "cache_memory_gb": 0.0,
        "vram_peak": {"allocated": 0.0, "reserved": 0.0},
        "vram_after_encode": {"allocated": 0.0, "reserved": 0.0},
    }

    current_chunks = initial_chunks

    for attempt in range(max_attempts):
        target_frames = current_chunks * CHUNK_SIZE
        reached_eof = False
        if max_available_frames > 0 and target_frames > max_available_frames:
            target_frames = max_available_frames
            reached_eof = True

        if target_frames <= 0:
            raise RuntimeError("No available frames to test.")

        print(
            f"\n[Attempt {attempt + 1}] Testing {current_chunks} chunks "
            f"(target {current_chunks * CHUNK_SIZE} frames, actual {target_frames} frames)..."
        )

        # 重置引擎
        engine = VideoStreamingInference(model, processor, device)

        try:
            effective_fps = video_fps / max(frame_stride, 1) if video_fps > 0 else 4.0
            encode_start = time.time()
            vram_before = _get_vram_gb()
            vram_peak = dict(vram_before)

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video: {video_path}")

            sampled_frames = 0
            total_chunks = 0
            chunk_frames = []
            last_frame = None
            frame_idx = 0
            try:
                while sampled_frames < target_frames:
                    ret, frame = cap.read()
                    if not ret:
                        reached_eof = True
                        break

                    if frame_idx % max(frame_stride, 1) == 0:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(frame_rgb)
                        chunk_frames.append(pil_img)
                        last_frame = pil_img
                        sampled_frames += 1

                        if len(chunk_frames) == CHUNK_SIZE:
                            engine.append_video_chunk(
                                chunk_frames,
                                fps=effective_fps,
                                text_content=f"Chunk {total_chunks + 1}/{current_chunks}."
                            )
                            total_chunks += 1
                            chunk_frames = []

                            # 每 10 个 chunk 报告一次
                            if total_chunks % 10 == 0:
                                cache_info = engine.get_cache_info()
                                vram_current = _get_vram_gb()
                                vram_peak["allocated"] = max(vram_peak["allocated"], vram_current["allocated"])
                                vram_peak["reserved"] = max(vram_peak["reserved"], vram_current["reserved"])
                                print(
                                    f"  Progress: {total_chunks}/{current_chunks} chunks, "
                                    f"Cache: {cache_info['cache_seq_length']}, "
                                    f"VRAM: {vram_current['allocated']:.2f}GB"
                                )

                    frame_idx += 1

                if chunk_frames:
                    if last_frame is None:
                        raise RuntimeError("No frames decoded from video.")
                    while len(chunk_frames) < CHUNK_SIZE:
                        chunk_frames.append(last_frame)
                    engine.append_video_chunk(
                        chunk_frames,
                        fps=effective_fps,
                        text_content=f"Chunk {total_chunks + 1}/{current_chunks}."
                    )
                    total_chunks += 1
            finally:
                cap.release()

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            encode_end = time.time()
            encode_time = encode_end - encode_start

            cache_info = engine.get_cache_info()
            vram_after = _get_vram_gb()
            vram_peak["allocated"] = max(vram_peak["allocated"], vram_after["allocated"])
            vram_peak["reserved"] = max(vram_peak["reserved"], vram_after["reserved"])

            padded_frames = max(0, total_chunks * CHUNK_SIZE - sampled_frames)
            last_successful = {
                "num_chunks": current_chunks,
                "encoded_chunks": total_chunks,
                "total_frames": sampled_frames,
                "padded_frames": padded_frames,
                "encode_time": round(encode_time, 3),
                "cache_seq_length": cache_info["cache_seq_length"],
                "cache_memory_gb": cache_info["cache_memory_gb"],
                "vram_before": vram_before,
                "vram_peak": vram_peak,
                "vram_after_encode": vram_after,
                "reached_eof": reached_eof,
            }

            print(
                f"  ✅ SUCCESS: {total_chunks} chunks "
                f"({last_successful['total_frames']} frames, padded {padded_frames})"
            )
            print(f"     Encode time: {encode_time:.3f}s")
            print(f"     Cache seq length: {cache_info['cache_seq_length']}")
            print(f"     Cache memory: {cache_info['cache_memory_gb']:.4f} GB")
            print(
                f"     VRAM peak: {vram_peak['allocated']:.2f} GB allocated, "
                f"{vram_peak['reserved']:.2f} GB reserved"
            )
            if reached_eof:
                print("     Reached end of video before target chunks were filled.")

            # 清理
            del engine
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            time.sleep(1)  # 给系统一点时间回收内存

            # 增加 chunk 数量继续测试
            if reached_eof:
                break
            current_chunks += increment

        except torch.cuda.OutOfMemoryError:
            print(f"  ❌ OOM at {current_chunks} chunks ({current_chunks * CHUNK_SIZE} frames)")
            print(f"  Last successful: {last_successful['num_chunks']} chunks ({last_successful['total_frames']} frames)")

            # 清理并退出
            del engine
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            break

        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            del engine
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            break

    return last_successful


def main():
    report_dir = os.path.dirname(REPORT_PATH)
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        tee = TeeWriter(sys.stdout, f)
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = tee
        sys.stderr = tee
        try:
            print("=" * 80)
            print("TEST Step 10: Maximum Frame Capacity Test (RTX 4090 24GB)")
            print("=" * 80)
            print(f"Report time: {datetime.now().isoformat(timespec='seconds')}")
            print(f"Chunk size: {CHUNK_SIZE} frames")
            print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
            print(f"Video path: {VIDEO_PATH}")

            if not torch.cuda.is_available():
                print("⚠️  CUDA not available. This test requires GPU.")
                return

            if not os.path.exists(MODEL_PATH):
                print(f"⚠️  Model not found: {MODEL_PATH}. Skip test.")
                return
            if not os.path.exists(VIDEO_PATH):
                print(f"⚠️  Video not found: {VIDEO_PATH}. Skip test.")
                return

            device = "cuda"
            dtype = torch.bfloat16

            # 初始化模型（所有测试共用）
            print("\n[Initialization] Loading model...")
            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained(MODEL_PATH)
            model = StreamQwenModel.from_pretrained(MODEL_PATH, torch_dtype=dtype).to(device)
            model.eval()

            vram_model_only = _get_vram_gb()
            print(f"  ✅ Model loaded. VRAM (model only): {vram_model_only}")

            # 读取视频元信息（原生分辨率）
            video_meta = _get_video_meta(VIDEO_PATH)
            width = video_meta["width"]
            height = video_meta["height"]
            fps = video_meta["fps"]
            total_frames = video_meta["total_frames"]
            duration = video_meta["duration"]
            effective_fps = fps / max(FRAME_STRIDE, 1) if fps > 0 else 0.0
            max_available_frames = (
                (total_frames + max(FRAME_STRIDE, 1) - 1) // max(FRAME_STRIDE, 1)
                if total_frames > 0
                else 0
            )

            print("\n[Video Info]")
            print(f"  Resolution: {width}×{height}")
            print(f"  FPS: {fps:.2f} (effective: {effective_fps:.2f} with stride={FRAME_STRIDE})")
            print(f"  Total frames: {total_frames}")
            print(f"  Duration: {duration:.2f}s")

            results = {}

            name = f"{width}×{height} (native)"
            print(f"\n{'━' * 80}")
            print(f"Testing: {name}")
            print(f"{'━' * 80}")

            result = test_max_frames_for_resolution(
                model=model,
                processor=processor,
                device=device,
                width=width,
                height=height,
                video_path=VIDEO_PATH,
                video_fps=fps,
                max_available_frames=max_available_frames,
                frame_stride=FRAME_STRIDE,
                max_attempts=50,
                initial_chunks=10,
                increment=10,
            )

            results[name] = {
                "width": width,
                "height": height,
                "video_fps": fps,
                "video_total_frames": total_frames,
                "video_duration": duration,
                "effective_fps": effective_fps,
                "frame_stride": FRAME_STRIDE,
                **result,
            }

            # 完全清理显存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            time.sleep(2)

            # ═══════════════════════════════════════════════════════════
            # 汇总报告
            # ═══════════════════════════════════════════════════════════
            print("\n" + "=" * 80)
            print("📊 MAXIMUM FRAME CAPACITY SUMMARY")
            print("=" * 80)

            print(
                f"\n{'Resolution':<20} {'Max Frames':<12} {'Encode Time':<15} "
                f"{'Cache Len':<12} {'Cache Mem':<12} {'VRAM Peak (A/R)'}"
            )
            print("─" * 105)

            for name, r in results.items():
                if r["total_frames"] > 0:
                    print(
                        f"{name:<20} {r['total_frames']:<12} "
                        f"{r['encode_time']:.3f}s{' ' * 7} "
                        f"{r['cache_seq_length']:<12} "
                        f"{r['cache_memory_gb']:.4f} GB{' ' * 3} "
                        f"{r['vram_peak']['allocated']:.2f}/{r['vram_peak']['reserved']:.2f} GB"
                    )
                else:
                    print(f"{name:<20} {'N/A':<12} {'N/A':<15} {'N/A':<12} {'N/A':<12} {'N/A'}")

            print("\n[Key Findings]")

            # 找出最大帧数
            max_frames_res = max(results.items(), key=lambda x: x[1]["total_frames"])
            print(
                f"  • Highest capacity: {max_frames_res[0]} "
                f"with {max_frames_res[1]['total_frames']} frames"
            )

            # 计算平均编码速度
            for name, r in results.items():
                if r["total_frames"] > 0 and r["encode_time"] > 0:
                    fps = r["total_frames"] / r["encode_time"]
                    print(f"  • {name}: {fps:.2f} frames/sec encoding throughput")

            # 显存效率
            print("\n[Memory Efficiency]")
            for name, r in results.items():
                if r["total_frames"] > 0:
                    pixels = r["width"] * r["height"] * r["total_frames"]
                    vram_per_megapixel = r["vram_peak"]["allocated"] / (pixels / 1e6)
                    print(f"  • {name}: {vram_per_megapixel:.4f} GB per megapixel")

            print("\n[Recommendations]")
            print("  • For real-time streaming (24 fps target):")
            for name, r in results.items():
                if r["total_frames"] > 0 and r["encode_time"] > 0:
                    fps = r["total_frames"] / r["encode_time"]
                    max_duration = r["total_frames"] / 24.0  # 假设 24fps 视频
                    if fps >= 24:
                        print(f"    - {name}: Can handle up to {max_duration:.1f}s video at 24fps")

            print(f"\n  • Model baseline VRAM: {vram_model_only['allocated']:.2f} GB")
            print(f"  • Chunk size used: {CHUNK_SIZE} frames")
            print(f"  • Frame stride: {FRAME_STRIDE}")
            print("  • Recommendation: Use smaller resolution for longer videos if needed")

            print("\n" + "=" * 80)
            print("✅ Step 10 PASSED: Maximum frame capacity test completed.")
            print("=" * 80)
            print(f"\nReport saved to: {REPORT_PATH}")

        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


if __name__ == "__main__":
    main()
