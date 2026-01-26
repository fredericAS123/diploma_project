import argparse
import os
import sys
import time
import cv2
import torch
from PIL import Image
from transformers import AutoProcessor

# ================= 配置 =================
MODEL_PATH = "/root/autodl-tmp/Qwen/Qwen2___5-VL-3B-Instruct"
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.stream_qwen_model import StreamQwenModel
from model.video_stream_inference import VideoStreamingInference


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Qwen2.5-VL streaming TTFT on real video")
    parser.add_argument(
        "--video_path",
        type=str,
        default="/root/autodl-tmp/temporal_encoding/202208312002.mp4",
        help="Path to the input .mp4 video file",
    )
    parser.add_argument(
        "--sample_fps",
        type=float,
        default=1.0,
        help="Sampling rate in frames per second of video time",
    )
    parser.add_argument(
        "--max_side",
        type=int,
        default=1280,
        help="Optional max side length for OOM prevention (no resize if smaller)",
    )
    return parser.parse_args()


def get_vram_usage():
    if not torch.cuda.is_available():
        return "0.00 GB", "0.00 GB", 0.0
    current_gb = torch.cuda.memory_allocated() / (1024 ** 3)
    peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    return f"{current_gb:.2f} GB", f"{peak_gb:.2f} GB", current_gb


def write_vram_report(report_path, points, base_mem, base_mem_str):
    if not points:
        return
    points_sorted = sorted(points, key=lambda x: x[0])
    t0, m0 = points_sorted[0]
    t1, m1 = points_sorted[-1]
    duration = max(1e-6, t1 - t0)
    avg_growth = (m1 - m0) / duration

    # Linear regression slope (GB/s)
    ts = [p[0] for p in points_sorted]
    ms = [p[1] for p in points_sorted]
    t_mean = sum(ts) / len(ts)
    m_mean = sum(ms) / len(ms)
    num = sum((t - t_mean) * (m - m_mean) for t, m in zip(ts, ms))
    den = sum((t - t_mean) ** 2 for t in ts) or 1e-6
    slope = num / den

    peak_str, peak_val = "0.00 GB", 0.0
    if torch.cuda.is_available():
        peak_val = torch.cuda.max_memory_allocated() / (1024 ** 3)
        peak_str = f"{peak_val:.2f} GB"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("VRAM Growth Report (Benchmark)\n")
        f.write("==============================\n")
        f.write(f"Baseline: {base_mem_str}\n")
        f.write(f"Peak: {peak_str}\n")
        f.write(f"Duration: {duration:.2f}s\n")
        f.write(f"Avg growth (first-last): {avg_growth:.4f} GB/s\n")
        f.write(f"Linear regression slope: {slope:.4f} GB/s\n")
        f.write("\nPer-second VRAM (GB):\n")
        for t, m in points_sorted:
            delta = m - base_mem
            f.write(f"T={t:04d}s | {m:.3f} GB | Delta: +{delta:.3f} GB\n")


def main():
    args = parse_args()
    if not os.path.exists(args.video_path):
        raise FileNotFoundError(f"Video not found: {args.video_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print(f"🚀 Loading Model: {MODEL_PATH}")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model = StreamQwenModel.from_pretrained(MODEL_PATH, torch_dtype=dtype).to(device)
    model.eval()

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    base_mem_str, base_peak_str, base_mem = get_vram_usage()
    print(f"🧠 VRAM Baseline (Model Weights): {base_mem_str} | Peak: {base_peak_str}")

    engine = VideoStreamingInference(model, processor, device)

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 1.0
    print(f"📽️ Video FPS: {fps:.3f}")

    sample_interval = max(1, int(round(fps / args.sample_fps)))
    print(f"⏱️ Sampling every {sample_interval} frames (~{args.sample_fps} fps)")

    checkpoints = [5, 10, 30]
    asked = set()
    ttfts = []
    totals = []
    max_checkpoint = max(checkpoints)
    vram_points = []

    frame_idx = 0
    last_processed_second = -1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process at sampling interval
        if frame_idx % sample_interval != 0:
            frame_idx += 1
            continue

        # Compute video time in seconds
        current_second = int(frame_idx / fps)
        if current_second == last_processed_second:
            frame_idx += 1
            continue

        last_processed_second = current_second

        # Convert BGR -> RGB -> PIL (no forced resize unless OOM prevention)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        if args.max_side and max(pil_img.size) > args.max_side:
            scale = args.max_side / float(max(pil_img.size))
            new_size = (int(pil_img.size[0] * scale), int(pil_img.size[1] * scale))
            pil_img = pil_img.resize(new_size)

        engine.append_frame(pil_img, manual_time=current_second, text_content=f"Streaming frame at T={current_second}s")
        mem_str, peak_str, mem_val = get_vram_usage()
        vram_points.append((current_second, mem_val))
        delta = mem_val - base_mem
        print(f"[T={current_second}s] VRAM: {mem_str} (Delta: +{delta:.2f} GB) | Peak: {peak_str}")

        if current_second in checkpoints and current_second not in asked:
            asked.add(current_second)
            question = "Describe the current frame."
            mem_before_str, _, _ = get_vram_usage()
            print(f"[T={current_second}s] Before ask() VRAM: {mem_before_str}")
            response, metrics = engine.ask(question, manual_time=current_second + 1)
            mem_after_str, peak_str, _ = get_vram_usage()
            print(f"[T={current_second}s] After ask() VRAM: {mem_after_str} | Peak: {peak_str}")
            ttft = metrics.get("ttft", float("nan"))
            total = metrics.get("total_latency", float("nan"))
            ttfts.append(ttft)
            totals.append(total)
            print(f"[T={current_second}s] TTFT: {ttft:.3f}s | Total: {total:.3f}s | Resp: {response}")

        if current_second >= max_checkpoint and asked == set(checkpoints):
            break

        frame_idx += 1

    cap.release()

    if len(ttfts) == 0:
        print("FAILURE: No checkpoints were reached in the video.")
        write_vram_report("vram_benchmark_report.txt", vram_points, base_mem, base_mem_str)
        return

    avg_ttft = sum(ttfts) / len(ttfts)
    avg_total = sum(totals) / len(totals)
    print(f"\n📊 Avg TTFT: {avg_ttft:.3f}s | Avg Total: {avg_total:.3f}s")

    if avg_ttft < 1.0:
        print("SUCCESS")
    else:
        print("FAILURE")

    # Save VRAM growth report
    write_vram_report("vram_benchmark_report.txt", vram_points, base_mem, base_mem_str)


if __name__ == "__main__":
    main()
