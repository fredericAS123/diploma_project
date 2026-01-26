import os
import sys
import cv2
import torch
from PIL import Image
from transformers import AutoProcessor

# ================= 配置 =================
MODEL_PATH = "/root/autodl-tmp/Qwen/Qwen2___5-VL-3B-Instruct"
VIDEO_PATH = "/root/autodl-tmp/temporal_encoding/202208312002.mp4"

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model.stream_qwen_model import StreamQwenModel
from model.video_stream_inference import VideoStreamingInference


def extract_ground_truth_frames(video_path, target_seconds, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 1.0

    saved_paths = {}
    for sec in target_seconds:
        frame_idx = int(round(sec * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError(f"Failed to read frame at T={sec}s")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        out_path = os.path.join(output_dir, f"ground_truth_t{sec}.jpg")
        pil_img.save(out_path)
        saved_paths[sec] = out_path

    cap.release()
    return saved_paths


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
        f.write("VRAM Growth Report (Absolute Time)\n")
        f.write("===================================\n")
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    if not os.path.exists(VIDEO_PATH):
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

    # 1) Ground truth extraction
    target_seconds = [4, 15]
    print("🔎 Extracting ground-truth frames...")
    saved = extract_ground_truth_frames(VIDEO_PATH, target_seconds, ".")
    print(f"✅ Saved: {saved[4]}")
    print(f"✅ Saved: {saved[15]}")

    # 2) Initialize model/engine
    print(f"🚀 Loading Model: {MODEL_PATH}")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model = StreamQwenModel.from_pretrained(MODEL_PATH, torch_dtype=dtype).to(device)
    model.eval()

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    base_mem_str, base_peak_str, base_mem = get_vram_usage()
    print(f"🧠 VRAM Baseline (Model Weights): {base_mem_str} | Peak: {base_peak_str}")

    engine = VideoStreamingInference(model, processor, device)

    # 3) Stream video from T=0 to T=20 at 1 FPS (no questions during stream)
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 1.0

    sample_interval = max(1, int(round(fps / 1.0)))
    frame_idx = 0
    last_processed_second = -1
    vram_points = []

    print("🎬 Streaming from T=0 to T=20 at 1 FPS...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_interval != 0:
            frame_idx += 1
            continue

        current_second = int(frame_idx / fps)
        if current_second == last_processed_second:
            frame_idx += 1
            continue

        last_processed_second = current_second

        if current_second > 20:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        engine.append_frame(pil_img, manual_time=current_second)
        mem_str, peak_str, mem_val = get_vram_usage()
        vram_points.append((current_second, mem_val))
        delta = mem_val - base_mem
        print(f"[T={current_second}s] VRAM: {mem_str} (Delta: +{delta:.2f} GB) | Peak: {peak_str}")

        frame_idx += 1

    cap.release()

    # 4) Retrospective tests
    print("\n🧠 Retrospective Queries...")
    q1 = "Describe specifically what happened at the 4th second of the video."
    q2 = "Describe specifically what happened at the 15th second of the video."

    mem_before_str, _, _ = get_vram_usage()
    print(f"[Ask] Before ask() VRAM: {mem_before_str}")
    ans1, metrics1 = engine.ask(q1, manual_time=21)
    mem_after_str, peak_str, _ = get_vram_usage()
    print(f"[Ask] After ask() VRAM: {mem_after_str} | Peak: {peak_str}")

    mem_before_str, _, _ = get_vram_usage()
    print(f"[Ask] Before ask() VRAM: {mem_before_str}")
    ans2, metrics2 = engine.ask(q2, manual_time=21)
    mem_after_str, peak_str, _ = get_vram_usage()
    print(f"[Ask] After ask() VRAM: {mem_after_str} | Peak: {peak_str}")

    print("\n" + "=" * 60)
    print("T=4s Answer:")
    print(ans1)
    print("--> Please verify this text against saved image: ground_truth_t4.jpg")
    print("TTFT: {:.3f}s | Total: {:.3f}s".format(metrics1["ttft"], metrics1["total_latency"]))

    print("\n" + "=" * 60)
    print("T=15s Answer:")
    print(ans2)
    print("--> Please verify this text against saved image: ground_truth_t15.jpg")
    print("TTFT: {:.3f}s | Total: {:.3f}s".format(metrics2["ttft"], metrics2["total_latency"]))

    # Save VRAM growth report
    write_vram_report("vram_absolute_report.txt", vram_points, base_mem, base_mem_str)


if __name__ == "__main__":
    main()
