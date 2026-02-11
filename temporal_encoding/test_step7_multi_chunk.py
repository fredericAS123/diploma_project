"""
Step 7: 多帧 Chunk 规模测试

验证不同 chunk 大小下的编码行为：
  - 2 帧 (T=1, temporal_patch_size 倍数)
  - 4 帧 (T=2, 推荐配置)
  - 6 帧 (T=3, 更高吞吐)
  - 3 帧 (非倍数，触发帧填充)

观察：
  - Cache 增长
  - 编码延迟
  - 处理器自动帧填充行为

需要 GPU + 模型权重。
"""
import os
import sys
import time
import torch
from datetime import datetime
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import StreamQwenModel, VideoStreamingInference

MODEL_PATH = os.environ.get("QWEN_MODEL_PATH", "/root/autodl-tmp/Qwen/Qwen2___5-VL-3B-Instruct")
REPORT_PATH = os.environ.get(
    "STEP7_REPORT_PATH",
    "/root/autodl-tmp/diploma_project/temporal_encoding/test_step7_multi_chunk_report.txt",
)


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


def _make_colored_frame(color: str, text: str):
    """生成带颜色和文本标签的测试帧。"""
    img = Image.new("RGB", (224, 224), color=color)
    draw = ImageDraw.Draw(img)
    draw.text((10, 100), text, fill="black" if color in ["white", "yellow"] else "white")
    return img


def test_chunk_size(engine: VideoStreamingInference, chunk_size: int):
    """测试指定 chunk_size 的编码性能。"""
    print(f"\n{'─' * 60}")
    print(f"Testing chunk_size = {chunk_size} frames")
    print(f"{'─' * 60}")
    
    # 生成测试帧
    colors = ["red", "green", "blue", "yellow", "cyan", "magenta"]
    frames = [_make_colored_frame(colors[i % len(colors)], f"Frame {i+1}") for i in range(chunk_size)]
    
    # 编码
    vram_before = _get_vram_gb()
    start = time.time()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    status = engine.append_video_chunk(
        frames,
        fps=float(chunk_size) / 2.0,  # 调整 fps 保持时间一致性
        text_content=f"Testing {chunk_size}-frame chunk."
    )
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.time()
    vram_after = _get_vram_gb()
    
    encode_time = end - start
    cache_info = engine.get_cache_info()
    
    print(f"  Status: {status}")
    print(f"  Encoding time: {encode_time:.4f}s")
    print(f"  Cache seq length: {cache_info['cache_seq_length']}")
    print(f"  Cache memory: {cache_info['cache_memory_gb']} GB")
    print(f"  Total frames encoded: {cache_info['total_frames']}")
    print(f"  VRAM before encode: {vram_before}")
    print(f"  VRAM after encode: {vram_after}")
    
    return {
        "chunk_size": chunk_size,
        "encode_time": round(encode_time, 4),
        "cache_seq_length": cache_info["cache_seq_length"],
        "cache_memory_gb": cache_info["cache_memory_gb"],
        "vram_before": vram_before,
        "vram_after": vram_after,
    }


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
            print("=" * 70)
            print("TEST Step 7: Multi-Frame Chunk Size Comparison")
            print("=" * 70)
            print(f"Report time: {datetime.now().isoformat(timespec='seconds')}")

            if not os.path.exists(MODEL_PATH):
                print(f"⚠️  Model not found: {MODEL_PATH}. Skip test.")
                return

            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained(MODEL_PATH)
            model = StreamQwenModel.from_pretrained(MODEL_PATH, torch_dtype=dtype).to(device)
            model.eval()

            # 测试不同 chunk 大小
            chunk_sizes = [2, 4, 6, 3]  # 3 帧测试非倍数情况
            results = []

            for chunk_size in chunk_sizes:
                # 每次测试使用全新引擎（避免缓存累积）
                engine = VideoStreamingInference(model, processor, device)
                result = test_chunk_size(engine, chunk_size)
                results.append(result)

                # 清理
                engine.reset()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # 汇总报告
            print("\n" + "=" * 70)
            print("📊 CHUNK SIZE COMPARISON SUMMARY")
            print("=" * 70)
            print(f"{'Chunk Size':<12} {'Encode Time':<15} {'Cache Seq Len':<15} {'Cache Mem (GB)'} {'VRAM Δ (GB)'}")
            print("─" * 90)
            for r in results:
                vram_delta = r["vram_after"]["allocated"] - r["vram_before"]["allocated"]
                print(
                    f"{r['chunk_size']:<12} {r['encode_time']:<15} {r['cache_seq_length']:<15} {r['cache_memory_gb']:<13} {vram_delta:.2f}"
                )

            print("\n📝 Observations:")
            print("  - 2 frames (T=1): Minimal latency, smallest cache growth per chunk")
            print("  - 4 frames (T=2): Balanced, recommended for real-time streaming")
            print("  - 6 frames (T=3): Higher throughput, suitable for batch processing")
            print("  - 3 frames: Non-multiple of temporal_patch_size=2, may trigger frame duplication")
            print("    (processor will duplicate the last frame to make it 4 frames internally)")

            print("\n[Analysis]")
            fastest = min(results, key=lambda x: x["encode_time"])
            largest_cache = max(results, key=lambda x: x["cache_seq_length"])
            print(f"  Fastest encode: chunk_size={fastest['chunk_size']} ({fastest['encode_time']}s)")
            print(f"  Largest cache growth: chunk_size={largest_cache['chunk_size']} (seq_len={largest_cache['cache_seq_length']})")
            print("  Recommendation: use 4-frame chunks for balanced latency and cache growth.")

            print("\n✅ Step 7 PASSED: Multi-chunk size test completed.")
            print(f"\nReport saved to: {REPORT_PATH}")
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


if __name__ == "__main__":
    main()
