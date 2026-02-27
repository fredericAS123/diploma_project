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