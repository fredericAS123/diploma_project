"""
Step 9: Web Demo 后端集成测试

验证：
  1) QwenInferenceWrapper 能正确初始化
  2) process_frame() 单帧编码无异常
  3) process_video_chunk() 多帧编码无异常
    4) ask_question() 能正确返回答案和 metrics
    5) get_cache_info() 返回非零 cache_memory_gb
  6) reset() 后状态完全重置
  7) 连续 chunk 编码→问→继续编码→问 的完整流程

不启动 Gradio 服务器，仅验证后端推理引擎 API。
需要 GPU + 模型权重。
"""
import os
import sys
import torch
from datetime import datetime
from PIL import Image, ImageDraw

# web_demo 目录中的 Qwen_inference 需要特殊 import 处理
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_WEB_DEMO_DIR = os.path.join(os.path.dirname(_CURRENT_DIR), "web_demo")
if _WEB_DEMO_DIR not in sys.path:
    sys.path.insert(0, _WEB_DEMO_DIR)
# 同时确保 temporal_encoding 在 path 中
_TEMPORAL_DIR = os.path.dirname(_CURRENT_DIR) if os.path.basename(_CURRENT_DIR) == "temporal_encoding" else _CURRENT_DIR
if _TEMPORAL_DIR not in sys.path:
    sys.path.insert(0, _TEMPORAL_DIR)

from Qwen_inference import QwenInferenceWrapper

MODEL_PATH = os.environ.get("QWEN_MODEL_PATH", "/root/autodl-tmp/Qwen/Qwen2___5-VL-3B-Instruct")
REPORT_PATH = os.environ.get(
    "STEP9_REPORT_PATH",
    "/root/autodl-tmp/diploma_project/temporal_encoding/test_step9_web_demo_report.txt",
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


def _make_img(color: str, size: int = 224):
    img = Image.new("RGB", (size, size), color=color)
    draw = ImageDraw.Draw(img)
    draw.rectangle((40, 40, size - 40, size - 40), fill=color)
    return img


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
            print("TEST Step 9: Web Demo Backend Integration Test")
            print("=" * 70)
            print(f"Report time: {datetime.now().isoformat(timespec='seconds')}")

            if not os.path.exists(MODEL_PATH):
                print(f"⚠️  Model not found: {MODEL_PATH}. Skip test.")
                return

            # ── 1) 初始化 ──
            print("\n[1] Initializing QwenInferenceWrapper...")
            wrapper = QwenInferenceWrapper(model_path=MODEL_PATH)
            print("  ✓ QwenInferenceWrapper initialized successfully")

            # ── 2) 单帧编码 ──
            print("\n[2] Testing process_frame() (single image)...")
            red_img = _make_img("red")
            status = wrapper.process_frame(red_img)
            print(f"  Status: {status}")
            assert "encoded" in status.lower() or "cache" in status.lower(), \
                f"❌ Unexpected status: {status}"
            print("  ✓ process_frame() works correctly")

            info_after_frame = wrapper.get_cache_info()
            print(f"  Cache info: {info_after_frame}")
            assert info_after_frame["cache_seq_length"] > 0, "❌ Cache should not be empty after encoding"
            assert info_after_frame["cache_memory_gb"] > 0, \
                f"❌ cache_memory_gb should be > 0, got {info_after_frame['cache_memory_gb']}"
            print(f"  ✓ cache_memory_gb = {info_after_frame['cache_memory_gb']:.4f} GB (fix verified)")

            # ── 3) 多帧 chunk 编码 ──
            print("\n[3] Testing process_video_chunk()...")
            blue_frames = [_make_img("blue") for _ in range(4)]
            status2 = wrapper.process_video_chunk(blue_frames, fps=2.0)
            print(f"  Status: {status2}")

            info_after_chunk = wrapper.get_cache_info()
            print(f"  Cache info: {info_after_chunk}")
            assert info_after_chunk["cache_seq_length"] > info_after_frame["cache_seq_length"], \
                "❌ Cache should grow after chunk encoding"
            print(f"  ✓ Cache grew: {info_after_frame['cache_seq_length']} → {info_after_chunk['cache_seq_length']}")

            # ── 4) ask_question ──
            print("\n[4] Testing ask_question()...")
            response, metrics = wrapper.ask_question(
                "What colors do you see?",
                max_new_tokens=30,
                return_metrics=True,
            )
            print(f"  Answer: {response}")
            print(f"  TTFT: {metrics['ttft']:.3f}s, Total: {metrics['total_latency']:.3f}s")
            assert len(response) > 0, "❌ Empty response"
            assert "ttft" in metrics, "❌ Missing TTFT in metrics"
            print("  ✓ ask_question() returns valid response and metrics")

            # 缓存应该没变（update_state=False 默认）
            info_after_qa = wrapper.get_cache_info()
            assert info_after_qa["cache_seq_length"] == info_after_chunk["cache_seq_length"], \
                "❌ Cache should be restored after QA (update_state=False)"
            print("  ✓ Cache restored correctly after QA")

            # ── 5) kwargs 兼容性（旧接口 manual_time 参数应被忽略）──
            print("\n[5] Testing backward compatibility (manual_time kwarg ignored)...")
            response2 = wrapper.ask_question(
                "Describe the image.",
                max_new_tokens=20,
                manual_time=1.5,  # 旧参数，应被 **kwargs 吞掉
            )
            print(f"  Answer: {response2}")
            assert len(response2) > 0, "❌ Empty response with legacy kwarg"
            print("  ✓ Legacy manual_time kwarg ignored without error")

            # ── 6) reset ──
            print("\n[6] Testing reset()...")
            wrapper.reset()
            info_after_reset = wrapper.get_cache_info()
            print(f"  Cache info after reset: {info_after_reset}")
            assert info_after_reset["cache_seq_length"] == 0, \
                f"❌ Cache should be empty after reset, got {info_after_reset['cache_seq_length']}"
            assert info_after_reset["total_frames"] == 0, \
                f"❌ total_frames should be 0 after reset, got {info_after_reset['total_frames']}"
            print("  ✓ reset() clears all state correctly")

            # ── 7) 完整流程：chunk→ask→chunk→ask ──
            print("\n[7] Testing full pipeline: chunk→ask→chunk→ask...")
            green_frames = [_make_img("green") for _ in range(2)]
            wrapper.process_video_chunk(green_frames, fps=2.0)
            ans1, m1 = wrapper.ask_question("What do you see?", max_new_tokens=20, return_metrics=True)
            print(f"  Round 1: {ans1} (TTFT={m1['ttft']:.3f}s)")

            yellow_frames = [_make_img("yellow") for _ in range(2)]
            wrapper.process_video_chunk(yellow_frames, fps=2.0)
            ans2, m2 = wrapper.ask_question("What colors have appeared?", max_new_tokens=30, return_metrics=True)
            print(f"  Round 2: {ans2} (TTFT={m2['ttft']:.3f}s)")

            final_info = wrapper.get_cache_info()
            print(f"  Final cache: {final_info}")
            assert final_info["cache_seq_length"] > 0, "❌ Cache should not be empty"
            assert final_info["total_frames"] == 4, \
                f"❌ Expected 4 total frames, got {final_info['total_frames']}"
            print("  ✓ Full pipeline works: chunk→ask→chunk→ask")

            # ── Summary ──
            print("\n" + "=" * 70)
            print("✅ Step 9 PASSED: Web Demo backend integration verified.")
            print("   - QwenInferenceWrapper init ✓")
            print("   - process_frame / process_video_chunk ✓")
            print("   - ask_question (with metrics + backward compat) ✓")
            print("   - cache_memory_gb > 0 ✓")
            print("   - reset() ✓")
            print("   - Multi-round pipeline ✓")
            print("=" * 70)
            print(f"\nReport saved to: {REPORT_PATH}")

        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


if __name__ == "__main__":
    main()
