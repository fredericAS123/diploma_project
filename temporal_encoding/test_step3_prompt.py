"""
Step 3: Prompt 裁剪逻辑测试（无需 GPU）

验证 _extract_vision_segment 正确裁剪 chat template。
"""
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import VideoStreamingInference

REPORT_PATH = os.environ.get(
    "STEP3_REPORT_PATH",
    "/root/autodl-tmp/diploma_project/temporal_encoding/test_step3_prompt_report.txt",
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
            print("=" * 60)
            print("TEST Step 3: Prompt Trimming Logic")
            print("=" * 60)
            print(f"Report time: {datetime.now().isoformat(timespec='seconds')}")

            # Case 1: 正常 vision segment
            text = "<|im_start|>user\nhello<|vision_start|>XYZ<|vision_end|>\n<|im_end|>"
            seg = VideoStreamingInference._extract_vision_segment(text)
            assert seg == "<|vision_start|>XYZ<|vision_end|>", f"❌ Got: {seg}"
            print("  ✓ Normal vision segment extracted")

            # Case 2: 无 vision token 的 fallback
            text2 = "no vision tokens here"
            seg2 = VideoStreamingInference._extract_vision_segment(text2)
            assert seg2 == text2, f"❌ Fallback failed: {seg2}"
            print("  ✓ Fallback passthrough correct")

            # Case 3: 多个 vision segment（应取第一个）
            text3 = "A<|vision_start|>FIRST<|vision_end|>B<|vision_start|>SECOND<|vision_end|>C"
            seg3 = VideoStreamingInference._extract_vision_segment(text3)
            assert seg3 == "<|vision_start|>FIRST<|vision_end|>", f"❌ Multi-segment: {seg3}"
            print("  ✓ Multi-segment takes first")

            # Case 4: 空 vision segment
            text4 = "<|vision_start|><|vision_end|>"
            seg4 = VideoStreamingInference._extract_vision_segment(text4)
            assert seg4 == "<|vision_start|><|vision_end|>", f"❌ Empty segment: {seg4}"
            print("  ✓ Empty vision segment correct")

            print("\n[Analysis]")
            print("  _extract_vision_segment behaves correctly across normal, fallback, multi, and empty cases.")

            print("\n✅ Step 3 PASSED: Prompt trimming logic verified.")
            print(f"\nReport saved to: {REPORT_PATH}")
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


if __name__ == "__main__":
    main()
