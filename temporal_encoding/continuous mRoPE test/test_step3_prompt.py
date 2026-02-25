"""
Step 3: Prompt 裁剪逻辑测试（无需 GPU）

验证 _extract_vision_segment 和 _extract_user_vision_turn 正确裁剪 chat template。
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

            # ── _extract_vision_segment tests ──
            print("\n[Part A] _extract_vision_segment")

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

            # ── _extract_user_vision_turn tests ──
            print("\n[Part B] _extract_user_vision_turn (prompt structure optimization)")

            # Case 5: 正常 user turn 包裹
            text5 = "<|im_start|>system\nsys<|im_end|>\n<|im_start|>user\nhello<|vision_start|>IMG_TOKENS<|vision_end|>world<|im_end|>\n"
            turn5 = VideoStreamingInference._extract_user_vision_turn(text5)
            expected5 = "<|im_start|>user\n<|vision_start|>IMG_TOKENS<|vision_end|><|im_end|>\n"
            assert turn5 == expected5, f"❌ Got: {repr(turn5)}"
            print("  ✓ User turn with vision extracted correctly (with im_start/im_end)")

            # Case 6: 无 vision token 的 fallback → 原文返回
            text6 = "plain text without vision"
            turn6 = VideoStreamingInference._extract_user_vision_turn(text6)
            assert turn6 == text6, f"❌ Fallback failed: {turn6}"
            print("  ✓ Fallback passthrough for non-vision text")

            # Case 7: 验证后续帧包裹结构包含正确的特殊 token
            text7 = "<|im_start|>user\n<|vision_start|>VID_DATA<|vision_end|><|im_end|>\n"
            turn7 = VideoStreamingInference._extract_user_vision_turn(text7)
            assert "<|im_start|>user" in turn7, f"❌ Missing im_start user: {turn7}"
            assert "<|im_end|>" in turn7, f"❌ Missing im_end: {turn7}"
            assert "<|vision_start|>VID_DATA<|vision_end|>" in turn7, f"❌ Missing vision: {turn7}"
            print("  ✓ Wrapped turn preserves all structural tokens")

            # Case 8: 验证后续帧不包含系统提示
            text8 = "<|im_start|>system\nYou are helpful<|im_end|>\n<|im_start|>user\n<|vision_start|>FRAMES<|vision_end|><|im_end|>\n"
            turn8 = VideoStreamingInference._extract_user_vision_turn(text8)
            assert "system" not in turn8.split("user", 1)[-1], f"❌ System prompt leaked: {turn8}"
            assert "You are helpful" not in turn8, f"❌ System content leaked: {turn8}"
            print("  ✓ Wrapped turn does NOT contain system prompt")

            print("\n[Analysis]")
            print("  _extract_vision_segment: 4/4 cases passed (backward compatible)")
            print("  _extract_user_vision_turn: 4/4 cases passed (new optimization)")
            print("  Subsequent chunks now wrapped in <|im_start|>user...im_end> structure,")
            print("  reducing OOD effect from bare vision tokens.")

            print("\n✅ Step 3 PASSED: Prompt trimming logic verified (original + optimized).")
            print(f"\nReport saved to: {REPORT_PATH}")
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


if __name__ == "__main__":
    main()
