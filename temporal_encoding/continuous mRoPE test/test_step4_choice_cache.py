"""
Step 4: ask_choice() Cache 隔离测试

验证 ask_choice() 后：
  1) KV Cache 完全恢复（序列长度 + 校验和）
  2) model.stream_state 完全恢复
  3) 返回的选项不为空

需要 GPU + 模型权重。
"""
import os
import sys
import torch
from datetime import datetime
from PIL import Image, ImageDraw
from transformers import AutoProcessor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import StreamQwenModel, VideoStreamingInference

MODEL_PATH = os.environ.get("QWEN_MODEL_PATH", "/root/autodl-tmp/Qwen/Qwen2___5-VL-3B-Instruct")
REPORT_PATH = os.environ.get(
    "STEP4_REPORT_PATH",
    "/root/autodl-tmp/diploma_project/temporal_encoding/test_step4_choice_cache_report.txt",
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


def _cache_signature(cache):
    if cache is None:
        return (0, 0.0)
    if hasattr(cache, "get_seq_length"):
        try:
            s = float(cache.key_cache[0].sum().item()) if len(cache.key_cache) > 0 else 0.0
        except Exception:
            s = 0.0
        return (cache.get_seq_length(), s)
    seq_len = cache[0][0].shape[-2]
    sig = float(cache[0][0].sum().item())
    return (seq_len, sig)


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
            print("TEST Step 4: ask_choice() Cache Isolation")
            print("=" * 60)
            print(f"Report time: {datetime.now().isoformat(timespec='seconds')}")

            if not os.path.exists(MODEL_PATH):
                print(f"⚠️  Model not found: {MODEL_PATH}. Skip test.")
                return

            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

            processor = AutoProcessor.from_pretrained(MODEL_PATH)
            model = StreamQwenModel.from_pretrained(MODEL_PATH, torch_dtype=dtype).to(device)
            model.eval()

            engine = VideoStreamingInference(model, processor, device)

            # 追加带红色方块的帧
            img = Image.new("RGB", (224, 224), color="white")
            draw = ImageDraw.Draw(img)
            draw.rectangle((50, 50, 170, 170), fill="red", outline="red")
            engine.append_frame(img, text_content="A red square on white background.")

            sig_before = _cache_signature(engine.cache_manager.cache)
            state_before = engine.model.stream_state

            # ask_choice
            choice = engine.ask_choice(
                "What color is the shape? Choose one.",
                choices=["red", "blue", "green"],
            )
            print(f"  Choice: {choice}")
            assert choice != "", "❌ ask_choice returned empty"

            # 验证恢复
            sig_after = _cache_signature(engine.cache_manager.cache)
            state_after = engine.model.stream_state

            assert sig_before == sig_after, f"❌ Cache changed: {sig_before} → {sig_after}"
            assert state_before["last_cache_position"] == state_after["last_cache_position"], \
                f"❌ last_cache_position: {state_before['last_cache_position']} → {state_after['last_cache_position']}"

            rd_b, rd_a = state_before["rope_deltas"], state_after["rope_deltas"]
            assert (rd_b is None) == (rd_a is None), "❌ rope_deltas None mismatch"
            if rd_b is not None:
                assert torch.equal(rd_b, rd_a), "❌ rope_deltas value changed"

            print("\n[Analysis]")
            print(f"  Choice output: {choice}")
            print("  Cache and stream_state remained identical before/after ask_choice.")

            print("\n✅ Step 4 PASSED: ask_choice() cache + stream state isolation verified.")
            print(f"\nReport saved to: {REPORT_PATH}")
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


if __name__ == "__main__":
    main()
