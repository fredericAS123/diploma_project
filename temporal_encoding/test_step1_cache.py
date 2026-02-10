"""
Step 1: KV Cache + Stream State Snapshot/Restore 隔离测试

验证 ask(update_state=False) 后：
  1) KV Cache 序列长度和数值完全恢复
  2) model.stream_state (last_cache_position + rope_deltas) 完全恢复
  3) 恢复后可以继续正常 append_frame

需要 GPU + 模型权重。
"""
import os
import sys
import torch
from datetime import datetime
from PIL import Image
from transformers import AutoProcessor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import StreamQwenModel, VideoStreamingInference

MODEL_PATH = os.environ.get("QWEN_MODEL_PATH", "/root/autodl-tmp/Qwen/Qwen2___5-VL-3B-Instruct")
REPORT_PATH = os.environ.get(
    "STEP1_REPORT_PATH",
    "/root/autodl-tmp/diploma_project/temporal_encoding/test_step1_cache_report.txt",
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
    """获取 cache 的 (seq_length, checksum) 签名。"""
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
            print("TEST Step 1: Cache + Stream State Snapshot/Restore")
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

            # ── 追加一帧 ──
            img = Image.new("RGB", (224, 224), color="white")
            status = engine.append_frame(img, text_content="A white image.")
            print(f"  Append: {status}")

            # ── 记录 snapshot 前的状态 ──
            sig_before = _cache_signature(engine.cache_manager.cache)
            state_before = engine.model.stream_state
            print(f"  Before ask: cache_sig={sig_before}, pos={state_before['last_cache_position']}")

            # ── ask (不更新状态) ──
            answer, metrics = engine.ask("What is in the image?", max_new_tokens=8, update_state=False)
            print(f"  Answer: {answer}")
            print(f"  TTFT={metrics['ttft']:.3f}s, Total={metrics['total_latency']:.3f}s")

            # ── 验证恢复 ──
            sig_after = _cache_signature(engine.cache_manager.cache)
            state_after = engine.model.stream_state

            assert sig_before == sig_after, \
                f"❌ Cache changed: {sig_before} → {sig_after}"
            assert state_before["last_cache_position"] == state_after["last_cache_position"], \
                f"❌ last_cache_position: {state_before['last_cache_position']} → {state_after['last_cache_position']}"

            rd_b, rd_a = state_before["rope_deltas"], state_after["rope_deltas"]
            assert (rd_b is None) == (rd_a is None), "❌ rope_deltas None mismatch"
            if rd_b is not None:
                assert torch.equal(rd_b, rd_a), "❌ rope_deltas value changed"

            # ── 验证恢复后能继续追加 ──
            img2 = Image.new("RGB", (224, 224), color="red")
            status2 = engine.append_frame(img2, text_content="A red image.")
            print(f"  Post-restore append: {status2}")
            sig_after2 = _cache_signature(engine.cache_manager.cache)
            assert sig_after2[0] > sig_after[0], "❌ Cache did not grow after post-restore append"

            print("\n[Analysis]")
            print(f"  Cache signature before/after ask: {sig_before} -> {sig_after}")
            print(f"  Post-restore cache length: {sig_after2[0]}")
            print("  Stream state and cache remained isolated across QA.")

            print("\n✅ Step 1 PASSED: Snapshot/restore isolates ask() correctly.")
            print(f"\nReport saved to: {REPORT_PATH}")
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


if __name__ == "__main__":
    main()
