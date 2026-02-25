"""
Step 2: KVCacheManager + Stream State 纯逻辑测试（无需 GPU）

验证：
  1) snapshot/restore 正确保存和恢复 cache + stream_state
  2) clone() 创建独立 cache 副本
  3) discard_snapshot() 正确清理
  4) build_full_attention_mask() 拼接正确
  5) clear() 释放所有状态
  6) get_seq_length() 对各种 cache 格式正确
"""
import os
import sys
import torch
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import KVCacheManager

REPORT_PATH = os.environ.get(
    "STEP2_REPORT_PATH",
    "/root/autodl-tmp/diploma_project/temporal_encoding/test_step2_cache_logic_report.txt",
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


class DummyModel:
    """模拟 StreamQwenModel 的 stream_state 接口。"""
    def __init__(self, pos=0, rd=None):
        self._state = {"last_cache_position": pos, "rope_deltas": rd}

    @property
    def stream_state(self):
        return {
            "last_cache_position": self._state["last_cache_position"],
            "rope_deltas": self._state["rope_deltas"].clone() if self._state["rope_deltas"] is not None else None,
        }

    @stream_state.setter
    def stream_state(self, state):
        self._state = {
            "last_cache_position": state["last_cache_position"],
            "rope_deltas": state["rope_deltas"].clone() if state["rope_deltas"] is not None else None,
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
            print("=" * 60)
            print("TEST Step 2: KVCacheManager Pure Logic")
            print("=" * 60)
            print(f"Report time: {datetime.now().isoformat(timespec='seconds')}")

            mgr = KVCacheManager()
            model = DummyModel(pos=42, rd=torch.tensor([[-3]]))

            # ── 1. snapshot/restore ─────────────────────────
            fake_cache = ((torch.randn(1, 2, 4, 8), torch.randn(1, 2, 4, 8)),)
            mgr.cache = fake_cache
            original_sum = float(fake_cache[0][0].sum().item())

            mgr.snapshot(model)

            # 模拟 ask() 污染
            mgr.cache = ((torch.zeros(1, 2, 8, 8), torch.zeros(1, 2, 8, 8)),)
            model.stream_state = {"last_cache_position": 99, "rope_deltas": torch.tensor([[7]])}

            mgr.restore(model)
            assert mgr.get_seq_length() == 4, f"Expected seq=4, got {mgr.get_seq_length()}"
            assert abs(float(mgr.cache[0][0].sum().item()) - original_sum) < 1e-5
            assert model.stream_state["last_cache_position"] == 42
            assert model.stream_state["rope_deltas"].item() == -3
            print("  ✓ snapshot/restore correct")

            # ── 2. clone() 独立副本 ─────────────────────────
            original = ((torch.ones(1, 2, 4, 8), torch.ones(1, 2, 4, 8)),)
            cloned = KVCacheManager.clone(original)
            cloned[0][0].fill_(0)
            assert original[0][0].sum().item() > 0, "clone() did not create independent copy"
            print("  ✓ clone() independent")

            # ── 3. discard_snapshot() ───────────────────────
            mgr.cache = fake_cache
            mgr.snapshot(model)
            mgr.discard_snapshot()
            model.stream_state = {"last_cache_position": 0, "rope_deltas": None}
            mgr.restore(model)  # should be no-op
            assert model.stream_state["last_cache_position"] == 0
            print("  ✓ discard_snapshot() correct")

            # ── 4. build_full_attention_mask ────────────────
            mgr.cache = ((torch.randn(1, 2, 10, 8), torch.randn(1, 2, 10, 8)),)
            new_mask = torch.ones(1, 5)
            full = mgr.build_full_attention_mask(new_mask)
            assert full.shape == (1, 15), f"Expected (1,15), got {full.shape}"
            assert full.sum().item() == 15.0
            print("  ✓ build_full_attention_mask correct")

            # ── 5. clear() ─────────────────────────────────
            mgr.cache = fake_cache
            mgr.snapshot(model)
            mgr.clear()
            assert mgr.cache is None
            assert mgr.get_seq_length() == 0
            print("  ✓ clear() correct")

            # ── 6. get_seq_length 各格式 ───────────────────
            mgr.cache = None
            assert mgr.get_seq_length() == 0
            mgr.cache = ((torch.randn(1, 2, 7, 8), torch.randn(1, 2, 7, 8)),)
            assert mgr.get_seq_length() == 7
            print("  ✓ get_seq_length() correct")

            print("\n[Analysis]")
            print("  All KVCacheManager functions behaved as expected for tuple-format caches.")
            print("  Snapshot/restore preserved stream_state and cache values.")

            print("\n✅ Step 2 PASSED: All KVCacheManager logic verified.")
            print(f"\nReport saved to: {REPORT_PATH}")
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


if __name__ == "__main__":
    main()
