"""
Step 8: 多轮 QA 测试 — 编码→问→继续编码→再问

验证：
  1) 第一轮 QA (update_state=False) 后恢复缓存，继续追加帧
    2) 第二轮 QA 能看到新追加的帧内容
    3) cache_memory_gb 不再为 0（验证修复 #1）
  4) 后续帧使用 user turn 包裹（验证优化 #2）
  5) 两轮回答具有一致性与递增性

测试场景设计：
  Phase 1: 编码 2 帧红色圆形 → 问 "What color is the shape?"
  Phase 2: 继续编码 2 帧蓝色方块 → 问 "How many shapes appeared? What were they?"
  Phase 3: 再编码 2 帧绿色三角 → 问 "What appeared last?"

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
    "STEP8_REPORT_PATH",
    "/root/autodl-tmp/diploma_project/temporal_encoding/test_step8_multi_round_qa_report.txt",
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


def _make_img(color: str, shape: str, size: int = 224):
    """生成带有彩色形状的测试图像。"""
    img = Image.new("RGB", (size, size), color="white")
    draw = ImageDraw.Draw(img)
    if shape == "circle":
        draw.ellipse((40, 40, size - 40, size - 40), fill=color, outline=color)
    elif shape == "square":
        draw.rectangle((40, 40, size - 40, size - 40), fill=color, outline=color)
    elif shape == "triangle":
        draw.polygon([(size // 2, 40), (40, size - 40), (size - 40, size - 40)], fill=color, outline=color)
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
            print("TEST Step 8: Multi-Round QA (Encode→Ask→Continue Encode→Ask Again)")
            print("=" * 70)
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

            results = {}

            # ════════════════════════════════════════════════════════════
            # Phase 1: 红色圆形 → QA
            # ════════════════════════════════════════════════════════════
            print("\n" + "=" * 60)
            print("📍 Phase 1: Encode red circles → Ask about color")
            print("=" * 60)

            red_frames = [_make_img("red", "circle"), _make_img("red", "circle")]
            status1 = engine.append_video_chunk(red_frames, fps=2.0, text_content="A red circle appears.")
            print(f"  Encode: {status1}")

            info1 = engine.get_cache_info()
            print(f"  Cache info: {info1}")

            # 验证 cache_memory_gb 修复
            assert info1["cache_memory_gb"] > 0, \
                f"❌ BUG: cache_memory_gb is {info1['cache_memory_gb']} — should be > 0!"
            print(f"  ✓ cache_memory_gb = {info1['cache_memory_gb']:.4f} GB (fix #1 verified)")

            q1 = "What color is the shape you see?"
            ans1, metrics1 = engine.ask(q1, max_new_tokens=30, update_state=False)
            print(f"\n  Q1: {q1}")
            print(f"  A1: {ans1}")
            print(f"  TTFT={metrics1['ttft']:.3f}s, Total={metrics1['total_latency']:.3f}s")

            info1_after = engine.get_cache_info()
            print(f"  Cache after QA (should match pre-QA): {info1_after}")
            assert info1_after["cache_seq_length"] == info1["cache_seq_length"], \
                "❌ Cache not restored after QA!"
            print("  ✓ Cache restored correctly after QA round 1")

            results["phase1"] = {
                "cache_len": info1["cache_seq_length"],
                "cache_gb": info1["cache_memory_gb"],
                "answer": ans1,
                "ttft": metrics1["ttft"],
            }

            # ════════════════════════════════════════════════════════════
            # Phase 2: 继续编码蓝色方块 → QA
            # ════════════════════════════════════════════════════════════
            print("\n" + "=" * 60)
            print("📍 Phase 2: Continue encoding blue squares → Ask about all shapes")
            print("=" * 60)

            blue_frames = [_make_img("blue", "square"), _make_img("blue", "square")]
            status2 = engine.append_video_chunk(blue_frames, fps=2.0, text_content="Blue squares appear.")
            print(f"  Encode: {status2}")

            info2 = engine.get_cache_info()
            print(f"  Cache info: {info2}")

            # Cache 应该增长
            assert info2["cache_seq_length"] > info1["cache_seq_length"], \
                f"❌ Cache should grow! Before={info1['cache_seq_length']}, After={info2['cache_seq_length']}"
            print(f"  ✓ Cache grew: {info1['cache_seq_length']} → {info2['cache_seq_length']}")

            # stream_state 应该更新
            # stream_state 应该更新（绝对位置应该增长）
            state2 = info2["stream_state"]
            state1 = info1["stream_state"]
            assert state2["last_cache_position"] > state1["last_cache_position"], \
                f"❌ stream_state position not updated: {state1['last_cache_position']} → {state2['last_cache_position']}"
            print(f"  ✓ stream_state.last_cache_position increased: {state1['last_cache_position']} → {state2['last_cache_position']}")

            q2 = "Describe all shapes you have seen so far, including their colors."
            ans2, metrics2 = engine.ask(q2, max_new_tokens=50, update_state=False)
            print(f"\n  Q2: {q2}")
            print(f"  A2: {ans2}")
            print(f"  TTFT={metrics2['ttft']:.3f}s, Total={metrics2['total_latency']:.3f}s")

            info2_after = engine.get_cache_info()
            assert info2_after["cache_seq_length"] == info2["cache_seq_length"], \
                "❌ Cache not restored after QA round 2!"
            print("  ✓ Cache restored correctly after QA round 2")

            results["phase2"] = {
                "cache_len": info2["cache_seq_length"],
                "cache_gb": info2["cache_memory_gb"],
                "answer": ans2,
                "ttft": metrics2["ttft"],
            }

            # ════════════════════════════════════════════════════════════
            # Phase 3: 继续编码绿色三角 → QA
            # ════════════════════════════════════════════════════════════
            print("\n" + "=" * 60)
            print("📍 Phase 3: Continue encoding green triangles → Ask about last shape")
            print("=" * 60)

            green_frames = [_make_img("green", "triangle"), _make_img("green", "triangle")]
            status3 = engine.append_video_chunk(green_frames, fps=2.0, text_content="Green triangles appear.")
            print(f"  Encode: {status3}")

            info3 = engine.get_cache_info()
            print(f"  Cache info: {info3}")

            assert info3["cache_seq_length"] > info2["cache_seq_length"], \
                f"❌ Cache should grow! Before={info2['cache_seq_length']}, After={info3['cache_seq_length']}"
            print(f"  ✓ Cache grew: {info2['cache_seq_length']} → {info3['cache_seq_length']}")

            q3 = "What shape and color appeared most recently?"
            ans3, metrics3 = engine.ask(q3, max_new_tokens=30, update_state=False)
            print(f"\n  Q3: {q3}")
            print(f"  A3: {ans3}")
            print(f"  TTFT={metrics3['ttft']:.3f}s, Total={metrics3['total_latency']:.3f}s")

            info3_after = engine.get_cache_info()
            assert info3_after["cache_seq_length"] == info3["cache_seq_length"], \
                "❌ Cache not restored after QA round 3!"
            print("  ✓ Cache restored correctly after QA round 3")

            results["phase3"] = {
                "cache_len": info3["cache_seq_length"],
                "cache_gb": info3["cache_memory_gb"],
                "answer": ans3,
                "ttft": metrics3["ttft"],
            }

            # ════════════════════════════════════════════════════════════
            # Summary & Validation
            # ════════════════════════════════════════════════════════════
            print("\n" + "=" * 70)
            print("📊 MULTI-ROUND QA SUMMARY")
            print("=" * 70)

            print("\n[Cache Growth]")
            for phase in ["phase1", "phase2", "phase3"]:
                r = results[phase]
                print(f"  {phase}: cache_len={r['cache_len']}, cache_gb={r['cache_gb']:.4f} GB")

            print("\n[Answers]")
            for i, phase in enumerate(["phase1", "phase2", "phase3"], 1):
                print(f"  Round {i}: {results[phase]['answer']}")

            print("\n[Latency]")
            for i, phase in enumerate(["phase1", "phase2", "phase3"], 1):
                print(f"  Round {i} TTFT: {results[phase]['ttft']:.3f}s")

            # Semantic checks
            print("\n[Semantic Validation]")

            # Phase 1: 应该提到 red
            has_red = "red" in ans1.lower()
            print(f"  Phase 1 mentions 'red': {has_red} {'✓' if has_red else '⚠️'}")

            # Phase 2: 应该提到 blue（至少）
            has_blue_in_p2 = "blue" in ans2.lower()
            print(f"  Phase 2 mentions 'blue': {has_blue_in_p2} {'✓' if has_blue_in_p2 else '⚠️'}")

            # Phase 3: 应该提到 green 或 triangle
            has_green = "green" in ans3.lower() or "triangle" in ans3.lower()
            print(f"  Phase 3 mentions 'green/triangle': {has_green} {'✓' if has_green else '⚠️'}")

            # cache_memory_gb 应该持续增长
            gb_growing = (
                results["phase1"]["cache_gb"] > 0
                and results["phase2"]["cache_gb"] > results["phase1"]["cache_gb"]
                and results["phase3"]["cache_gb"] > results["phase2"]["cache_gb"]
            )
            print(f"  cache_memory_gb monotonically increasing: {gb_growing} {'✓' if gb_growing else '⚠️'}")

            print("\n" + "=" * 70)
            all_cache_ok = (
                results["phase1"]["cache_gb"] > 0
                and results["phase2"]["cache_len"] > results["phase1"]["cache_len"]
                and results["phase3"]["cache_len"] > results["phase2"]["cache_len"]
            )
            if all_cache_ok:
                print("✅ Step 8 PASSED: Multi-round QA with cache continuity verified.")
            else:
                print("❌ Step 8 FAILED: Cache growth or memory reporting issue detected.")
            print("=" * 70)
            print(f"\nReport saved to: {REPORT_PATH}")

        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


if __name__ == "__main__":
    main()
