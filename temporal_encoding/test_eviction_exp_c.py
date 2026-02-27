"""
实验 C: 滑窗逐段 + 周期性提问，提取视频歌词/字幕

验证:
  1) 每编码 ASK_INTERVAL 个 chunk 后自动提问一次
  2) 淘汰不影响 ask() 的 snapshot/restore
  3) 收集所有回答，去重后拼接为完整歌词
  4) 全程不 OOM
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
VIDEO_PATH = os.environ.get(
    "VIDEO_PATH",
    "/root/autodl-tmp/diploma_project/temporal_encoding/202208312002.mp4",
)
REPORT_PATH = os.environ.get(
    "REPORT_PATH",
    "/root/autodl-tmp/diploma_project/temporal_encoding/test_eviction_exp_c_report.txt",
)

# ── 淘汰参数 (全自动) ──
# ⬇️ 应与实验 B 最终调优值一致; 更大 → 窗口更大 → 近期帧更多 → 回答更好
MAX_CACHE_TOKENS = 150_000  # 与实验 B 调优后保持一致

# ── 编码参数 ──
CHUNK_FRAMES = 4
SAMPLE_FPS = 2.0

# ── 提问参数 ──
ASK_INTERVAL = 25       # 每 25 个 chunk (~50 秒视频) 提问一次
MAX_NEW_TOKENS = 200

QUESTION = (
    "Read all text, lyrics, subtitles, or captions currently visible on screen. "
    "Output them verbatim. If there is no text, say 'No text visible'. "
    "Do NOT repeat previously mentioned text."
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


def get_vram_gb():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return {
            "allocated": round(torch.cuda.memory_allocated() / (1024 ** 3), 3),
            "reserved": round(torch.cuda.memory_reserved() / (1024 ** 3), 3),
        }
    return {}


def extract_frames_from_video(video_path, fps=2.0):
    """从视频中按指定 fps 采样帧。"""
    try:
        import decord
        from decord import VideoReader, cpu
        decord.bridge.set_bridge("native")
        vr = VideoReader(video_path, ctx=cpu(0))
        video_fps = vr.get_avg_fps()
        total_frames = len(vr)
        duration = total_frames / video_fps
        sample_interval = video_fps / fps
        indices = [int(i * sample_interval) for i in range(int(total_frames / sample_interval))]
        indices = [i for i in indices if i < total_frames]
        print(f"  Video: {video_path}")
        print(f"  Duration: {duration:.1f}s, Total: {total_frames} frames")
        print(f"  Sampling at {fps} fps → {len(indices)} frames")
        frames = []
        for idx in indices:
            frame = vr[idx].asnumpy()
            frames.append(Image.fromarray(frame))
        return frames, duration
    except ImportError:
        import cv2
        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps if video_fps > 0 else 0
        sample_interval = video_fps / fps
        indices = [int(i * sample_interval) for i in range(int(total_frames / sample_interval))]
        indices = [i for i in indices if i < total_frames]
        print(f"  Video: {video_path}")
        print(f"  Duration: {duration:.1f}s, Total: {total_frames} frames")
        print(f"  Sampling at {fps} fps → {len(indices)} frames")
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        cap.release()
        return frames, duration


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
            print("EXPERIMENT C: Sliding Window + Periodic Auto-Questioning")
            print("=" * 70)
            print(f"Report time: {datetime.now().isoformat(timespec='seconds')}")
            print(f"Ask interval: every {ASK_INTERVAL} chunks (~{ASK_INTERVAL * CHUNK_FRAMES / SAMPLE_FPS:.0f}s)")
            print(f"Eviction: max_cache_tokens={MAX_CACHE_TOKENS}, sink/window=auto")
            print()

            # ── 0) 检查 ──
            if not os.path.exists(MODEL_PATH):
                print(f"❌ Model not found: {MODEL_PATH}")
                return
            if not os.path.exists(VIDEO_PATH):
                print(f"❌ Video not found: {VIDEO_PATH}")
                return

            # ── 1) 加载模型 ──
            print("[1] Loading model...")
            from transformers import AutoProcessor
            device = "cuda"
            dtype = torch.bfloat16
            processor = AutoProcessor.from_pretrained(MODEL_PATH)
            model = StreamQwenModel.from_pretrained(
                MODEL_PATH, torch_dtype=dtype
            ).to(device)
            model.eval()
            print(f"  VRAM: {get_vram_gb()}")
            print()

            # ── 2) 提取帧 ──
            print("[2] Extracting frames...")
            all_frames, duration = extract_frames_from_video(VIDEO_PATH, fps=SAMPLE_FPS)
            total_frame_count = len(all_frames)
            print()

            # ── 3) 创建引擎 ──
            print("[3] Creating engine with eviction...")
            eviction_config = EvictionConfig(
                max_cache_tokens=MAX_CACHE_TOKENS,
            )
            engine = VideoStreamingInference(
                model, processor, device, eviction_config=eviction_config
            )
            print()

            # ── 4) 编码 + 周期性提问 ──
            print("[4] Encoding with periodic questioning...")
            all_answers = []
            chunk_count = 0
            t_start = time.time()

            for i in range(0, total_frame_count, CHUNK_FRAMES):
                chunk = all_frames[i : i + CHUNK_FRAMES]
                if len(chunk) == 0:
                    continue
                if len(chunk) % 2 != 0:
                    chunk.append(chunk[-1])

                engine.append_video_chunk(chunk, fps=SAMPLE_FPS)
                chunk_count += 1

                # 周期性提问
                if chunk_count % ASK_INTERVAL == 0:
                    time_pos = (i + CHUNK_FRAMES) / SAMPLE_FPS
                    print(f"\n  ─── Ask at chunk {chunk_count} "
                          f"(video ~{time_pos:.0f}s / {duration:.0f}s) ───")

                    info = engine.get_cache_info()
                    vram = get_vram_gb()
                    eviction_str = ""
                    if "eviction_stats" in info:
                        es = info["eviction_stats"]
                        eviction_str = f", evictions={es.get('total_evictions', 0)}"
                    print(f"  Cache: len={info['cache_seq_length']}, "
                          f"mem={info.get('cache_memory_gb', 0):.3f} GB, "
                          f"VRAM={vram.get('allocated', 0):.2f} GB"
                          f"{eviction_str}")

                    # 记录 ask 前 cache 长度
                    pre_ask_len = info['cache_seq_length']

                    answer, metrics = engine.ask(
                        QUESTION,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=True,
                        temperature=0.3,
                    )

                    # 验证 snapshot/restore
                    post_ask_len = engine.cache_manager.get_seq_length()
                    restored = (post_ask_len == pre_ask_len)

                    all_answers.append({
                        "chunk": chunk_count,
                        "time_pos": f"~{time_pos:.0f}s",
                        "answer": answer.strip(),
                        "ttft": metrics["ttft"],
                        "cache_restored": restored,
                    })
                    print(f"  Answer: {answer.strip()[:150]}...")
                    print(f"  TTFT: {metrics['ttft']:.3f}s, "
                          f"Cache restored: {'✅' if restored else '❌'}")

            t_total = time.time() - t_start

            # 最后一段如果还没问过，补一次
            if chunk_count % ASK_INTERVAL != 0:
                print(f"\n  ─── Final ask at chunk {chunk_count} ───")
                pre_ask_len = engine.cache_manager.get_seq_length()
                answer, metrics = engine.ask(
                    QUESTION,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=0.3,
                )
                post_ask_len = engine.cache_manager.get_seq_length()
                restored = (post_ask_len == pre_ask_len)
                all_answers.append({
                    "chunk": chunk_count,
                    "time_pos": f"~{total_frame_count / SAMPLE_FPS:.0f}s",
                    "answer": answer.strip(),
                    "ttft": metrics["ttft"],
                    "cache_restored": restored,
                })
                print(f"  Answer: {answer.strip()[:150]}...")

            print(f"\n  ✅ Done: {chunk_count} chunks, {len(all_answers)} questions asked")
            print(f"  Total time: {t_total:.1f}s")
            print()

            # ── 5) 汇总所有歌词 ──
            print("=" * 70)
            print("ALL COLLECTED LYRICS / SUBTITLES")
            print("=" * 70)

            seen_lines = set()
            unique_lyrics = []

            for entry in all_answers:
                print(f"\n[{entry['time_pos']}] (chunk {entry['chunk']}):")
                print(f"  {entry['answer']}")

                lines = entry["answer"].split("\n")
                for line in lines:
                    line_clean = line.strip().lower()
                    if (
                        line_clean
                        and line_clean not in seen_lines
                        and "no text" not in line_clean
                        and "no lyrics" not in line_clean
                        and "no subtitle" not in line_clean
                        and "no caption" not in line_clean
                        and "no visible" not in line_clean
                    ):
                        seen_lines.add(line_clean)
                        unique_lyrics.append(line.strip())

            print()
            print("=" * 70)
            print("DEDUPLICATED LYRICS (all unique lines)")
            print("=" * 70)
            for line in unique_lyrics:
                print(f"  {line}")
            print(f"\n  Total unique lines: {len(unique_lyrics)}")

            # ── 6) 总结 + 通过判定 ──
            print()
            print("=" * 70)
            print("SUMMARY & PASS/FAIL")
            print("=" * 70)
            final_info = engine.get_cache_info()
            print(f"  Video duration: {duration:.0f}s")
            print(f"  Total chunks: {chunk_count}")
            print(f"  Questions asked: {len(all_answers)}")
            print(f"  Unique lyric lines: {len(unique_lyrics)}")
            print(f"  Final cache_len: {final_info['cache_seq_length']}")
            print(f"  Total time: {t_total:.1f}s")

            if "eviction_stats" in final_info:
                es = final_info["eviction_stats"]
                print(f"  Total evictions: {es.get('total_evictions', 0)}")
                print(f"  Total tokens evicted: {es.get('total_tokens_evicted', 0)}")

            avg_ttft = sum(a["ttft"] for a in all_answers) / max(len(all_answers), 1)
            print(f"  Average TTFT: {avg_ttft:.3f}s")

            # 通过判定
            print()
            all_pass = True

            # C1: 不 OOM
            print(f"  ✅ [C1] No OOM — processed all {chunk_count} chunks")

            # C2: 所有 ask 后 cache 恢复
            all_restored = all(a["cache_restored"] for a in all_answers)
            if all_restored:
                print(f"  ✅ [C2] All {len(all_answers)} ask() calls "
                      f"correctly restored cache (snapshot/restore)")
            else:
                failed = [a for a in all_answers if not a["cache_restored"]]
                print(f"  ❌ [C2] {len(failed)} ask() calls did not restore cache!")
                all_pass = False

            # C3: 至少 N 次提问有非空回答
            non_empty = [
                a for a in all_answers
                if a["answer"]
                and "no text" not in a["answer"].lower()
                and "no visible" not in a["answer"].lower()
            ]
            if len(non_empty) >= 1:
                print(f"  ✅ [C3] {len(non_empty)}/{len(all_answers)} answers "
                      f"contained text/lyrics")
            else:
                print(f"  ⚠️ [C3] All answers were empty/no text — "
                      f"video may not contain visible text")

            # C4: 提取到歌词行
            if len(unique_lyrics) >= 1:
                print(f"  ✅ [C4] Extracted {len(unique_lyrics)} unique lyric lines")
            else:
                print(f"  ⚠️ [C4] No lyrics extracted — may be expected if video has no text")

            # C5: TTFT 合理 (< 10s)
            if avg_ttft < 10.0:
                print(f"  ✅ [C5] Average TTFT ({avg_ttft:.3f}s) < 10s")
            else:
                print(f"  ⚠️ [C5] Average TTFT ({avg_ttft:.3f}s) ≥ 10s — may be slow")

            print()
            if all_pass:
                print("🎉 EXPERIMENT C: ALL PASSED")
            else:
                print("⚠️ EXPERIMENT C: SOME CHECKS FAILED — see above")

        except torch.cuda.OutOfMemoryError:
            print(f"\n❌ EXPERIMENT C FAILED: CUDA OOM!")
            import traceback
            traceback.print_exc()

        except Exception as e:
            print(f"\n❌ EXPERIMENT C FAILED: {e}")
            import traceback
            traceback.print_exc()

        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr

    print(f"\nReport saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
