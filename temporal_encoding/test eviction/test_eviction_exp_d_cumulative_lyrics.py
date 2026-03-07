"""
实验 D: 全视频滑动 + 截至当前时刻累计歌词抽取

目标:
1) 使用现有流式滑动缓存方案处理完整视频。
2) 周期性提问“截至目前出现过的所有歌词/字幕”，而非只问当前画面。
3) 自动汇总、去重、按首次出现时间排序，得到尽可能完整的歌词清单。
4) 若结果不足，便于迭代 prompt / 提问频率。
"""

import os
import re
import sys
import time
import gc
from datetime import datetime
from typing import List, Dict

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import StreamQwenModel, VideoStreamingInference, EvictionConfig

MODEL_PATH = os.environ.get("QWEN_MODEL_PATH", "/root/autodl-tmp/Qwen/Qwen2___5-VL-3B-Instruct")
VIDEO_PATH = os.environ.get("VIDEO_PATH", "/root/autodl-tmp/diploma_project/temporal_encoding/202208312002.mp4")
REPORT_PATH = os.environ.get(
    "REPORT_PATH",
    "/root/autodl-tmp/diploma_project/temporal_encoding/test_eviction_exp_d_cumulative_lyrics_report.txt",
)

MAX_CACHE_TOKENS = int(os.environ.get("MAX_CACHE_TOKENS", "150000"))
CHUNK_FRAMES = int(os.environ.get("CHUNK_FRAMES", "4"))
SAMPLE_FPS = float(os.environ.get("SAMPLE_FPS", "2.0"))
ASK_INTERVAL = int(os.environ.get("ASK_INTERVAL", "5"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "256"))

QUESTION_CUMULATIVE = (
    "请基于你截至当前时刻在视频中看见过的所有画面，"
    "只汇总歌词/字幕的正文句子（不要标题、歌手名、品牌词、制作信息）。"
    "要求：\n"
    "1) 只输出你确认看见过的歌词正文，不要猜测；\n"
    "2) 每行一条，尽量完整句子；\n"
    "3) 不要输出：安慕希、歌名、人名、词/曲/编曲/原唱等信息；\n"
    "5) 尽量补全此前漏掉的歌词句子。"
    "4) 若当前还无法确认任何歌词正文，只输出：无文字。"
)

QUESTION_CURRENT = (
    "请读取当前画面可见的歌词/字幕。"
    "如果看到两行歌词，请两行都输出（每行一条）；"
    "如果只看到一行，就输出一行；"
    "如果没有可见歌词，输出：无文字。"
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
            "max_allocated": round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3),
        }
    return {}


def extract_frames_from_video(video_path, fps=2.0):
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
        frames = [Image.fromarray(vr[idx].asnumpy()) for idx in indices]
        return frames, duration, total_frames, video_fps
    except ImportError:
        import cv2

        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps if video_fps > 0 else 0
        sample_interval = video_fps / fps
        indices = [int(i * sample_interval) for i in range(int(total_frames / sample_interval))]
        indices = [i for i in indices if i < total_frames]
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        cap.release()
        return frames, duration, total_frames, video_fps


def normalize_line(line: str) -> str:
    s = line.strip()
    s = re.sub(r"^[\-\*\d\.)、\s]+", "", s)
    s = s.strip(" \t\r\n\"'“”‘’")
    return s


def parse_lines(answer: str) -> List[str]:
    bad = ["无文字", "no text", "看不清", "无法确认", "none", "没有"]
    bad_meta = ["安慕希", "董书含", "词 ", "曲 ", "编曲", "原唱", "歌手", "演唱"]

    lines = []
    for raw in answer.split("\n"):
        s = normalize_line(raw)
        if not s:
            continue
        ls = s.lower()
        if any(k in ls for k in bad):
            continue
        if any(m in s for m in bad_meta):
            continue
        # 过滤过短碎片（如“安”“董书”）
        if len(s) < 6:
            continue
        # 过滤“xx可以”这类明显非歌词碎句
        if s.endswith("可以") and len(s) <= 8:
            continue
        lines.append(s)
    return lines


def main():
    report_dir = os.path.dirname(REPORT_PATH)
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        tee = TeeWriter(sys.stdout, f)
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = tee
        sys.stderr = tee

        try:
            print("=" * 78)
            print("EXPERIMENT D: CUMULATIVE LYRICS OVER FULL VIDEO")
            print("=" * 78)
            print(f"Report time: {datetime.now().isoformat(timespec='seconds')}")
            print(f"VIDEO_PATH={VIDEO_PATH}")
            print(f"MAX_CACHE_TOKENS={MAX_CACHE_TOKENS}, CHUNK_FRAMES={CHUNK_FRAMES}, SAMPLE_FPS={SAMPLE_FPS}")
            print(f"ASK_INTERVAL={ASK_INTERVAL}, MAX_NEW_TOKENS={MAX_NEW_TOKENS}")
            print()

            if not os.path.exists(MODEL_PATH):
                print(f"❌ Model not found: {MODEL_PATH}")
                return
            if not os.path.exists(VIDEO_PATH):
                print(f"❌ Video not found: {VIDEO_PATH}")
                return

            print("[1] Loading model...")
            from transformers import AutoProcessor

            processor = AutoProcessor.from_pretrained(MODEL_PATH)
            model = StreamQwenModel.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16).to("cuda")
            model.eval()
            print(f"  VRAM after load: {get_vram_gb()}")
            print()

            print("[2] Extracting sampled frames...")
            frames, duration, total_frames, video_fps = extract_frames_from_video(VIDEO_PATH, fps=SAMPLE_FPS)
            total_sampled = len(frames)
            expected_chunks = (total_sampled + CHUNK_FRAMES - 1) // CHUNK_FRAMES
            print(f"  Raw video: duration={duration:.1f}s, fps={video_fps:.2f}, frames={total_frames}")
            print(f"  Sampled: {total_sampled} frames => {expected_chunks} chunks")
            print()

            print("[3] Running streaming inference with sliding window...")
            config = EvictionConfig(max_cache_tokens=MAX_CACHE_TOKENS)
            engine = VideoStreamingInference(model, processor, "cuda", eviction_config=config)

            seen_map: Dict[str, Dict[str, str]] = {}
            ask_records = []
            chunk_count = 0
            t_start = time.time()

            for i in range(0, total_sampled, CHUNK_FRAMES):
                chunk = frames[i: i + CHUNK_FRAMES]
                if not chunk:
                    continue
                if len(chunk) % 2 != 0:
                    chunk.append(chunk[-1])

                engine.append_video_chunk(chunk, fps=SAMPLE_FPS)
                chunk_count += 1

                should_ask = (chunk_count % ASK_INTERVAL == 0) or (i + CHUNK_FRAMES >= total_sampled)
                if not should_ask:
                    continue

                time_pos = (i + CHUNK_FRAMES) / SAMPLE_FPS
                info = engine.get_cache_info()
                pre_len = info["cache_seq_length"]

                ans_cum, m1 = engine.ask(
                    QUESTION_CUMULATIVE,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    temperature=0.1,
                )
                post_len1 = engine.cache_manager.get_seq_length()

                ans_cur, m2 = engine.ask(
                    QUESTION_CURRENT,
                    max_new_tokens=96,
                    do_sample=False,
                    temperature=0.1,
                )
                post_len2 = engine.cache_manager.get_seq_length()

                restore_ok = (post_len1 == pre_len) and (post_len2 == pre_len)

                lines_cum = parse_lines(ans_cum)
                lines_cur = parse_lines(ans_cur)
                merged = lines_cum + [x for x in lines_cur if x not in lines_cum]

                new_lines = []
                for ln in merged:
                    key = ln.lower()
                    if key not in seen_map:
                        seen_map[key] = {
                            "line": ln,
                            "first_time": f"~{time_pos:.0f}s",
                            "first_chunk": str(chunk_count),
                        }
                        new_lines.append(ln)

                ask_records.append(
                    {
                        "chunk": chunk_count,
                        "time": f"~{time_pos:.0f}s",
                        "cache_len": pre_len,
                        "ttft_cum": m1["ttft"],
                        "ttft_cur": m2["ttft"],
                        "restore_ok": restore_ok,
                        "ans_cum": ans_cum.strip(),
                        "ans_cur": ans_cur.strip(),
                        "new_lines": new_lines,
                    }
                )

                print(f"  Ask@chunk={chunk_count:>3d}, t~{time_pos:>5.0f}s, cache={pre_len}, restore={'✅' if restore_ok else '❌'}")
                print(f"    cumulative: {ans_cum.strip()[:140]}")
                print(f"    current   : {ans_cur.strip()[:140]}")
                if new_lines:
                    print(f"    + New lines ({len(new_lines)}): {new_lines}")
                else:
                    print("    + New lines: 0")

            total_time = time.time() - t_start
            final_info = engine.get_cache_info()

            print("\n" + "=" * 78)
            print("FINAL DEDUPLICATED LYRICS (ORDERED BY FIRST APPEARANCE)")
            print("=" * 78)
            ordered = sorted(seen_map.values(), key=lambda x: int(x["first_chunk"]))
            for row in ordered:
                print(f"[{row['first_time']}] {row['line']}")
            print(f"\nTotal unique lines: {len(ordered)}")

            print("\n" + "=" * 78)
            print("ANALYSIS")
            print("=" * 78)
            restored_all = all(r["restore_ok"] for r in ask_records)
            avg_ttft_cum = sum(r["ttft_cum"] for r in ask_records) / max(len(ask_records), 1)
            avg_ttft_cur = sum(r["ttft_cur"] for r in ask_records) / max(len(ask_records), 1)

            print(f"Total chunks processed: {chunk_count}/{expected_chunks}")
            print(f"Total asks: {len(ask_records)}")
            print(f"All snapshot/restore valid: {restored_all}")
            print(f"Avg TTFT cumulative ask: {avg_ttft_cum:.3f}s")
            print(f"Avg TTFT current ask: {avg_ttft_cur:.3f}s")
            print(f"Final cache_len: {final_info['cache_seq_length']}")
            if "eviction_stats" in final_info:
                es = final_info["eviction_stats"]
                print(f"Total evictions: {es.get('total_evictions', 0)}")
                print(f"Total tokens evicted: {es.get('total_tokens_evicted', 0)}")
            print(f"Total runtime: {total_time:.1f}s")

            print("\nPass/Fail checks:")
            ok1 = chunk_count == expected_chunks
            ok2 = restored_all
            ok3 = len(ordered) >= 6
            ok4 = final_info["cache_seq_length"] <= MAX_CACHE_TOKENS
            print(f"  [D1] Full-video sliding completed: {'✅' if ok1 else '❌'}")
            print(f"  [D2] All ask restore cache: {'✅' if ok2 else '❌'}")
            print(f"  [D3] Extracted enough unique lyric lines (>=6): {'✅' if ok3 else '❌'}")
            print(f"  [D4] cache_len <= max_cache_tokens: {'✅' if ok4 else '❌'}")

            if ok1 and ok2 and ok3 and ok4:
                print("\n🎉 EXPERIMENT D PASSED")
            else:
                print("\n⚠️ EXPERIMENT D NOT FULLY PASSED — please iterate prompt/frequency.")

        except Exception as e:
            print(f"\n❌ EXPERIMENT D FAILED: {e}")
            import traceback
            traceback.print_exc()

        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    print(f"\nReport saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
