"""
D Trace v2: anti-repetition + transparent merge
"""
import os
import re
import time
from datetime import datetime
from typing import Dict, List, Tuple

import torch
from PIL import Image

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import StreamQwenModel, VideoStreamingInference, EvictionConfig
from transformers import AutoProcessor

MODEL_PATH = os.environ.get("QWEN_MODEL_PATH", "/root/autodl-tmp/Qwen/Qwen2___5-VL-3B-Instruct")
VIDEO_PATH = os.environ.get("VIDEO_PATH", "/root/autodl-tmp/diploma_project/temporal_encoding/202208312002.mp4")
REPORT_PATH = os.environ.get("REPORT_PATH", "/root/autodl-tmp/diploma_project/temporal_encoding/test_eviction_exp_d_trace_report_v2.txt")

MAX_CACHE_TOKENS = int(os.environ.get("MAX_CACHE_TOKENS", "150000"))
CHUNK_FRAMES = int(os.environ.get("CHUNK_FRAMES", "4"))
SAMPLE_FPS = float(os.environ.get("SAMPLE_FPS", "2.0"))
ASK_INTERVAL = int(os.environ.get("ASK_INTERVAL", "6"))
MAX_ASKS = int(os.environ.get("MAX_ASKS", "0"))  # 0 = all

# 为了抑制静态画面复读，降低生成长度并启用轻采样
CUM_MAX_NEW_TOKENS = int(os.environ.get("CUM_MAX_NEW_TOKENS", "128"))
CUR_MAX_NEW_TOKENS = int(os.environ.get("CUR_MAX_NEW_TOKENS", "72"))
GEN_TEMPERATURE = float(os.environ.get("GEN_TEMPERATURE", "0.35"))
GEN_TOP_P = float(os.environ.get("GEN_TOP_P", "0.85"))

QUESTION_CUM = "基于截至当前时刻你已看过的所有画面，汇总出现过的歌词正文。每行一条，不要编号，不要输出人名品牌和制作信息。若不确定，宁可不写。"
QUESTION_CUR = "只读取当前画面可见的字幕/歌词正文。每行一条，不要解释，不要补写上下文。如果看不清或没有，输出：无文字。"


def normalize(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^[\-\*\d\.)、\s]+", "", s)
    s = s.strip(" \t\r\n\"'“”‘’")
    s = s.replace("　", "")
    s = re.sub(r"\s+", "", s)
    return s


def similar(a: str, b: str) -> bool:
    na, nb = normalize(a), normalize(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    if (na in nb or nb in na) and abs(len(na) - len(nb)) <= 3:
        return True
    return False


def strip_generation_loop(answer: str, max_same_line_repeat: int = 1, max_lines: int = 16) -> Tuple[str, Dict[str, int]]:
    """抑制模型在静态画面中的行级复读（仅对输出文本做清洗，不改模型权重）。"""
    raw_lines = [x.strip() for x in answer.split("\n") if x.strip()]
    if not raw_lines:
        return answer, {"raw_lines": 0, "removed_loop": 0, "trim_tail": 0}

    kept: List[str] = []
    seen_count: Dict[str, int] = {}
    removed_loop = 0

    for ln in raw_lines:
        k = normalize(ln)
        if not k:
            continue
        seen_count[k] = seen_count.get(k, 0) + 1
        if seen_count[k] > max_same_line_repeat:
            removed_loop += 1
            continue
        # 连续重复行也直接压掉
        if kept and normalize(kept[-1]) == k:
            removed_loop += 1
            continue
        kept.append(ln)

    trim_tail = 0
    if len(kept) > max_lines:
        trim_tail = len(kept) - max_lines
        kept = kept[:max_lines]

    cleaned = "\n".join(kept)
    return cleaned, {"raw_lines": len(raw_lines), "removed_loop": removed_loop, "trim_tail": trim_tail}


def parse_lines(answer: str) -> List[str]:
    bad = ["无文字", "no text", "看不清", "无法确认", "none", "没有"]
    bad_meta = ["安慕希", "董书含", "词", "曲", "编曲", "原唱", "演唱", "歌手", "下一个天亮》", "浙江卫视", "酷狗音乐"]
    out: List[str] = []
    for raw in answer.split("\n"):
        s = raw.strip()
        if not s:
            continue
        ls = s.lower()
        if any(k in ls for k in bad):
            continue
        if any(m in s for m in bad_meta):
            continue
        s = re.sub(r"^[\-\*\d\.)、\s]+", "", s).strip(" \t\r\n\"'“”‘’")
        if len(normalize(s)) < 6:
            continue
        out.append(s)
    return out


def dedup_in_answer(lines: List[str]) -> Tuple[List[str], Dict[str, int]]:
    stats = {"raw": len(lines), "dedup_exact": 0, "dedup_near": 0}
    uniq: List[str] = []
    for s in lines:
        if any(normalize(s) == normalize(t) for t in uniq):
            stats["dedup_exact"] += 1
            continue
        hit = -1
        for i, t in enumerate(uniq):
            if similar(s, t):
                hit = i
                break
        if hit >= 0:
            stats["dedup_near"] += 1
            if len(normalize(s)) > len(normalize(uniq[hit])):
                uniq[hit] = s
            continue
        uniq.append(s)
    return uniq, stats


def extract_frames(video_path: str, fps: float = 2.0):
    try:
        import decord
        from decord import VideoReader, cpu

        decord.bridge.set_bridge("native")
        vr = VideoReader(video_path, ctx=cpu(0))
        vf = vr.get_avg_fps()
        n = len(vr)
        step = vf / fps
        idx = [int(i * step) for i in range(int(n / step))]
        idx = [i for i in idx if i < n]
        frames = [Image.fromarray(vr[i].asnumpy()) for i in idx]
        return frames, n / vf, n, vf
    except Exception:
        import cv2

        cap = cv2.VideoCapture(video_path)
        vf = cap.get(cv2.CAP_PROP_FPS)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = vf / fps
        idx = [int(i * step) for i in range(int(n / step))]
        idx = [i for i in idx if i < n]
        frames = []
        for i in idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, fr = cap.read()
            if ok:
                frames.append(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
        cap.release()
        return frames, (n / vf if vf > 0 else 0), n, vf


def main():
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        def w(x=""):
            print(x)
            f.write(x + "\n")
            f.flush()

        w("=" * 90)
        w("EXPERIMENT D TRACE REPORT V2 (ANTI-REPETITION)")
        w("=" * 90)
        w(f"time={datetime.now().isoformat(timespec='seconds')}")
        w(f"video={VIDEO_PATH}")
        w(
            f"cfg: max_cache={MAX_CACHE_TOKENS}, chunk={CHUNK_FRAMES}, fps={SAMPLE_FPS}, "
            f"ask_interval={ASK_INTERVAL}, max_asks={MAX_ASKS}, "
            f"cum_max_new={CUM_MAX_NEW_TOKENS}, cur_max_new={CUR_MAX_NEW_TOKENS}, "
            f"temp={GEN_TEMPERATURE}, top_p={GEN_TOP_P}"
        )

        processor = AutoProcessor.from_pretrained(MODEL_PATH)
        model = StreamQwenModel.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16).to("cuda").eval()
        frames, dur, total, vf = extract_frames(VIDEO_PATH, SAMPLE_FPS)
        w(f"video_duration={dur:.1f}s, raw_frames={total}, sampled_frames={len(frames)}, raw_fps={vf:.2f}")

        engine = VideoStreamingInference(
            model, processor, "cuda", eviction_config=EvictionConfig(max_cache_tokens=MAX_CACHE_TOKENS)
        )

        seen: Dict[str, Dict[str, str]] = {}
        ask_n = 0
        t0 = time.time()

        for i in range(0, len(frames), CHUNK_FRAMES):
            chunk = frames[i : i + CHUNK_FRAMES]
            if not chunk:
                continue
            if len(chunk) % 2:
                chunk.append(chunk[-1])
            engine.append_video_chunk(chunk, fps=SAMPLE_FPS)

            chunk_id = i // CHUNK_FRAMES + 1
            if (chunk_id % ASK_INTERVAL != 0) and (i + CHUNK_FRAMES < len(frames)):
                continue

            if MAX_ASKS > 0 and (ask_n + 1) > MAX_ASKS:
                break
            ask_n += 1

            tsec = (i + CHUNK_FRAMES) / SAMPLE_FPS
            pre = engine.cache_manager.get_seq_length()
            ans_cum_raw, _ = engine.ask(
                QUESTION_CUM,
                max_new_tokens=CUM_MAX_NEW_TOKENS,
                do_sample=True,
                temperature=GEN_TEMPERATURE,
                top_p=GEN_TOP_P,
            )
            ans_cur_raw, _ = engine.ask(
                QUESTION_CUR,
                max_new_tokens=CUR_MAX_NEW_TOKENS,
                do_sample=True,
                temperature=GEN_TEMPERATURE,
                top_p=GEN_TOP_P,
            )
            post = engine.cache_manager.get_seq_length()

            ans_cum, loop_stat_c = strip_generation_loop(ans_cum_raw)
            ans_cur, loop_stat_r = strip_generation_loop(ans_cur_raw)

            raw_cum = parse_lines(ans_cum)
            raw_cur = parse_lines(ans_cur)
            cand, st = dedup_in_answer(raw_cum + raw_cur)

            w("\n" + "-" * 90)
            w(f"ASK #{ask_n} | chunk={chunk_id} | t~{tsec:.0f}s | cache_pre={pre} | restore={'OK' if pre==post else 'BAD'}")
            w("[RAW cumulative answer]")
            w(ans_cum_raw.strip() if ans_cum_raw.strip() else "(empty)")
            w("[RAW current answer]")
            w(ans_cur_raw.strip() if ans_cur_raw.strip() else "(empty)")
            w(f"[loop_clean_stats] cum={loop_stat_c}, cur={loop_stat_r}")
            w("[CLEANED cumulative answer]")
            w(ans_cum.strip() if ans_cum.strip() else "(empty)")
            w("[CLEANED current answer]")
            w(ans_cur.strip() if ans_cur.strip() else "(empty)")

            w("[Parsed before in-ask dedup]")
            for j, x in enumerate(raw_cum + raw_cur, 1):
                w(f"  {j}. {x}")

            w("[Parsed after in-ask dedup]")
            if cand:
                for j, x in enumerate(cand, 1):
                    w(f"  {j}. {x}")
            else:
                w("  (none)")
            w(f"in_ask_stats={st}")

            added, duplicated = [], []
            for ln in cand:
                k = normalize(ln)
                if not k:
                    continue
                if k not in seen:
                    seen[k] = {"line": ln, "chunk": str(chunk_id), "time": f"~{tsec:.0f}s"}
                    added.append(ln)
                else:
                    duplicated.append(ln)

            w("[Merge action]")
            w(f"  added={len(added)}, duplicated={len(duplicated)}, total_seen={len(seen)}")
            if added:
                w("  + Added:")
                for x in added:
                    w(f"    + {x}")
            if duplicated:
                w("  = Duplicated:")
                for x in duplicated:
                    w(f"    = {x}")

        ordered = sorted(seen.values(), key=lambda x: int(x["chunk"]))
        w("\n" + "=" * 90)
        w("FINAL STITCHED LYRICS")
        w("=" * 90)
        for x in ordered:
            w(f"[{x['time']}] {x['line']}")
        w(f"total_unique={len(ordered)}")
        w(f"elapsed={time.time()-t0:.1f}s, asks={ask_n}, final_cache={engine.get_cache_info()['cache_seq_length']}")

    print("report saved:", REPORT_PATH)


if __name__ == "__main__":
    main()
