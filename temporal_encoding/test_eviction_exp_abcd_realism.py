"""
ABCD Realism Re-evaluation (no finetuning)
"""

import os
import re
import sys
import time
import json
from datetime import datetime
from statistics import mean
from typing import List, Dict, Tuple

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import StreamQwenModel, VideoStreamingInference, EvictionConfig

MODEL_PATH = os.environ.get("QWEN_MODEL_PATH", "/root/autodl-tmp/Qwen/Qwen2___5-VL-3B-Instruct")
VIDEO_PATH = os.environ.get("VIDEO_PATH", "/root/autodl-tmp/diploma_project/temporal_encoding/202208312002.mp4")
REPORT_PATH = os.environ.get("REPORT_PATH", "/root/autodl-tmp/diploma_project/temporal_encoding/test_eviction_exp_abcd_realism_report.txt")

MAX_CACHE_TOKENS = int(os.environ.get("MAX_CACHE_TOKENS", "150000"))
CHUNK_FRAMES = int(os.environ.get("CHUNK_FRAMES", "4"))
SAMPLE_FPS = float(os.environ.get("SAMPLE_FPS", "2.0"))
ASK_INTERVAL_C = int(os.environ.get("ASK_INTERVAL_C", "12"))
ASK_INTERVAL_D = int(os.environ.get("ASK_INTERVAL_D", "6"))

REFERENCE_LINES = [
    "用起伏的背影挡住哭泣的心","有些故事不必说给每个人听","许多眼睛看得太浅太近","错过我没被看见那个自己",
    "用简单的言语解开超载的心","有些情绪是该说给懂的人听","你的热泪比我激动怜惜","我发誓要更努力 更有勇气",
    "等下一个天亮","去上次牵手赏花那里散步好吗","有些积雪会自己融化","你的肩膀是我豁达的天堂",
    "把偷拍我看海的照片送我好吗","我喜欢我飞舞的头发","和飘着雨还是眺望的眼光","时间可以磨去我的棱角",
    "有些坚持却永远磨不掉","请容许我小小的骄傲","因为有你这样的依靠",
]

QUESTION_CURRENT_STRICT = "只读取当前画面可见的字幕/歌词正文。每行一条，不要解释，不要补写上下文。如果看不清或没有，输出：无文字。"
QUESTION_CUMULATIVE_STRICT = "基于截至当前时刻你已看过的所有画面，汇总出现过的歌词正文。每行一条，不要编号，不要输出人名品牌和制作信息。若不确定，宁可不写。"

class TeeWriter:
    def __init__(self, *writers): self._writers = writers
    def write(self, text):
        for w in self._writers: w.write(text)
        self.flush()
    def flush(self):
        for w in self._writers: w.flush()

def normalize(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^[\-\*\d\.)、\s]+", "", s)
    s = s.strip(" \t\r\n\"'“”‘’")
    s = s.replace("　", "")
    s = re.sub(r"\s+", "", s)
    return s

def is_match(ref: str, hyp: str) -> bool:
    r, h = normalize(ref), normalize(hyp)
    return (r in h) or (h in r)

def get_vram() -> Dict[str, float]:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return {
            "allocated": round(torch.cuda.memory_allocated() / (1024 ** 3), 3),
            "reserved": round(torch.cuda.memory_reserved() / (1024 ** 3), 3),
            "max_allocated": round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3),
        }
    return {}

def extract_frames(video_path: str, fps: float = 2.0) -> Tuple[List[Image.Image], float, int, float]:
    try:
        import decord
        from decord import VideoReader, cpu
        decord.bridge.set_bridge("native")
        vr = VideoReader(video_path, ctx=cpu(0))
        vf = vr.get_avg_fps(); n = len(vr); dur = n / vf
        step = vf / fps
        idx = [int(i * step) for i in range(int(n / step)) if int(i * step) < n]
        frames = [Image.fromarray(vr[i].asnumpy()) for i in idx]
        return frames, dur, n, vf
    except ImportError:
        import cv2
        cap = cv2.VideoCapture(video_path)
        vf = cap.get(cv2.CAP_PROP_FPS); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); dur = n / vf if vf > 0 else 0
        step = vf / fps
        idx = [int(i * step) for i in range(int(n / step)) if int(i * step) < n]
        frames = []
        for i in idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, fr = cap.read()
            if ok:
                frames.append(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
        cap.release()
        return frames, dur, n, vf

def parse_lines(answer: str) -> List[str]:
    bad = ["无文字", "no text", "看不清", "无法确认", "none", "没有"]
    bad_meta = ["安慕希", "董书含", "词", "曲", "编曲", "原唱", "演唱", "歌手", "下一个天亮》"]
    out = []
    for raw in answer.split("\n"):
        s = raw.strip()
        if not s: continue
        ls = s.lower()
        if any(k in ls for k in bad): continue
        if any(m in s for m in bad_meta): continue
        s = re.sub(r"^[\-\*\d\.)、\s]+", "", s).strip(" \t\r\n\"'“”‘’")
        if len(normalize(s)) < 6: continue
        out.append(s)
    return out

def evaluate_coverage(lines: List[str]) -> Dict[str, object]:
    matched = {}
    for ref in REFERENCE_LINES:
        hit = ""
        for hyp in lines:
            if is_match(ref, hyp):
                hit = hyp; break
        matched[ref] = hit
    n = sum(1 for v in matched.values() if v)
    return {"matched": matched, "coverage": n / len(REFERENCE_LINES), "hit": n, "total": len(REFERENCE_LINES)}

def evaluate_order(lines: List[str]) -> Dict[str, object]:
    norm_lines = [normalize(x) for x in lines]
    first_pos = []
    for ref in REFERENCE_LINES:
        rn = normalize(ref); pos = None
        for i, hypn in enumerate(norm_lines):
            if (rn in hypn) or (hypn in rn): pos = i; break
        first_pos.append(pos)
    pairs = ok_pairs = 0
    for i in range(len(first_pos)):
        for j in range(i + 1, len(first_pos)):
            if first_pos[i] is None or first_pos[j] is None: continue
            pairs += 1
            if first_pos[i] < first_pos[j]: ok_pairs += 1
    pair_acc = (ok_pairs / pairs) if pairs else 0.0
    seq = []
    for hypn in norm_lines:
        for ridx, ref in enumerate(REFERENCE_LINES):
            rn = normalize(ref)
            if (rn in hypn) or (hypn in rn): seq.append(ridx); break
    seen = set(); pred_order = []
    for ridx in seq:
        if ridx not in seen: seen.add(ridx); pred_order.append(ridx)
    n, m = len(REFERENCE_LINES), len(pred_order)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            dp[i][j] = 1 + dp[i + 1][j + 1] if i == pred_order[j] else max(dp[i + 1][j], dp[i][j + 1])
    lcs_len = dp[0][0]
    return {"pairwise_order_acc": pair_acc, "pair_count": pairs, "lcs_len": lcs_len, "lcs_ratio": lcs_len / n if n else 0.0, "pred_order": pred_order}

def eval_hallucination(lines: List[str]) -> Dict[str, object]:
    unmatched = [hyp for hyp in lines if not any(is_match(ref, hyp) for ref in REFERENCE_LINES)]
    return {"unmatched_count": len(unmatched), "total": len(lines), "unmatched_ratio": (len(unmatched) / len(lines)) if lines else 0.0, "examples": unmatched[:10]}

def run_exp_a(model, processor):
    print("\n" + "=" * 80); print("A) AUTO SINK/WINDOW SANITY"); print("=" * 80)
    frames = [Image.new("RGB", (1920, 1080), (30 * i, 100, 200 - 20 * i)) for i in range(4)]
    engine = VideoStreamingInference(model, processor, "cuda", eviction_config=EvictionConfig(max_cache_tokens=MAX_CACHE_TOKENS, sink_size=0, window_size=0))
    lens = []
    for _ in range(5):
        engine.append_video_chunk(frames, fps=SAMPLE_FPS)
        lens.append(engine.cache_manager.get_seq_length())
    ev = engine.cache_manager.evictor
    sink, window, avg_chunk = ev.effective_sink_size, ev.effective_window_size, ev._avg_chunk_tokens
    ok = (sink == lens[0]) and (window == MAX_CACHE_TOKENS - sink) and (avg_chunk > 0)
    print(f"cache_lens={lens}"); print(f"sink={sink}, window={window}, avg_chunk={avg_chunk:.1f}"); print("A_PASS=", ok)
    return {"pass": ok, "cache_lens": lens, "sink": sink, "window": window, "avg_chunk": avg_chunk}

def run_exp_b(model, processor, frames):
    print("\n" + "=" * 80); print("B) OOM-FREE + MEMORY PLATEAU"); print("=" * 80)
    engine = VideoStreamingInference(model, processor, "cuda", eviction_config=EvictionConfig(max_cache_tokens=MAX_CACHE_TOKENS))
    cache_trace = []; vram_trace = []; t0 = time.time()
    for i in range(0, len(frames), CHUNK_FRAMES):
        chunk = frames[i:i + CHUNK_FRAMES]
        if not chunk: continue
        if len(chunk) % 2: chunk.append(chunk[-1])
        engine.append_video_chunk(chunk, fps=SAMPLE_FPS)
        if ((i // CHUNK_FRAMES + 1) % 10 == 0) or (i == 0):
            info = engine.get_cache_info(); v = get_vram()
            cache_trace.append(info["cache_seq_length"]); vram_trace.append(v.get("allocated", 0.0))
            print(f"chunk={i//CHUNK_FRAMES+1}, cache={info['cache_seq_length']}, vram={v.get('allocated',0.0):.2f}GB")
    elapsed = time.time() - t0
    fin = engine.get_cache_info(); es = fin.get("eviction_stats", {})
    cache_ok = fin["cache_seq_length"] <= MAX_CACHE_TOKENS
    evict_ok = es.get("total_evictions", 0) > 0
    plateau_ok = (max(cache_trace) - min(cache_trace[-3:])) < MAX_CACHE_TOKENS * 0.2 if len(cache_trace) >= 3 else True
    all_ok = cache_ok and evict_ok and plateau_ok
    print(f"final_cache={fin['cache_seq_length']}, evictions={es.get('total_evictions', 0)}, elapsed={elapsed:.1f}s")
    print(f"B_PASS={all_ok} (cache_ok={cache_ok}, evict_ok={evict_ok}, plateau_ok={plateau_ok})")
    return {"pass": all_ok, "elapsed": elapsed, "final_cache": fin["cache_seq_length"], "evictions": es.get("total_evictions", 0), "cache_trace": cache_trace, "vram_trace": vram_trace, "cache_ok": cache_ok, "evict_ok": evict_ok, "plateau_ok": plateau_ok}

def run_exp_c(model, processor, frames):
    print("\n" + "=" * 80); print("C) PERIODIC CURRENT-SCREEN OCR (REALISM-FIRST)"); print("=" * 80)
    engine = VideoStreamingInference(model, processor, "cuda", eviction_config=EvictionConfig(max_cache_tokens=MAX_CACHE_TOKENS))
    asks = abstain = restore_fail = line_count = 0; ttfts = []
    for i in range(0, len(frames), CHUNK_FRAMES):
        chunk = frames[i:i + CHUNK_FRAMES]
        if not chunk: continue
        if len(chunk) % 2: chunk.append(chunk[-1])
        engine.append_video_chunk(chunk, fps=SAMPLE_FPS)
        chunk_id = i // CHUNK_FRAMES + 1
        if (chunk_id % ASK_INTERVAL_C != 0) and (i + CHUNK_FRAMES < len(frames)): continue
        asks += 1
        pre = engine.cache_manager.get_seq_length()
        ans, m = engine.ask(QUESTION_CURRENT_STRICT, max_new_tokens=128, do_sample=False, temperature=0.1)
        post = engine.cache_manager.get_seq_length()
        if pre != post: restore_fail += 1
        if "无文字" in ans: abstain += 1
        ls = parse_lines(ans); line_count += len(ls); ttfts.append(m.get("ttft", 0.0))
        print(f"ask#{asks} chunk={chunk_id} restore={'OK' if pre==post else 'BAD'} lines={len(ls)} ans={ans.strip()[:80]}")
    abstain_ratio = (abstain / asks) if asks else 1.0; avg_ttft = mean(ttfts) if ttfts else 0.0
    pass_c = (restore_fail == 0) and (asks >= 5)
    print(f"C_PASS={pass_c}, asks={asks}, abstain_ratio={abstain_ratio:.2f}, line_count={line_count}, avg_ttft={avg_ttft:.3f}")
    return {"pass": pass_c, "asks": asks, "abstain_ratio": abstain_ratio, "line_count": line_count, "restore_fail": restore_fail, "avg_ttft": avg_ttft}

def run_exp_d(model, processor, frames):
    print("\n" + "=" * 80); print("D) CUMULATIVE LYRICS + ORDER/HALLUCINATION AUDIT"); print("=" * 80)
    engine = VideoStreamingInference(model, processor, "cuda", eviction_config=EvictionConfig(max_cache_tokens=MAX_CACHE_TOKENS))
    seen = {}; asks = restore_fail = 0
    for i in range(0, len(frames), CHUNK_FRAMES):
        chunk = frames[i:i + CHUNK_FRAMES]
        if not chunk: continue
        if len(chunk) % 2: chunk.append(chunk[-1])
        engine.append_video_chunk(chunk, fps=SAMPLE_FPS)
        chunk_id = i // CHUNK_FRAMES + 1
        if (chunk_id % ASK_INTERVAL_D != 0) and (i + CHUNK_FRAMES < len(frames)): continue
        asks += 1; tsec = (i + CHUNK_FRAMES) / SAMPLE_FPS
        pre = engine.cache_manager.get_seq_length()
        ans_cum, _ = engine.ask(QUESTION_CUMULATIVE_STRICT, max_new_tokens=320, do_sample=False, temperature=0.1)
        ans_cur, _ = engine.ask(QUESTION_CURRENT_STRICT, max_new_tokens=128, do_sample=False, temperature=0.1)
        post = engine.cache_manager.get_seq_length()
        if pre != post: restore_fail += 1
        for ln in parse_lines(ans_cum) + parse_lines(ans_cur):
            k = normalize(ln)
            if not k: continue
            if k not in seen: seen[k] = {"line": ln, "chunk": chunk_id, "time": f"~{tsec:.0f}s"}
        print(f"ask#{asks} chunk={chunk_id} seen={len(seen)} restore={'OK' if pre==post else 'BAD'}")
    ordered = sorted(seen.values(), key=lambda x: int(x["chunk"])); lines = [x["line"] for x in ordered]
    cov = evaluate_coverage(lines); order = evaluate_order(lines); hall = eval_hallucination(lines)
    print("\nDEDUP_LYRICS:")
    for x in ordered: print(f"[{x['time']}] {x['line']}")
    print(f"\ncoverage={cov['hit']}/{cov['total']} ({cov['coverage']*100:.1f}%)")
    print(f"order_pair_acc={order['pairwise_order_acc']*100:.2f}% (pairs={order['pair_count']})")
    print(f"order_lcs={order['lcs_len']}/{len(REFERENCE_LINES)} ({order['lcs_ratio']*100:.2f}%)")
    print(f"hallucination={hall['unmatched_count']}/{hall['total']} ({hall['unmatched_ratio']*100:.1f}%)")
    pass_d = (cov["coverage"] >= 0.80) and (order["lcs_ratio"] >= 0.90) and (hall["unmatched_ratio"] <= 0.35) and (restore_fail == 0)
    print(f"D_PASS={pass_d}, asks={asks}, restore_fail={restore_fail}")
    return {"pass": pass_d, "asks": asks, "restore_fail": restore_fail, "total_unique": len(lines), "coverage": cov, "order": order, "hall": hall}

def main():
    report_dir = os.path.dirname(REPORT_PATH)
    if report_dir: os.makedirs(report_dir, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        tee = TeeWriter(sys.stdout, f)
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = tee; sys.stderr = tee
        try:
            print("=" * 80); print("ABCD REALISM RE-EVALUATION (NO FINETUNING)"); print("=" * 80)
            print(f"time={datetime.now().isoformat(timespec='seconds')}")
            print(f"model={MODEL_PATH}"); print(f"video={VIDEO_PATH}")
            print(f"cfg: max_cache={MAX_CACHE_TOKENS}, chunk={CHUNK_FRAMES}, fps={SAMPLE_FPS}, askC={ASK_INTERVAL_C}, askD={ASK_INTERVAL_D}")
            print("notes: realism-first => coverage + order + hallucination + cache-restore\n")
            if not os.path.exists(MODEL_PATH): print(f"FATAL: model not found: {MODEL_PATH}"); return
            if not os.path.exists(VIDEO_PATH): print(f"FATAL: video not found: {VIDEO_PATH}"); return
            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained(MODEL_PATH)
            model = StreamQwenModel.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16).to("cuda").eval()
            print("vram_after_load=", get_vram())
            frames, dur, nraw, vf = extract_frames(VIDEO_PATH, SAMPLE_FPS)
            print(f"video_duration={dur:.1f}s, raw_frames={nraw}, sampled_frames={len(frames)}, raw_fps={vf:.2f}")
            ta = run_exp_a(model, processor)
            tb = run_exp_b(model, processor, frames)
            tc = run_exp_c(model, processor, frames)
            td = run_exp_d(model, processor, frames)
            final = {"A": ta, "B": tb, "C": tc, "D": td, "all_pass": bool(ta["pass"] and tb["pass"] and tc["pass"] and td["pass"])}
            print("\n" + "=" * 80); print("FINAL DECISION"); print("=" * 80)
            print(json.dumps(final, ensure_ascii=False, indent=2))
            if final["all_pass"]:
                print("\n结论: 在不微调前提下，Qwen2.5-VL + sink/window 流式改造可用，且在本视频上具备较真实歌词恢复能力。")
            else:
                print("\n结论: 在不微调前提下，工程方案可运行，但真实性指标尚未全部达标，需要继续迭代策略。")
        finally:
            sys.stdout = old_out; sys.stderr = old_err
    print(f"report saved: {REPORT_PATH}")

if __name__ == "__main__":
    main()
