"""
实验 D-Iter: 全视频累计歌词抽取 + 与参考答案自动比对

不做模型微调，仅通过推理策略迭代验证能力上限。
"""
import os
import re
import sys
import time
from datetime import datetime
from typing import List, Dict

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import StreamQwenModel, VideoStreamingInference, EvictionConfig

MODEL_PATH = os.environ.get("QWEN_MODEL_PATH", "/root/autodl-tmp/Qwen/Qwen2___5-VL-3B-Instruct")
VIDEO_PATH = os.environ.get("VIDEO_PATH", "/root/autodl-tmp/diploma_project/temporal_encoding/202208312002.mp4")
REPORT_PATH = os.environ.get("REPORT_PATH", "/root/autodl-tmp/diploma_project/temporal_encoding/test_eviction_exp_d_iter_report.txt")

MAX_CACHE_TOKENS = int(os.environ.get("MAX_CACHE_TOKENS", "150000"))
CHUNK_FRAMES = int(os.environ.get("CHUNK_FRAMES", "4"))
SAMPLE_FPS = float(os.environ.get("SAMPLE_FPS", "2.0"))
ASK_INTERVAL = int(os.environ.get("ASK_INTERVAL", "5"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "320"))

REFERENCE_LINES = [
    "用起伏的背影挡住哭泣的心",
    "有些故事不必说给每个人听",
    "许多眼睛看得太浅太近",
    "错过我没被看见那个自己",
    "用简单的言语解开超载的心",
    "有些情绪是该说给懂的人听",
    "你的热泪比我激动怜惜",
    "我发誓要更努力 更有勇气",
    "等下一个天亮",
    "去上次牵手赏花那里散步好吗",
    "有些积雪会自己融化",
    "你的肩膀是我豁达的天堂",
    "把偷拍我看海的照片送我好吗",
    "我喜欢我飞舞的头发",
    "和飘着雨还是眺望的眼光",
    "时间可以磨去我的棱角",
    "有些坚持却永远磨不掉",
    "请容许我小小的骄傲",
    "因为有你这样的依靠",
]

QUESTION_CUMULATIVE = (
    "请基于你截至当前时刻在视频中看见过的所有画面，汇总出现过的歌词正文。\n"
    "要求：\n"
    "1) 只输出歌词句子本身，不要输出标题/人名/品牌词/制作信息；\n"
    "2) 每行一条，不要编号；\n"
    "3) 尽量补全之前漏掉的歌词句子；\n"
    "4) 若确实没有可确认歌词，输出：无文字。"
)

QUESTION_CURRENT = (
    "请读取当前画面可见的歌词字幕。"
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


def normalize(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^[\-\*\d\.)、\s]+", "", s)
    s = s.strip(" \t\r\n\"'“”‘’")
    s = s.replace("　", "")
    s = re.sub(r"\s+", "", s)
    return s


def get_vram():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return {
            "allocated": round(torch.cuda.memory_allocated() / (1024 ** 3), 3),
            "reserved": round(torch.cuda.memory_reserved() / (1024 ** 3), 3),
            "max_allocated": round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3),
        }
    return {}


def extract_frames(video_path, fps=2.0):
    try:
        import decord
        from decord import VideoReader, cpu
        decord.bridge.set_bridge("native")
        vr = VideoReader(video_path, ctx=cpu(0))
        vf = vr.get_avg_fps()
        n = len(vr)
        dur = n / vf
        step = vf / fps
        idx = [int(i * step) for i in range(int(n / step))]
        idx = [i for i in idx if i < n]
        frames = [Image.fromarray(vr[i].asnumpy()) for i in idx]
        return frames, dur, n, vf
    except ImportError:
        import cv2
        cap = cv2.VideoCapture(video_path)
        vf = cap.get(cv2.CAP_PROP_FPS)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        dur = n / vf if vf > 0 else 0
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
        return frames, dur, n, vf


def parse_lines(answer: str) -> List[str]:
    bad = ["无文字", "no text", "看不清", "无法确认", "none", "没有"]
    bad_meta = ["安慕希", "董书含", "词", "曲", "编曲", "原唱", "演唱", "歌手", "下一个天亮》"]
    out = []
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


def is_match(ref: str, hyp: str) -> bool:
    r = normalize(ref)
    h = normalize(hyp)
    return (r in h) or (h in r)


def evaluate(lines: List[str]):
    matched = {}
    for ref in REFERENCE_LINES:
        ok = False
        hit = ""
        for hyp in lines:
            if is_match(ref, hyp):
                ok = True
                hit = hyp
                break
        matched[ref] = hit if ok else ""
    n = sum(1 for v in matched.values() if v)
    return matched, n / len(REFERENCE_LINES)


def main():
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        tee = TeeWriter(sys.stdout, f)
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = tee
        sys.stderr = tee
        try:
            print("=" * 80)
            print("EXPERIMENT D-ITER: CUMULATIVE LYRICS VS REFERENCE")
            print("=" * 80)
            print(f"time={datetime.now().isoformat(timespec='seconds')}")
            print(f"video={VIDEO_PATH}")
            print(f"max_cache={MAX_CACHE_TOKENS}, fps={SAMPLE_FPS}, chunk={CHUNK_FRAMES}, ask_interval={ASK_INTERVAL}")
            print()

            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained(MODEL_PATH)
            model = StreamQwenModel.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16).to("cuda").eval()
            print("vram after load:", get_vram())

            frames, dur, total, vf = extract_frames(VIDEO_PATH, SAMPLE_FPS)
            print(f"video duration={dur:.1f}s, raw_frames={total}, sampled_frames={len(frames)}")
            exp_chunks = (len(frames) + CHUNK_FRAMES - 1) // CHUNK_FRAMES
            print(f"expected_chunks={exp_chunks}")

            engine = VideoStreamingInference(
                model,
                processor,
                "cuda",
                eviction_config=EvictionConfig(max_cache_tokens=MAX_CACHE_TOKENS),
            )

            seen: Dict[str, Dict[str, str]] = {}
            ask_n = 0
            t0 = time.time()

            for i in range(0, len(frames), CHUNK_FRAMES):
                chunk = frames[i:i+CHUNK_FRAMES]
                if not chunk:
                    continue
                if len(chunk) % 2:
                    chunk.append(chunk[-1])
                engine.append_video_chunk(chunk, fps=SAMPLE_FPS)
                chunk_id = i // CHUNK_FRAMES + 1

                if (chunk_id % ASK_INTERVAL != 0) and (i + CHUNK_FRAMES < len(frames)):
                    continue

                ask_n += 1
                tsec = (i + CHUNK_FRAMES) / SAMPLE_FPS
                pre = engine.cache_manager.get_seq_length()

                ans_cum, m1 = engine.ask(QUESTION_CUMULATIVE, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, temperature=0.1)
                ans_cur, m2 = engine.ask(QUESTION_CURRENT, max_new_tokens=128, do_sample=False, temperature=0.1)
                post = engine.cache_manager.get_seq_length()
                restore_ok = (pre == post)

                lines = parse_lines(ans_cum) + parse_lines(ans_cur)
                new_cnt = 0
                for ln in lines:
                    k = normalize(ln)
                    if not k:
                        continue
                    if k not in seen:
                        seen[k] = {"line": ln, "time": f"~{tsec:.0f}s", "chunk": str(chunk_id)}
                        new_cnt += 1

                print(f"ask#{ask_n} chunk={chunk_id} t~{tsec:.0f}s cache={pre} restore={'✅' if restore_ok else '❌'} new={new_cnt}")
                print("  cumulative:", ans_cum.strip()[:120])
                print("  current   :", ans_cur.strip()[:120])

            elapsed = time.time() - t0
            final_info = engine.get_cache_info()

            ordered = sorted(seen.values(), key=lambda x: int(x["chunk"]))
            lines = [x["line"] for x in ordered]
            matched, ratio = evaluate(lines)

            print("\n" + "=" * 80)
            print("DEDUP LYRICS")
            print("=" * 80)
            for x in ordered:
                print(f"[{x['time']}] {x['line']}")
            print(f"total_unique={len(ordered)}")

            print("\n" + "=" * 80)
            print("MATCH AGAINST REFERENCE")
            print("=" * 80)
            hit = 0
            for ref in REFERENCE_LINES:
                m = matched[ref]
                ok = bool(m)
                hit += int(ok)
                print(f"{'✅' if ok else '❌'} REF: {ref}")
                if ok:
                    print(f"    HIT: {m}")
            print(f"\nmatch={hit}/{len(REFERENCE_LINES)} = {ratio*100:.1f}%")

            print("\n" + "=" * 80)
            print("RUNTIME / CACHE")
            print("=" * 80)
            print(f"elapsed={elapsed:.1f}s, asks={ask_n}")
            print(f"final_cache_len={final_info['cache_seq_length']}")
            if 'eviction_stats' in final_info:
                es = final_info['eviction_stats']
                print(f"evictions={es.get('total_evictions',0)}, evicted_tokens={es.get('total_tokens_evicted',0)}")
            print("vram final:", get_vram())

            target = 0.80
            print("\nRESULT:")
            if ratio >= target:
                print(f"🎉 PASS: 匹配率 {ratio*100:.1f}% >= {target*100:.0f}%")
            else:
                print(f"⚠️ NEED ITERATION: 匹配率 {ratio*100:.1f}% < {target*100:.0f}%")

        finally:
            sys.stdout = old_out
            sys.stderr = old_err

    print(f"report saved: {REPORT_PATH}")


if __name__ == '__main__':
    main()
