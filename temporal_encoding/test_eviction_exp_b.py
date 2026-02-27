"""
实验 B: OOM-Free 长视频处理

验证:
  1) 启用 Level 1 KV Cache 淘汰 (Sink + Window, 全自动参数)
  2) 以 4 帧/chunk、fps=2 的方式逐段编码整个 1.mp4
  3) 显存保持稳定，不 OOM
  4) cache_len 在触发淘汰后保持 ≤ max_cache_tokens
  5) 最后提一个问题验证 cache 可用性
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
    "/root/autodl-tmp/diploma_project/temporal_encoding/test_eviction_exp_b_report.txt",
)

# ── 淘汰参数 (全自动) ──
# ⬇️ 这是需要实验调优的核心超参数。
# 100K=保守(3.4GB cache), 130K=中等(4.5GB), 150K=激进(5.2GB, 接近极限)
# 峰值 cache = max + 1 chunk (~5.4K), 不可超 ~155K (4090 24GB)
# 过小→window不足→近期信息丢失→回答质量下降; 过大→OOM
# 建议从 130K 开始, 若稳定则尝试 150K
MAX_CACHE_TOKENS = 150_000  # 中等配置, ~4.5 GB cache, total ~11.6 GB

# ── 编码参数 ──
CHUNK_FRAMES = 4      # 每次追加 4 帧 (与 test_step10 一致)
SAMPLE_FPS = 2.0      # 采样帧率
PRINT_INTERVAL = 10   # 每 10 个 chunk 打印一次


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
    """从视频文件中按指定 fps 采样帧。返回 PIL Image 列表。"""
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
        print(f"  Duration: {duration:.1f}s, FPS: {video_fps:.1f}, Total frames: {total_frames}")
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
        print(f"  Duration: {duration:.1f}s, FPS: {video_fps:.1f}, Total frames: {total_frames}")
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
            print("EXPERIMENT B: OOM-Free Long Video Processing with KV Cache Eviction")
            print("=" * 70)
            print(f"Report time: {datetime.now().isoformat(timespec='seconds')}")
            print(f"Eviction config: max_cache_tokens={MAX_CACHE_TOKENS}, "
                  f"sink=auto, window=auto")
            print(f"Expected: test_step10 shows OOM at 40 chunks (160 frames) without eviction.")
            print(f"With eviction, should process ALL chunks without OOM.")
            print()

            # ── 0) 检查文件 ──
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
            vram_model = get_vram_gb()
            print(f"  VRAM after model load: {vram_model}")
            print()

            # ── 2) 提取视频帧 ──
            print("[2] Extracting frames from video...")
            all_frames, duration = extract_frames_from_video(VIDEO_PATH, fps=SAMPLE_FPS)
            total_frame_count = len(all_frames)
            expected_chunks = (total_frame_count + CHUNK_FRAMES - 1) // CHUNK_FRAMES
            print(f"  Total frames extracted: {total_frame_count}")
            print(f"  Expected chunks (4 frames/chunk): {expected_chunks}")
            print(f"  ⚠️ Without eviction, OOM at ~40 chunks ({40*CHUNK_FRAMES} frames).")
            print(f"  With eviction (max={MAX_CACHE_TOKENS}), should handle all {expected_chunks} chunks.")
            print()

            # ── 3) 创建引擎 (启用 Level 1 淘汰, 全自动参数) ──
            print("[3] Creating streaming inference engine with eviction...")
            eviction_config = EvictionConfig(
                max_cache_tokens=MAX_CACHE_TOKENS,
                # sink_size=0  → 自动检测首 chunk
                # window_size=0 → 自动计算
            )
            engine = VideoStreamingInference(
                model, processor, device, eviction_config=eviction_config
            )
            print()

            # ── 4) 逐 chunk 编码 ──
            print("[4] Encoding video chunks...")
            t_start = time.time()
            vram_history = []
            cache_history = []
            chunk_count = 0
            first_eviction_chunk = None

            for i in range(0, total_frame_count, CHUNK_FRAMES):
                chunk = all_frames[i : i + CHUNK_FRAMES]
                if len(chunk) == 0:
                    continue
                # 补齐偶数帧 (temporal_patch_size=2)
                if len(chunk) % 2 != 0:
                    chunk.append(chunk[-1])

                result = engine.append_video_chunk(chunk, fps=SAMPLE_FPS)
                chunk_count += 1

                info = engine.get_cache_info()
                cache_len = info["cache_seq_length"]

                # 记录首次淘汰
                if "eviction_stats" in info:
                    es = info["eviction_stats"]
                    if es.get("total_evictions", 0) > 0 and first_eviction_chunk is None:
                        first_eviction_chunk = chunk_count

                if chunk_count % PRINT_INTERVAL == 0 or chunk_count == 1:
                    vram = get_vram_gb()
                    vram_history.append({
                        "chunk": chunk_count,
                        "cache_len": cache_len,
                        "vram_alloc": vram.get("allocated", 0),
                        "vram_reserved": vram.get("reserved", 0),
                    })
                    cache_history.append(cache_len)

                    eviction_str = ""
                    if "eviction_stats" in info:
                        es = info["eviction_stats"]
                        eviction_str = (
                            f", evictions={es.get('total_evictions', 0)}, "
                            f"evicted={es.get('total_tokens_evicted', 0)}"
                        )

                    print(
                        f"  Chunk {chunk_count:>4d}/{expected_chunks}: "
                        f"cache_len={cache_len:>6d}, "
                        f"mem={info.get('cache_memory_gb', 0):.3f} GB, "
                        f"VRAM={vram.get('allocated', 0):.2f}/{vram.get('reserved', 0):.2f} GB"
                        f"{eviction_str}"
                    )

            t_encode = time.time() - t_start

            # 汇总
            print(f"\n  ✅ Encoding completed: {chunk_count} chunks, "
                  f"{total_frame_count} frames in {t_encode:.1f}s")
            final_vram = get_vram_gb()
            print(f"  Final VRAM: {final_vram}")
            if first_eviction_chunk:
                print(f"  First eviction at chunk: {first_eviction_chunk}")

            # 获取 evictor 状态
            evictor = engine.cache_manager.evictor
            if evictor:
                print(f"  Effective sink_size: {evictor.effective_sink_size}")
                print(f"  Effective window_size: {evictor.effective_window_size}")
                print(f"  Avg chunk tokens: {evictor._avg_chunk_tokens:.0f}")
            print()

            # ── 5) 验证 ask 仍可用 ──
            print("[5] Verification: asking a question...")
            final_info = engine.get_cache_info()
            print(f"  Pre-ask cache: len={final_info['cache_seq_length']}, "
                  f"mem={final_info.get('cache_memory_gb', 0):.3f} GB")

            answer, metrics = engine.ask(
                "Briefly describe what you saw in the entire video.",
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
            )
            print(f"  Answer: {answer}")
            print(f"  TTFT: {metrics['ttft']:.3f}s")

            # 验证 ask 后 cache 恢复 (snapshot/restore)
            post_ask_info = engine.get_cache_info()
            print(f"  Post-ask cache: len={post_ask_info['cache_seq_length']}")
            assert post_ask_info['cache_seq_length'] == final_info['cache_seq_length'], \
                "ask() 后 cache 长度应恢复 (snapshot/restore)"
            print(f"  ✅ Cache restored after ask()")
            print()

            # ── 6) 总结 ──
            print("=" * 70)
            print("SUMMARY")
            print("=" * 70)
            print(f"  Video: {VIDEO_PATH} ({duration:.0f}s)")
            print(f"  Total frames: {total_frame_count}")
            print(f"  Total chunks: {chunk_count}")
            print(f"  Encoding time: {t_encode:.1f}s "
                  f"({total_frame_count / t_encode:.2f} frames/sec)")
            print(f"  Max cache tokens: {MAX_CACHE_TOKENS}")
            print(f"  Final cache_len: {final_info['cache_seq_length']}")
            print(f"  Final VRAM: allocated={final_vram.get('allocated', 0):.2f} GB, "
                  f"reserved={final_vram.get('reserved', 0):.2f} GB, "
                  f"max_allocated={final_vram.get('max_allocated', 0):.2f} GB")

            if "eviction_stats" in final_info:
                es = final_info["eviction_stats"]
                print(f"  Total evictions: {es.get('total_evictions', 0)}")
                print(f"  Total tokens evicted: {es.get('total_tokens_evicted', 0)}")

            # 通过判定
            print()
            print("─" * 70)
            print("PASS/FAIL CRITERIA:")
            all_pass = True

            # 1) 没 OOM (走到这里说明没 OOM)
            print(f"  ✅ [P1] No OOM — processed all {chunk_count} chunks "
                  f"(test_step10 OOM at 40 chunks without eviction)")

            # 2) cache_len ≤ max_cache_tokens
            if final_info['cache_seq_length'] <= MAX_CACHE_TOKENS:
                print(f"  ✅ [P2] cache_len ({final_info['cache_seq_length']}) "
                      f"≤ max ({MAX_CACHE_TOKENS})")
            else:
                print(f"  ❌ [P2] cache_len ({final_info['cache_seq_length']}) "
                      f"> max ({MAX_CACHE_TOKENS})")
                all_pass = False

            # 3) 有淘汰发生
            if "eviction_stats" in final_info:
                es = final_info["eviction_stats"]
                if es.get("total_evictions", 0) > 0:
                    print(f"  ✅ [P3] Eviction occurred "
                          f"({es['total_evictions']} times, "
                          f"{es['total_tokens_evicted']} tokens)")
                else:
                    print(f"  ❌ [P3] No eviction occurred — config may not have been applied")
                    all_pass = False

            # 4) VRAM 未超 23 GB
            max_alloc = final_vram.get("max_allocated", 0)
            if max_alloc < 23.0:
                print(f"  ✅ [P4] Max VRAM allocated ({max_alloc:.2f} GB) < 23 GB")
            else:
                print(f"  ⚠️ [P4] Max VRAM allocated ({max_alloc:.2f} GB) ≥ 23 GB")
                all_pass = False

            # 5) ask 正常
            if answer and len(answer) > 5:
                print(f"  ✅ [P5] ask() returned valid answer ({len(answer)} chars)")
            else:
                print(f"  ❌ [P5] ask() returned empty/short answer")
                all_pass = False

            print()
            if all_pass:
                print("🎉 EXPERIMENT B: ALL PASSED")
            else:
                print("⚠️ EXPERIMENT B: SOME CHECKS FAILED — see above")

        except torch.cuda.OutOfMemoryError:
            print(f"\n❌ EXPERIMENT B FAILED: CUDA OOM!")
            print(f"  This means eviction did not prevent OOM.")
            print(f"  Possible causes:")
            print(f"    1) Eviction not triggered — check EvictionConfig")
            print(f"    2) max_cache_tokens too large — try 50,000")
            print(f"    3) torch reserved memory fragmentation")
            import traceback
            traceback.print_exc()

        except Exception as e:
            print(f"\n❌ EXPERIMENT B FAILED: {e}")
            import traceback
            traceback.print_exc()

        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr

    print(f"\nReport saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()