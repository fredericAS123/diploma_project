"""
Step 6: 流式推理 vs 原生离线推理对比测试（核心需求）

测试目标：
  1) 使用真实视频 1.mp4 (~3s, 30fps)
  2) 流式模式：按 4 帧 chunk 流式编码至 2s，暂停后回答问题
  3) 原生模式：一次性加载完整视频并回答同一问题
  4) 对比：VRAM 使用、响应时间（TTFT、总延迟）、答案质量

需要 GPU + 模型权重 + 视频文件。
"""
import os
import sys
import cv2
import time
import torch
from datetime import datetime
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import StreamQwenModel, VideoStreamingInference

MODEL_PATH = os.environ.get("QWEN_MODEL_PATH", "/root/autodl-tmp/Qwen/Qwen2___5-VL-3B-Instruct")
VIDEO_PATH = os.environ.get("VIDEO_PATH", "/root/autodl-tmp/diploma_project/temporal_encoding/1.mp4")
REPORT_PATH = os.environ.get(
    "STEP6_REPORT_PATH",
    "/root/autodl-tmp/diploma_project/temporal_encoding/test_step6_stream_vs_native_report.txt",
)
STREAM_DURATION = 2.0  # 流式编码至 2 秒
CHUNK_SIZE = 4         # 每个 chunk 包含 4 帧
TEST_QUESTION = "Describe what is happening in this video."
BASE_FRAME_STRIDE = int(os.environ.get("FRAME_STRIDE", "1"))


class NativeOOMError(RuntimeError):
    """Native offline mode OOM error."""
    pass


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


def _get_vram_gb():
    """获取当前 CUDA VRAM 使用量（GB）。"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        return {"allocated": round(allocated, 2), "reserved": round(reserved, 2)}
    return {"allocated": 0.0, "reserved": 0.0}


def _load_video_frames(video_path: str, max_duration: float = None, frame_stride: int = 1):
    """从视频加载帧，返回 (frames, fps, total_duration)。"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    frames = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % max(frame_stride, 1) == 0:
            # 转换为 PIL Image (RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            frames.append(pil_img)
        
        frame_idx += 1
        if max_duration is not None and fps > 0:
            if frame_idx / fps >= max_duration:
                break
    
    cap.release()
    return frames, fps, duration


def test_streaming_mode(frame_stride: int = 1):
    """流式模式：逐 chunk 编码至指定时长，然后回答问题。"""
    print("\n" + "=" * 70)
    print("📹 STREAMING MODE TEST")
    print("=" * 70)
    
    if not os.path.exists(VIDEO_PATH):
        print(f"⚠️  Video not found: {VIDEO_PATH}. Skip streaming test.")
        return None
    
    if not os.path.exists(MODEL_PATH):
        print(f"⚠️  Model not found: {MODEL_PATH}. Skip streaming test.")
        return None
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
    # 1) 加载视频帧（仅前 STREAM_DURATION 秒）
    print(f"\n[1] Loading video frames (first {STREAM_DURATION}s, stride={frame_stride})...")
    frames, video_fps, total_duration = _load_video_frames(
        VIDEO_PATH,
        max_duration=STREAM_DURATION,
        frame_stride=frame_stride,
    )
    print(f"    ✅ Loaded {len(frames)} frames (fps={video_fps:.2f}, total={total_duration:.2f}s)")
    
    # 2) 初始化流式引擎
    print("\n[2] Initializing streaming engine...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model = StreamQwenModel.from_pretrained(MODEL_PATH, torch_dtype=dtype).to(device)
    model.eval()
    engine = VideoStreamingInference(model, processor, device)
    
    vram_init = _get_vram_gb()
    print(f"    ✅ VRAM after model load: {vram_init}")
    
    # 3) 流式编码（按 CHUNK_SIZE 分 chunk）
    print(f"\n[3] Streaming encoding ({CHUNK_SIZE}-frame chunks)...")
    encode_start = time.time()
    
    num_chunks = (len(frames) + CHUNK_SIZE - 1) // CHUNK_SIZE
    for i in range(num_chunks):
        start_idx = i * CHUNK_SIZE
        end_idx = min(start_idx + CHUNK_SIZE, len(frames))
        chunk_frames = frames[start_idx:end_idx]
        
        chunk_fps = video_fps / frame_stride  # 使用原始视频帧率/采样步长作为 chunk fps
        status = engine.append_video_chunk(
            chunk_frames,
            fps=chunk_fps,
            text_content=f"Processing video chunk {i+1}/{num_chunks}."
        )
        print(f"    Chunk {i+1}/{num_chunks}: {status}")
    
    encode_end = time.time()
    encode_time = encode_end - encode_start
    
    cache_info = engine.get_cache_info()
    vram_after_encode = _get_vram_gb()
    print(f"\n    ✅ Encoding completed in {encode_time:.3f}s")
    print(f"    Cache info: {cache_info}")
    print(f"    VRAM after encoding: {vram_after_encode}")
    
    # 4) 回答问题
    print(f"\n[4] Asking question: '{TEST_QUESTION}'")
    answer, metrics = engine.ask(TEST_QUESTION, max_new_tokens=128, update_state=False)
    
    vram_after_qa = _get_vram_gb()
    print(f"\n    ✅ Answer: {answer}")
    print(f"    TTFT: {metrics['ttft']:.3f}s")
    print(f"    Total QA latency: {metrics['total_latency']:.3f}s")
    print(f"    VRAM after QA: {vram_after_qa}")
    
    return {
        "mode": "streaming",
        "frames_encoded": len(frames),
        "encoding_time": round(encode_time, 3),
        "frame_stride": frame_stride,
        "cache_seq_length": cache_info["cache_seq_length"],
        "cache_memory_gb": cache_info["cache_memory_gb"],
        "vram_init": vram_init,
        "vram_after_encode": vram_after_encode,
        "vram_after_qa": vram_after_qa,
        "ttft": round(metrics["ttft"], 3),
        "total_qa_latency": round(metrics["total_latency"], 3),
        "answer": answer,
    }


def test_native_offline_mode(frame_stride: int = 1):
    """原生离线模式：一次性加载完整视频并回答问题。"""
    print("\n" + "=" * 70)
    print("🎬 NATIVE OFFLINE MODE TEST")
    print("=" * 70)
    
    if not os.path.exists(VIDEO_PATH):
        print(f"⚠️  Video not found: {VIDEO_PATH}. Skip native test.")
        return None
    
    if not os.path.exists(MODEL_PATH):
        print(f"⚠️  Model not found: {MODEL_PATH}. Skip native test.")
        return None
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
    # 1) 加载视频帧（相同的前 STREAM_DURATION 秒，保证公平对比）
    print(f"\n[1] Loading video frames (first {STREAM_DURATION}s, stride={frame_stride})...")
    frames, video_fps, total_duration = _load_video_frames(
        VIDEO_PATH,
        max_duration=STREAM_DURATION,
        frame_stride=frame_stride,
    )
    print(f"    ✅ Loaded {len(frames)} frames (fps={video_fps:.2f}, total={total_duration:.2f}s)")
    
    # 2) 初始化原生模型
    print("\n[2] Initializing native model...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_PATH, torch_dtype=dtype).to(device)
    model.eval()
    
    vram_init = _get_vram_gb()
    print(f"    ✅ VRAM after model load: {vram_init}")
    
    # 3) 一次性编码完整视频 + 问题
    print(f"\n[3] Encoding full video + question...")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": frames, "fps": video_fps},
                {"type": "text", "text": TEST_QUESTION},
            ]
        }
    ]
    
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text_prompt],
        videos=[frames],
        padding=True,
        return_tensors="pt",
        videos_kwargs={"fps": video_fps}
    ).to(device)
    
    encode_start = time.time()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Prefill: 计算首个 token
    with torch.inference_mode():
        outputs = model(**inputs, use_cache=True)
        past_key_values = outputs.past_key_values
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    encode_end = time.time()
    ttft = encode_end - encode_start
    
    vram_after_prefill = _get_vram_gb()
    print(f"    ✅ Prefill completed, TTFT: {ttft:.3f}s")
    print(f"    VRAM after prefill: {vram_after_prefill}")
    
    # 4) Decode 生成答案
    print(f"\n[4] Generating answer...")
    decode_start = time.time()
    
    try:
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
            )
    except torch.OutOfMemoryError as exc:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise NativeOOMError("Native offline generate OOM") from exc
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    decode_end = time.time()
    total_latency = decode_end - encode_start
    
    # 提取生成的 token（移除输入部分）
    input_len = inputs.input_ids.shape[1]
    generated_tokens = generated_ids[0, input_len:]
    answer = processor.decode(generated_tokens, skip_special_tokens=True)
    
    vram_after_decode = _get_vram_gb()
    print(f"\n    ✅ Answer: {answer}")
    print(f"    TTFT: {ttft:.3f}s")
    print(f"    Total latency: {total_latency:.3f}s")
    print(f"    VRAM after decode: {vram_after_decode}")
    
    # 估算 cache 大小
    cache_seq_len = 0
    if past_key_values is not None:
        if hasattr(past_key_values, "get_seq_length"):
            cache_seq_len = past_key_values.get_seq_length()
        elif isinstance(past_key_values, (tuple, list)) and len(past_key_values) > 0:
            cache_seq_len = past_key_values[0][0].shape[2]
    
    return {
        "mode": "native_offline",
        "frames_encoded": len(frames),
        "encoding_time": round(ttft, 3),
        "frame_stride": frame_stride,
        "cache_seq_length": cache_seq_len,
        "vram_init": vram_init,
        "vram_after_prefill": vram_after_prefill,
        "vram_after_decode": vram_after_decode,
        "ttft": round(ttft, 3),
        "total_latency": round(total_latency, 3),
        "answer": answer,
    }


def print_comparison(streaming_result, native_result, attempt_logs):
    """打印详细对比报告。"""
    print("\n" + "=" * 70)
    print("📊 COMPARISON REPORT")
    print("=" * 70)
    
    if streaming_result is None or native_result is None:
        print("⚠️  One or both tests failed. Cannot compare.")
        return
    
    print("\n[Encoding Performance]")
    print(f"  Streaming encoding time: {streaming_result['encoding_time']}s")
    print(f"  Native prefill time:     {native_result['encoding_time']}s")

    print("\n[Frame Sampling]")
    print(f"  Streaming frame stride:  {streaming_result['frame_stride']}")
    print(f"  Native frame stride:     {native_result['frame_stride']}")
    
    print("\n[QA Performance]")
    print(f"  Streaming TTFT:          {streaming_result['ttft']}s")
    print(f"  Native TTFT:             {native_result['ttft']}s")
    print(f"  Streaming total QA:      {streaming_result['total_qa_latency']}s")
    print(f"  Native total latency:    {native_result['total_latency']}s")
    
    print("\n[VRAM Usage (Allocated GB)]")
    print(f"  Streaming after encode:  {streaming_result['vram_after_encode']['allocated']} GB")
    print(f"  Native after prefill:    {native_result['vram_after_prefill']['allocated']} GB")
    print(f"  Streaming after QA:      {streaming_result['vram_after_qa']['allocated']} GB")
    print(f"  Native after decode:     {native_result['vram_after_decode']['allocated']} GB")
    
    print("\n[Cache Info]")
    print(f"  Streaming cache length:  {streaming_result['cache_seq_length']}")
    print(f"  Streaming cache memory:  {streaming_result['cache_memory_gb']} GB")
    print(f"  Native cache length:     {native_result['cache_seq_length']}")
    
    print("\n[Answers]")
    print(f"  Streaming: {streaming_result['answer'][:100]}...")
    print(f"  Native:    {native_result['answer'][:100]}...")

    print("\n[Conversation Log]")
    print(f"  Question: {TEST_QUESTION}")
    print(f"  Streaming Answer: {streaming_result['answer']}")
    print(f"  Native Answer: {native_result['answer']}")

    print("\n[Retry/Iteration Log]")
    for entry in attempt_logs:
        print(f"  - {entry}")

    print("\n[Analysis]")
    ttft_speedup = native_result["ttft"] / max(streaming_result["ttft"], 1e-6)
    total_speedup = native_result["total_latency"] / max(streaming_result["total_qa_latency"], 1e-6)
    vram_prefill = native_result["vram_after_prefill"]["allocated"]
    vram_stream = streaming_result["vram_after_encode"]["allocated"]
    vram_delta = vram_prefill - vram_stream
    print(f"  TTFT speedup (Native/Streaming): {ttft_speedup:.2f}x")
    print(f"  Total latency speedup (Native/Streaming): {total_speedup:.2f}x")
    print(f"  VRAM delta (Native prefill - Streaming encode): {vram_delta:.2f} GB")
    if streaming_result["frame_stride"] != 1:
        print("  Note: Frame stride > 1 was used to avoid native OOM; results are fair within the same stride.")
    
    print("\n" + "=" * 70)
    print("✅ Step 6 COMPLETED: Streaming vs Native Offline Comparison")
    print("=" * 70)


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
            print("TEST Step 6: Streaming vs Native Offline Inference Comparison")
            print("=" * 70)
            print(f"Report time: {datetime.now().isoformat(timespec='seconds')}")
            print(f"Video: {VIDEO_PATH}")
            print(f"Model: {MODEL_PATH}")
            print(f"Stream duration: {STREAM_DURATION}s")
            print(f"Chunk size: {CHUNK_SIZE} frames")
            print(f"Question: '{TEST_QUESTION}'")
    
    # 尝试不同帧采样步长，避免 Native OOM
            stride_candidates = [max(BASE_FRAME_STRIDE, 1)]
            stride_candidates += [stride_candidates[0] * 2, stride_candidates[0] * 4]
            stride_candidates = [s for i, s in enumerate(stride_candidates) if s not in stride_candidates[:i]]

            streaming_result = None
            native_result = None
            attempt_logs = []

            for stride in stride_candidates:
                print("\n" + "-" * 70)
                print(f"Attempt with frame stride = {stride}")
                print("-" * 70)

                # 测试流式模式
                streaming_result = test_streaming_mode(frame_stride=stride)
                attempt_logs.append(f"Stride {stride}: streaming OK")

                # 清理 GPU 内存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # 测试原生模式
                try:
                    native_result = test_native_offline_mode(frame_stride=stride)
                    attempt_logs.append(f"Stride {stride}: native OK")
                    break
                except NativeOOMError:
                    attempt_logs.append(f"Stride {stride}: native OOM")
                    print("⚠️  Native offline mode OOM. Retrying with higher frame stride...")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue

            # 打印对比报告
            print_comparison(streaming_result, native_result, attempt_logs)
            print(f"\nReport saved to: {REPORT_PATH}")
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


if __name__ == "__main__":
    main()
