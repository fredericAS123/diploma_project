import gradio as gr
import threading
import time
import torch
import numpy as np
from PIL import Image
from Qwen_inference import QwenInferenceWrapper

try:
    from decord import VideoReader, cpu
except ImportError:
    VideoReader = None
    cpu = None

class HistorySynchronizer:
    def __init__(self):
        self.chat_history = []

    def get_chat_history(self):
        return self.chat_history

    def update(self, role, content):
        self.chat_history.append(gr.ChatMessage(role=role, content=str(content)))

    def add_message(self, role, content):
        self.chat_history.append(gr.ChatMessage(role=role, content=str(content)))
        return len(self.chat_history) - 1

    def update_message(self, idx, role, content):
        if 0 <= idx < len(self.chat_history):
            self.chat_history[idx] = gr.ChatMessage(role=role, content=str(content))

    def reset(self):
        self.chat_history = []

class VideoChatWebUI:
    def __init__(self, inference_engine: QwenInferenceWrapper):
        """
        初始化 UI 逻辑，传入推理引擎实例。
        已适配新的流式引擎 API（无 manual_time，支持 chunk 编码）。
        """
        self.inference_engine = inference_engine
        self.history_synchronizer = HistorySynchronizer()
        
        # [控制标志]
        self.pause_event = threading.Event()
        self.pause_event.set() 
        self.stop_signal = False 
        self.is_streaming = False            
        self.current_streaming_time = 0.0
        
        # [数据缓存]
        self.cached_video_path = None 
        self.cached_video_data = None 

        # [原生推理模式]
        self.mode = "streaming"  # "streaming" or "native"
        self.native_frame_buffer = []
        self.native_fps = 2.0
        self.last_streaming_metrics = {}
        self.last_native_metrics = {}
        self.last_single_frame_metrics = {}
        self.last_frame_pil = None
        self._last_eviction_notified = 0
        self.stream_frame_interval = 1.0
        self.stream_target_fps = 1.0
        self.stream_chunk_size = 4
        self._chunk_token_history = []
        self._prev_cache_len = None
        self._prev_total_evicted_tokens = 0
        self._mem_history = []
        self._mem_t0 = time.perf_counter()

    def _format_cache_status(self, cache_info):
        """格式化 KV Cache 状态信息，包含淘汰统计。"""
        parts = [
            f"Seq: {cache_info['cache_seq_length']}",
            f"Mem: {cache_info['cache_memory_gb']:.2f}GB",
            f"Frames: {cache_info['total_frames']}",
        ]
        evict = cache_info.get('eviction_stats', {})
        if evict:
            n_evictions = evict.get('total_evictions', 0)
            if n_evictions > 0:
                total_evicted = evict.get('total_tokens_evicted', 0)
                evicted_str = f"{total_evicted // 1000}K" if total_evicted >= 1000 else str(total_evicted)
                parts.append(f"✂️ {n_evictions}次(-{evicted_str})")
            sink = evict.get('effective_sink_size')
            window = evict.get('effective_window_size')
            if sink and sink > 0:
                parts.append(f"Sink:{sink}")
            if window and window > 0:
                parts.append(f"Win:{window}")

            time_span_str = self._estimate_cached_time_span(cache_info)
            if time_span_str:
                parts.append(time_span_str)

        return " | ".join(parts)

    def _reset_stream_estimation_state(self):
        self._chunk_token_history = []
        self._prev_cache_len = None
        self._prev_total_evicted_tokens = 0

    def _record_chunk_token_delta(self, cache_info, chunk_start_t: float, chunk_end_t: float):
        """记录每个chunk的近似token增量，用于更准确估算窗口时间覆盖。"""
        cache_len = int(cache_info.get('cache_seq_length', 0))
        evict = cache_info.get('eviction_stats', {})
        total_evicted = int(evict.get('total_tokens_evicted', 0)) if evict else 0

        evicted_delta = max(0, total_evicted - self._prev_total_evicted_tokens)
        if self._prev_cache_len is None:
            new_chunk_tokens = cache_len
        else:
            # cache_after = cache_before + new_chunk - evicted_delta
            new_chunk_tokens = cache_len - self._prev_cache_len + evicted_delta

        new_chunk_tokens = max(1, int(new_chunk_tokens))
        self._chunk_token_history.append(
            {
                "start_t": float(chunk_start_t),
                "end_t": float(chunk_end_t),
                "tokens": new_chunk_tokens,
            }
        )

        self._prev_cache_len = cache_len
        self._prev_total_evicted_tokens = total_evicted

    def _estimate_cached_time_span(self, cache_info) -> str:
        """基于chunk token历史估算当前cache覆盖的视频时间区间。"""
        if not self._chunk_token_history:
            return ""

        evict = cache_info.get('eviction_stats', {})
        if not evict:
            return ""

        cache_seq = float(cache_info.get('cache_seq_length', 0))
        if cache_seq <= 0:
            return ""

        sink_tokens = float(evict.get('effective_sink_size') or 0)
        sink_tokens_in_mem = min(sink_tokens, cache_seq)
        window_tokens_in_mem = max(0.0, cache_seq - sink_tokens_in_mem)

        # Sink 时间范围：严格按“首chunk时间范围”展示，避免token均值换算误差
        first_chunk = self._chunk_token_history[0]
        sink_start_t = max(0.0, float(first_chunk['start_t']))
        sink_end_t = max(sink_start_t, float(first_chunk['end_t']))

        # Window 起点：按“尾部 token 预算”从最近chunk向前回推
        now_t = max(0.0, float(self.current_streaming_time))
        if window_tokens_in_mem <= 0:
            return f"Time≈Sink[{sink_start_t:.1f}-{sink_end_t:.1f}s]"

        acc = 0.0
        win_start_t = now_t
        for rec in reversed(self._chunk_token_history):
            tok = max(1.0, float(rec['tokens']))
            st = float(rec['start_t'])
            ed = float(rec['end_t'])

            if acc + tok >= window_tokens_in_mem:
                need = window_tokens_in_mem - acc
                ratio = max(0.0, min(1.0, need / tok))
                win_start_t = ed - (ed - st) * ratio
                break

            acc += tok
            win_start_t = st

        win_start_t = max(0.0, min(win_start_t, now_t))
        return f"Time≈Sink[{sink_start_t:.1f}-{sink_end_t:.1f}s]+Win[{win_start_t:.1f}-{now_t:.1f}s]"

    def _format_fps_display(self, throughput_fps: float) -> str:
        """格式化编码吞吐显示：frames/s + 相对采样视频的实时倍率。"""
        throughput_fps = max(0.0, float(throughput_fps))
        if self.stream_target_fps > 0:
            rt_factor = throughput_fps / self.stream_target_fps
            return f"{throughput_fps:.2f} (x{rt_factor:.2f} realtime)"
        return f"{throughput_fps:.2f}"

    def _reset_memory_monitor(self):
        self._mem_history = []
        self._mem_t0 = time.perf_counter()

    def _build_memory_wave_html(self):
        if not self._mem_history:
            return "<div style='color:#999;'>等待显存数据...</div>"

        data = self._mem_history[-120:]
        w, h = 520, 160
        pad_l, pad_r, pad_t, pad_b = 38, 8, 8, 20
        pw = w - pad_l - pad_r
        ph = h - pad_t - pad_b

        max_y = max(max(d["allocated"], d["model"], d["kv"], d["runtime"]) for d in data)
        max_y = max(0.1, max_y * 1.1)

        def xy(i, y):
            x = pad_l + (i / max(1, len(data) - 1)) * pw
            yy = pad_t + (1 - y / max_y) * ph
            return x, yy

        def poly(points, color, width=2):
            return f"<polyline fill='none' stroke='{color}' stroke-width='{width}' points='{points}'/>"

        def make_points(key):
            pts = [xy(i, d[key]) for i, d in enumerate(data)]
            return " ".join([f"{x:.1f},{y:.1f}" for x, y in pts])

        grid = []
        for v in [0.25, 0.5, 0.75, 1.0]:
            yy = pad_t + (1 - v) * ph
            val = max_y * v
            grid.append(f"<line x1='{pad_l}' y1='{yy:.1f}' x2='{w-pad_r}' y2='{yy:.1f}' stroke='#2a2a2a' stroke-width='1' />")
            grid.append(f"<text x='2' y='{yy+4:.1f}' fill='#888' font-size='10'>{val:.1f}GB</text>")

        svg = [
            f"<svg width='{w}' height='{h}' viewBox='0 0 {w} {h}' style='background:#111;border:1px solid #333;border-radius:6px;'>",
            "".join(grid),
            poly(make_points("allocated"), "#f59e0b", 2),
            poly(make_points("model"), "#38bdf8", 2),
            poly(make_points("kv"), "#22c55e", 2),
            poly(make_points("runtime"), "#ef4444", 2),
            "</svg>",
            "<div style='font-size:12px;margin-top:4px;'>"
            "<span style='color:#f59e0b'>■ Alloc</span> "
            "<span style='color:#38bdf8'>■ Model</span> "
            "<span style='color:#22c55e'>■ KV</span> "
            "<span style='color:#ef4444'>■ Runtime/Act</span>"
            "</div>",
        ]
        return "".join(svg)

    def _update_memory_monitor(self, cache_info=None):
        if cache_info is None:
            cache_info = self.inference_engine.get_cache_info()
        mem = self.inference_engine.get_memory_breakdown(cache_info)
        t_rel = time.perf_counter() - self._mem_t0
        self._mem_history.append(
            {
                "t": t_rel,
                "model": float(mem.get("model_gb", 0.0)),
                "kv": float(mem.get("kv_cache_gb", 0.0)),
                "runtime": float(mem.get("runtime_gb", 0.0)),
                "allocated": float(mem.get("allocated_gb", 0.0)),
                "reserved": float(mem.get("reserved_gb", 0.0)),
            }
        )
        if len(self._mem_history) > 240:
            self._mem_history = self._mem_history[-240:]

        mem_str = (
            f"Model: {mem.get('model_gb', 0.0):.2f}GB | "
            f"KV: {mem.get('kv_cache_gb', 0.0):.2f}GB | "
            f"Runtime/Act: {mem.get('runtime_gb', 0.0):.2f}GB | "
            f"Alloc: {mem.get('allocated_gb', 0.0):.2f}GB | "
            f"Reserved: {mem.get('reserved_gb', 0.0):.2f}GB"
        )
        return mem_str, self._build_memory_wave_html()

    def generate_answer(self, question):
        """手动提问：支持流式、原生、单帧三模式对比"""

        # ─── 原生推理模式 ───
        if self.mode == "native":
            if not self.native_frame_buffer:
                yield (self.history_synchronizer.get_chat_history() + [
                    gr.ChatMessage(role="assistant", content="请先加载视频帧（点击 Start Native）")
                ], "", "", "", "", "")
                return

            auto_paused = False
            if self.is_streaming and self.pause_event.is_set():
                self.pause_event.clear()
                auto_paused = True
                time.sleep(0.1)

            self.history_synchronizer.update("user", question)
            n_frames = len(self.native_frame_buffer)
            self._reset_memory_monitor()
            mem_str, mem_wave = self._update_memory_monitor()
            yield (self.history_synchronizer.get_chat_history() + [
                gr.ChatMessage(role="assistant",
                    content=f"🔄 正在进行双模式对比推理 ({n_frames} 帧 vs 单帧)...")
            ], "", "", "", mem_str, mem_wave)

            # 1) 原生推理（全部帧）
            try:
                native_response, native_metrics = self.inference_engine.native_video_inference(
                    frames=self.native_frame_buffer,
                    question=question,
                    fps=self.native_fps,
                    max_new_tokens=256,
                    min_new_tokens=8,
                    do_sample=False,
                )
                self.last_native_metrics = native_metrics
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                native_response = (
                    "⚠️ OOM: 原生模式显存不足！这正说明流式推理的优势——"
                    "原生模式需要一次性处理所有帧，显存需求远高于流式模式。"
                )
                native_metrics = {"ttft": 0.0, "vram_peak_gb": 0.0, "num_frames": n_frames, "input_tokens": 0}
                self.last_native_metrics = native_metrics

            # 2) 单帧推理（仅最后一帧）
            last_frame = self.native_frame_buffer[-1]
            try:
                sf_response, sf_metrics = self.inference_engine.single_frame_inference(
                    image=last_frame,
                    question=question,
                    max_new_tokens=256,
                    min_new_tokens=8,
                    do_sample=False,
                )
                self.last_single_frame_metrics = sf_metrics
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                sf_response = "⚠️ 单帧推理 OOM"
                sf_metrics = {"ttft": 0.0, "vram_peak_gb": 0.0, "input_tokens": 0}
                self.last_single_frame_metrics = sf_metrics

            self.history_synchronizer.update(
                "assistant",
                f"📦 [原生推理 - 全部 {n_frames} 帧]\n{native_response}"
            )
            self.history_synchronizer.update(
                "assistant",
                f"🖼️ [单帧推理 - 仅最后一帧]\n{sf_response}"
            )

            native_info = (
                f"[Native] Frames: {native_metrics.get('num_frames', 0)} | "
                f"Tokens: {native_metrics.get('input_tokens', 0)} | "
                f"VRAM Peak: {native_metrics.get('vram_peak_gb', 0):.2f}GB"
            )
            comparison = self._build_comparison_text()
            mem_str, mem_wave = self._update_memory_monitor()
            if auto_paused:
                self.pause_event.set()
            yield (self.history_synchronizer.get_chat_history(),
                   f"{native_metrics.get('ttft', 0):.3f}", native_info, comparison, mem_str, mem_wave)
            return

        # ─── 流式推理模式 ───
        if not self.is_streaming and self.current_streaming_time <= 0:
            yield (self.history_synchronizer.get_chat_history() + [
                gr.ChatMessage(role="assistant", content="请先播放视频")
            ], "", "", "", "", "")
            return

        # 自动暂停机制
        auto_paused = False
        if self.is_streaming and self.pause_event.is_set():
            print("⚠️ Detected streaming active. Auto-pausing for manual question...")
            self.pause_event.clear()
            auto_paused = True
            time.sleep(0.1)

        self.history_synchronizer.update("user", question)
        self._reset_memory_monitor()
        mem_str, mem_wave = self._update_memory_monitor()
        
        status_msg = "🔄 正在进行双模式对比推理 (流式 + 单帧)..."
        if auto_paused:
            status_msg += " (视频已自动暂停)"
            
        yield self.history_synchronizer.get_chat_history() + [gr.ChatMessage(role="assistant", content=status_msg)], "", "", "", mem_str, mem_wave

        try:
            # 重置 VRAM 峰值统计
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            assistant_prefix = "📡 [流式推理 - KV Cache 复用]\n"
            assistant_idx = self.history_synchronizer.add_message("assistant", assistant_prefix)
            final_text = ""
            ttft_stream = 0.0
            metrics = {"ttft": 0.0, "total_latency": 0.0}

            for event in self.inference_engine.ask_question_stream(
                question,
                max_new_tokens=256,
                min_new_tokens=8,
                do_sample=False,
                temperature=0.7,
                top_p=0.9,
            ):
                if event.get("type") == "token":
                    final_text = event.get("text", final_text)
                    if event.get("ttft") is not None:
                        ttft_stream = float(event["ttft"])
                    self.history_synchronizer.update_message(
                        assistant_idx,
                        "assistant",
                        assistant_prefix + final_text,
                    )
                    cache_info = self.inference_engine.get_cache_info()
                    cache_str = self._format_cache_status(cache_info)
                    mem_str, mem_wave = self._update_memory_monitor(cache_info)
                    yield self.history_synchronizer.get_chat_history(), f"{ttft_stream:.3f}", cache_str, "", mem_str, mem_wave
                elif event.get("type") == "final":
                    final_text = event.get("text", final_text)
                    metrics = event.get("metrics", metrics)

            if torch.cuda.is_available():
                metrics["vram_peak_gb"] = round(
                    torch.cuda.max_memory_allocated() / (1024 ** 3), 3
                )
            self.last_streaming_metrics = metrics
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            response = "Error: OOM (显存不足)。请尝试减少视频帧数。"
            metrics = {"ttft": 0.0}
            self.history_synchronizer.update("assistant", f"📡 [流式推理 - KV Cache 复用]\n{response}")
            mem_str, mem_wave = self._update_memory_monitor()
            yield self.history_synchronizer.get_chat_history(), "0.000", "", "", mem_str, mem_wave
            return

        # 单帧推理对比
        sf_response = ""
        sf_metrics = {}
        if self.last_frame_pil is not None:
            try:
                sf_response, sf_metrics = self.inference_engine.single_frame_inference(
                    image=self.last_frame_pil,
                    question=question,
                    max_new_tokens=256,
                    min_new_tokens=8,
                    do_sample=False,
                )
                self.last_single_frame_metrics = sf_metrics
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                sf_response = "⚠️ 单帧推理 OOM"
                sf_metrics = {"ttft": 0.0, "vram_peak_gb": 0.0, "input_tokens": 0}
                self.last_single_frame_metrics = sf_metrics

        if sf_response:
            self.history_synchronizer.update("assistant", f"🖼️ [单帧推理 - 仅最后一帧]\n{sf_response}")

        ttft_val = metrics.get("ttft", 0.0)

        # 获取缓存信息
        cache_info = self.inference_engine.get_cache_info()
        cache_str = self._format_cache_status(cache_info)
        comparison = self._build_comparison_text()
        mem_str, mem_wave = self._update_memory_monitor(cache_info)
        yield self.history_synchronizer.get_chat_history(), f"{ttft_val:.3f}", cache_str, comparison, mem_str, mem_wave

    def start_chat(self, video_path, frame_interval, chunk_size, enable_eviction, max_cache_tokens, current_history):
        """
        智能启动：流式编码视频帧。
        
        支持 chunk 编码模式：
          - chunk_size=1: 逐帧 image 模式
          - chunk_size=2,4,6...: 多帧 video chunk 模式（推荐）
        """
        if not video_path:
            raise gr.Error("Please upload a video file.")

        if VideoReader is None:
            raise gr.Error("Missing dependency: decord. Please install it first.")

        # 1. 强制重置标志位（切换到流式模式）
        self.mode = "streaming"
        self.native_frame_buffer = []
        self.pause_event.set()
        self.stop_signal = False 
        self.current_streaming_time = 0.0

        if self.inference_engine is None:
            self.inference_engine = QwenInferenceWrapper()

        self.inference_engine.reset_with_eviction(
            enable_eviction=bool(enable_eviction),
            max_cache_tokens=int(max_cache_tokens),
        )
        self._last_eviction_notified = 0
        self._reset_stream_estimation_state()
        self._reset_memory_monitor()

        print(f"🔄 Loading video for streaming encode: {video_path} "
              f"(eviction={'ON, max=' + str(int(max_cache_tokens)) if enable_eviction else 'OFF'})")
        torch.cuda.empty_cache()

        vr = VideoReader(video_path, ctx=cpu(0))
        avg_fps = vr.get_avg_fps()
        target_fps = 1.0 / frame_interval
        step = max(1, int(avg_fps / target_fps))
        frame_indices = np.arange(0, len(vr), step)

        self.stream_frame_interval = float(frame_interval)
        self.stream_target_fps = float(target_fps)
        self.stream_chunk_size = int(chunk_size)

        self.history_synchronizer.reset()
        fps_display = 0.0
        chunk_size = max(1, int(chunk_size))
        
        print(f"🚀 开始流式编码，总帧数: {len(frame_indices)}, chunk_size={chunk_size}")
        self.inference_engine.log_vram("Start-Encode")
        self.is_streaming = True
        stopped_early = False
        
        try:
            last_timestamp = 0.0
            last_frame = None
            frame_buffer = []  # 累积帧用于 chunk 编码
            time_buffer = []

            for idx, frame_index in enumerate(frame_indices):
                if self.stop_signal:
                    print("🛑 Inference loop stopped by user.")
                    stopped_early = True
                    break

                if not self.pause_event.is_set():
                    self.pause_event.wait()
                    if self.stop_signal:
                        stopped_early = True
                        break
                
                img_np = vr[frame_index].asnumpy()
                img_pil = Image.fromarray(img_np)
                self.last_frame_pil = img_pil
                timestamp = frame_index / avg_fps

                frame_buffer.append(img_pil)
                time_buffer.append(timestamp)

                # 当 buffer 满时进行 chunk 编码
                if len(frame_buffer) >= chunk_size:
                    inference_start = time.perf_counter()

                    if chunk_size == 1:
                        # 单帧 image 模式
                        response = self.inference_engine.process_frame(frame_buffer[0])
                    else:
                        # 多帧 video chunk 模式
                        chunk_fps = target_fps  # 使用采样后的帧率
                        response = self.inference_engine.process_video_chunk(
                            frame_buffer, fps=chunk_fps
                        )

                    encoded_count = len(frame_buffer)  # 记录编码帧数（清空前）
                    chunk_start_t = time_buffer[0]
                    chunk_end_t = time_buffer[-1]
                    frame_buffer = []
                    time_buffer = []
                    self.current_streaming_time = timestamp

                    if idx % 10 == 0:
                        self.inference_engine.log_vram(f"Encode-{idx}")

                    self.history_synchronizer.update(
                        "assistant",
                        f"System: [T={timestamp:.1f}s] {response}"
                    )

                    # 检查淘汰事件并在首次触发时通知
                    evict_info = self.inference_engine.get_cache_info().get('eviction_stats', {})
                    cur_evict_n = evict_info.get('total_evictions', 0)
                    if cur_evict_n > 0 and self._last_eviction_notified == 0:
                        self._last_eviction_notified = cur_evict_n
                        self.history_synchronizer.update(
                            "assistant",
                            f"System: ✂️ KV Cache 淘汰首次触发！"
                            f"Sink={evict_info.get('effective_sink_size', '?')}, "
                            f"Window={evict_info.get('effective_window_size', '?')}"
                        )

                    # 记录chunk token增量用于时间范围估算
                    cache_info_after_chunk = self.inference_engine.get_cache_info()
                    self._record_chunk_token_delta(
                        cache_info_after_chunk,
                        chunk_start_t=chunk_start_t,
                        chunk_end_t=chunk_end_t,
                    )

                    cost_time = time.perf_counter() - inference_start
                    if cost_time > 0:
                        fps_display = encoded_count / cost_time

                last_timestamp = timestamp
                last_frame = img_np

                # 获取缓存信息（含淘汰统计）
                cache_info = self.inference_engine.get_cache_info()
                cache_str = self._format_cache_status(cache_info)
                mem_str, mem_wave = self._update_memory_monitor(cache_info)
                yield timestamp, img_np, None, self.history_synchronizer.get_chat_history(), self._format_fps_display(fps_display), cache_str, mem_str, mem_wave

            # 处理剩余帧
            if frame_buffer and not self.stop_signal:
                if chunk_size == 1:
                    self.inference_engine.process_frame(frame_buffer[0])
                else:
                    self.inference_engine.process_video_chunk(frame_buffer, fps=target_fps)
                cache_info_after_chunk = self.inference_engine.get_cache_info()
                self._record_chunk_token_delta(
                    cache_info_after_chunk,
                    chunk_start_t=time_buffer[0],
                    chunk_end_t=time_buffer[-1],
                )
                frame_buffer = []
                time_buffer = []

        except Exception as e:
            print(f"Runtime Error: {e}")
            raise e
        finally:
            self.is_streaming = False
            self.stop_signal = False 
            self.inference_engine.log_vram("Finished")
            
        if not stopped_early and last_frame is not None:
            cache_info = self.inference_engine.get_cache_info()
            cache_str = self._format_cache_status(cache_info)
            mem_str, mem_wave = self._update_memory_monitor(cache_info)
            yield last_timestamp, last_frame, None, self.history_synchronizer.get_chat_history(), self._format_fps_display(fps_display), cache_str, mem_str, mem_wave

    def _build_comparison_text(self):
        """构建流式 vs 原生 vs 单帧三模式对比文本"""
        s = self.last_streaming_metrics
        n = self.last_native_metrics
        f = self.last_single_frame_metrics

        if not s and not n and not f:
            return ""

        lines = ["══════ 三模式推理对比 ══════"]

        if s:
            s_vram_str = f"{s['vram_peak_gb']}GB" if 'vram_peak_gb' in s else "N/A"
            lines.append(f"📡 流式: TTFT={s.get('ttft', 0):.3f}s | VRAM Peak={s_vram_str}")
        else:
            lines.append("📡 流式: (尚未测试，请先运行 Start Streaming)")

        if n:
            lines.append(
                f"📦 原生: TTFT={n.get('ttft', 0):.3f}s | "
                f"VRAM Peak={n.get('vram_peak_gb', 0):.2f}GB | "
                f"Input Tok={n.get('input_tokens', 0)}"
            )
        else:
            lines.append("📦 原生: (尚未测试，请先运行 Start Native)")

        if f:
            lines.append(
                f"🖼️ 单帧: TTFT={f.get('ttft', 0):.3f}s | "
                f"VRAM Peak={f.get('vram_peak_gb', 0):.2f}GB | "
                f"Input Tok={f.get('input_tokens', 0)}"
            )
        else:
            lines.append("🖼️ 单帧: (尚未测试)")

        lines.append("─── 对比分析 ───")

        # 流式 vs 原生
        if s and n and s.get('ttft', 0) > 0 and n.get('ttft', 0) > 0:
            speedup = n['ttft'] / s['ttft']
            lines.append(f"🚀 流式 vs 原生 TTFT 加速: {speedup:.1f}x")

            s_vram = s.get('vram_peak_gb')
            n_vram = n.get('vram_peak_gb')
            if (isinstance(s_vram, (int, float)) and isinstance(n_vram, (int, float))
                    and s_vram > 0 and n_vram > 0):
                savings = (1 - s_vram / n_vram) * 100
                lines.append(f"💾 流式 vs 原生 VRAM 节省: {savings:.1f}% ({n_vram:.2f}GB → {s_vram:.2f}GB)")

        # 视频理解 vs 单帧
        if f and f.get('ttft', 0) > 0:
            if s and s.get('ttft', 0) > 0:
                ratio = f['ttft'] / s['ttft']
                lines.append(f"⚡ 流式 vs 单帧 TTFT: ×{ratio:.1f} (流式略慢但拥有完整时序理解)")
            if n and n.get('ttft', 0) > 0:
                ratio = n['ttft'] / f['ttft']
                lines.append(f"🐢 原生 vs 单帧 TTFT: ×{ratio:.1f} (原生处理全帧，延迟远高于单帧)")

        if (s or n) and f:
            lines.append("─── 结论 ───")
            lines.append("🎬 视频理解(流式/原生)利用多帧时序信息，回答更准确全面")
            lines.append("🖼️ 单帧仅看到一帧画面，缺乏时序动态理解能力")
            if s:
                lines.append("✅ 流式推理：接近单帧的速度 + 完整视频的理解力 = 最佳方案")

        return "\n".join(lines)

    def start_native_chat(self, video_path, frame_interval, current_history):
        """
        原生推理模式启动：提取帧并缓存，不做实时编码。
        用户提问时一次性将所有帧送入模型推理。
        """
        if not video_path:
            raise gr.Error("请上传视频文件。")
        if VideoReader is None:
            raise gr.Error("缺少依赖: decord。请先安装。")

        # 切换到原生模式
        self.mode = "native"
        self.native_frame_buffer = []
        self.pause_event.set()
        self.stop_signal = False
        self.current_streaming_time = 0.0
        self.history_synchronizer.reset()
        self._reset_stream_estimation_state()
        self._reset_memory_monitor()

        # 重置流式引擎状态
        if self.inference_engine is not None:
            self.inference_engine.reset()
        torch.cuda.empty_cache()

        vr = VideoReader(video_path, ctx=cpu(0))
        avg_fps = vr.get_avg_fps()
        target_fps = 1.0 / frame_interval
        step = max(1, int(avg_fps / target_fps))
        frame_indices = np.arange(0, len(vr), step)

        self.stream_frame_interval = float(frame_interval)
        self.stream_target_fps = float(target_fps)
        self.stream_chunk_size = 1
        self.native_fps = target_fps
        self.is_streaming = True
        stopped_early = False

        print(f"📹 [Native Mode] 提取 {len(frame_indices)} 帧 (仅缓存，不编码)")

        last_frame = None
        fps_display = 0.0

        try:
            for idx, frame_index in enumerate(frame_indices):
                if self.stop_signal:
                    print("🛑 Native frame extraction stopped by user.")
                    stopped_early = True
                    break
                if not self.pause_event.is_set():
                    self.pause_event.wait()
                    if self.stop_signal:
                        stopped_early = True
                        break

                extract_start = time.perf_counter()
                img_np = vr[frame_index].asnumpy()
                img_pil = Image.fromarray(img_np)
                timestamp = frame_index / avg_fps

                self.native_frame_buffer.append(img_pil)
                self.current_streaming_time = timestamp
                last_frame = img_np

                extract_cost = time.perf_counter() - extract_start
                if extract_cost > 0:
                    fps_display = 1.0 / extract_cost

                if idx == 0:
                    self.history_synchronizer.update(
                        "assistant",
                        "System: [Native Mode] 📦 开始提取帧... (不进行模型编码)"
                    )

                status = (
                    f"[Native] 已缓存: {len(self.native_frame_buffer)}/{len(frame_indices)} 帧 | "
                    f"T={timestamp:.1f}s | 无模型编码"
                )
                mem_str, mem_wave = self._update_memory_monitor()
                yield (timestamp, img_np, None,
                       self.history_synchronizer.get_chat_history(),
                      self._format_fps_display(fps_display), status, mem_str, mem_wave)

        except Exception as e:
            print(f"Native extraction error: {e}")
            raise e
        finally:
            self.is_streaming = False
            self.stop_signal = False

        if stopped_early:
            self.history_synchronizer.update(
                "assistant",
                "System: ⏹️ [Native] 已停止帧提取。"
            )
            if last_frame is not None:
                final_status = f"[Native] 已停止: 已缓存 {len(self.native_frame_buffer)} 帧"
                mem_str, mem_wave = self._update_memory_monitor()
                yield (self.current_streaming_time, last_frame, None,
                       self.history_synchronizer.get_chat_history(),
                                             self._format_fps_display(fps_display), final_status, mem_str, mem_wave)
            return

        # 完成消息
        n_frames = len(self.native_frame_buffer)
        self.history_synchronizer.update(
            "assistant",
            f"System: ✅ [Native] 帧缓存完成: {n_frames} 帧 | FPS={self.native_fps:.1f}\n"
            f"💡 请在右侧输入问题，模型将一次性处理所有 {n_frames} 帧进行推理。"
        )

        if last_frame is not None:
            final_status = (
                f"[Native] ✅ Ready: {n_frames} frames | "
                f"FPS={self.native_fps:.1f} | 等待提问"
            )
            mem_str, mem_wave = self._update_memory_monitor()
            yield (self.current_streaming_time, last_frame, None,
                   self.history_synchronizer.get_chat_history(),
                   self._format_fps_display(fps_display), final_status, mem_str, mem_wave)

    def toggle_pause(self):
        if self.pause_event.is_set():
            self.pause_event.clear()
            return "Resume Video", self.history_synchronizer.get_chat_history()
        else:
            self.pause_event.set()
            return "Pause Video", self.history_synchronizer.get_chat_history()

    def stop_chat(self):
        print("🛑 Stop command received.")
        self.stop_signal = True 
        self.pause_event.set() 
        time.sleep(0.1)
        self.is_streaming = False
        self.current_streaming_time = 0.0
        self._last_eviction_notified = 0
        self._reset_stream_estimation_state()
        self._reset_memory_monitor()
        self.native_frame_buffer = []
        self.last_frame_pil = None
        self.last_single_frame_metrics = {}
        self.mode = "streaming"
        if self.inference_engine is not None:
            self.inference_engine.reset()
        return 0, None, None, [], "0.00", "", "", "", "", "<div style='color:#999;'>等待显存数据...</div>"

    def create_interface(self):
        with gr.Blocks(title="Qwen2.5-VL Streaming Video Chat") as demo:
            pil_frames_state = gr.State()

            gr.Markdown("# 🎬 Qwen2.5-VL Streaming Video Chat\n流式视频理解 + 实时问答 | 支持流式 vs 原生推理对比")

            with gr.Row():
                with gr.Column(scale=3):
                    gr_frame_display = gr.Image(label="Current Frame", interactive=False, height=400)
                    with gr.Row():
                        gr_time_display = gr.Number(label="Video Time (s)", value=0)
                        gr_fps_display = gr.Textbox(label="Encode Throughput (frames/s)", value="0.00")
                        gr_ttft_display = gr.Textbox(label="Ask TTFT (s)", value="")
                    with gr.Row():
                        gr_cache_display = gr.Textbox(label="KV Cache Status", value="", interactive=False)
                    with gr.Row():
                        gr_mem_display = gr.Textbox(label="GPU Memory Breakdown (GB)", value="", interactive=False)
                    with gr.Row():
                        gr_mem_wave = gr.HTML(value="<div style='color:#999;'>等待显存数据...</div>")
                    with gr.Row():
                        gr_pause_button = gr.Button("Pause Video")
                        gr_stop_button = gr.Button("Stop Video", variant="stop")

                with gr.Column(scale=2):
                    gr_chat_interface = gr.Chatbot(label="Chat History", height=500)
                    gr_question_input = gr.Textbox(label="Manual Question (Auto-pauses video)")

            with gr.Row():
                with gr.Column():
                    gr_video_upload = gr.Video(label="1. Upload Video")
                with gr.Column():
                    gr_frame_interval = gr.Slider(
                        minimum=0.1, maximum=5.0, step=0.1, value=1.0,
                        interactive=True, label="2. Frame Interval (s)"
                    )
                    gr_chunk_size = gr.Slider(
                        minimum=1, maximum=8, step=1, value=4,
                        interactive=True, label="3. Chunk Size (frames, 推荐 2/4/6)"
                    )
                    gr.Markdown("编码吞吐=模型编码速度（frames/s），括号内为相对采样视频速率的实时倍率。小于 1 通常表示编码速度低于实时播放速度。")
                    gr.Markdown("#### ✂️ KV Cache 淘汰 (Sink + Sliding Window)")
                    with gr.Row():
                        gr_eviction_enable = gr.Checkbox(
                            label="启用淘汰 (防 OOM，支持无限长视频)",
                            value=True,
                            interactive=True,
                        )
                        gr_max_cache_tokens = gr.Slider(
                            minimum=50000, maximum=200000, step=10000, value=150000,
                            interactive=True,
                            label="Max Cache Tokens (越大保留越多历史，VRAM越高)",
                        )
                    with gr.Row():
                        gr_start_button = gr.Button("4a. Start Streaming ▶️", variant="primary")
                        gr_start_native_button = gr.Button("4b. Start Native 📦", variant="secondary")

            with gr.Row():
                gr_comparison_display = gr.Textbox(
                    label="📊 三模式对比: Streaming vs Native vs Single-Frame",
                    value="分别运行流式和原生模式后，提问时自动对比三种推理模式",
                    lines=8,
                    interactive=False,
                )

            gr_question_input.submit(
                self.generate_answer,
                inputs=[gr_question_input],
                outputs=[gr_chat_interface, gr_ttft_display, gr_cache_display, gr_comparison_display, gr_mem_display, gr_mem_wave],
                queue=True,
            )
            gr_start_button.click(
                self.start_chat,
                inputs=[gr_video_upload, gr_frame_interval, gr_chunk_size,
                        gr_eviction_enable, gr_max_cache_tokens, gr_chat_interface],
                outputs=[gr_time_display, gr_frame_display, pil_frames_state,
                         gr_chat_interface, gr_fps_display, gr_cache_display, gr_mem_display, gr_mem_wave],
            )
            gr_start_native_button.click(
                self.start_native_chat,
                inputs=[gr_video_upload, gr_frame_interval, gr_chat_interface],
                outputs=[gr_time_display, gr_frame_display, pil_frames_state, gr_chat_interface, gr_fps_display, gr_cache_display, gr_mem_display, gr_mem_wave],
            )
            gr_pause_button.click(
                self.toggle_pause,
                inputs=[],
                outputs=[gr_pause_button, gr_chat_interface],
            )
            gr_stop_button.click(
                self.stop_chat,
                inputs=[],
                outputs=[gr_time_display, gr_frame_display, pil_frames_state, gr_chat_interface, gr_fps_display, gr_ttft_display, gr_cache_display, gr_comparison_display, gr_mem_display, gr_mem_wave],
            )
        
        return demo
