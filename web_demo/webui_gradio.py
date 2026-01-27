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

    def reset(self):
        self.chat_history = []

class VideoChatWebUI:
    def __init__(self, inference_engine: QwenInferenceWrapper):
        """
        初始化 UI 逻辑，传入推理引擎实例
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

    def generate_answer(self, question):
        """手动提问：处理自动暂停逻辑"""
        if not self.is_streaming and self.current_streaming_time <= 0:
            yield self.history_synchronizer.get_chat_history() + [
                gr.ChatMessage(role="assistant", content="请先播放视频")
            ], ""
            return

        # 自动暂停机制
        auto_paused = False
        if self.is_streaming and self.pause_event.is_set():
            print("⚠️ Detected streaming active. Auto-pausing for manual question...")
            self.pause_event.clear() # 暂停流式循环
            auto_paused = True
            time.sleep(0.1)

        self.history_synchronizer.update("user", question)
        
        status_msg = "Thinking..."
        if auto_paused:
            status_msg += " (Video Auto-Paused)"
            
        yield self.history_synchronizer.get_chat_history() + [gr.ChatMessage(role="assistant", content=status_msg)], ""

        try:
            # 调用封装好的推理引擎
            start_t = time.perf_counter()
            response, metrics = self.inference_engine.ask_question(
                question,
                manual_time=self.current_streaming_time,
                max_new_tokens=256,
                min_new_tokens=8,
                do_sample=False,
                temperature=0.7,
                top_p=0.9,
                return_metrics=True,
            )
            cost = time.perf_counter() - start_t
            print(f"Manual Inference Latency: {cost:.4f}s")
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            response = "Error: OOM (显存不足)。请尝试减少视频帧数。"
            metrics = {"ttft": 0.0}
        

        self.history_synchronizer.update("assistant", response)
        ttft_val = metrics.get("ttft", 0.0)
        yield self.history_synchronizer.get_chat_history(), f"{ttft_val:.3f}"

    def start_chat(self, video_path, frame_interval, current_history):
        """智能启动：缓存复用 + 修复 Stop/Restart"""
        if not video_path:
            raise gr.Error("Please upload a video file.")

        if VideoReader is None:
            raise gr.Error("Missing dependency: decord. Please install it first.")

        # 1. 强制重置标志位
        self.pause_event.set()
        self.stop_signal = False 
        self.current_streaming_time = 0.0

        if self.inference_engine is None:
            self.inference_engine = QwenInferenceWrapper()

        self.inference_engine.reset()

        print(f"🔄 Loading video for streaming encode: {video_path}")
        torch.cuda.empty_cache()

        vr = VideoReader(video_path, ctx=cpu(0))
        avg_fps = vr.get_avg_fps()
        target_fps = 1.0 / frame_interval
        step = max(1, int(avg_fps / target_fps))
        frame_indices = np.arange(0, len(vr), step)

        self.history_synchronizer.reset()
        fps_display = 0.0
        
        print(f"🚀 开始流式编码，总帧数: {len(frame_indices)}")
        self.inference_engine.log_vram("Start-Encode")
        self.is_streaming = True
        
        try:
            last_timestamp = 0.0
            last_frame = None
            for idx, frame_index in enumerate(frame_indices):
                if self.stop_signal:
                    print("🛑 Inference loop stopped by user.")
                    break

                if not self.pause_event.is_set():
                    self.pause_event.wait()
                    if self.stop_signal: break
                
                inference_start = time.perf_counter()

                img_np = vr[frame_index].asnumpy()
                img_pil = Image.fromarray(img_np)
                timestamp = frame_index / avg_fps

                response = self.inference_engine.process_frame(img_pil, manual_time=timestamp)
                self.current_streaming_time = timestamp
                if idx % 10 == 0:
                    self.inference_engine.log_vram(f"Encode-{idx}")

                self.history_synchronizer.update(
                    "assistant",
                    f"System: [T={timestamp:.1f}s] {response}"
                )

                last_timestamp = timestamp
                last_frame = img_np

                cost_time = time.perf_counter() - inference_start
                if cost_time > 0:
                    current_fps = 1.0 / cost_time
                    # fps_display = 0.8 * fps_display + 0.2 * current_fps if idx > 0 else current_fps
                    fps_display = current_fps
                yield timestamp, img_np, None, self.history_synchronizer.get_chat_history(), f"{fps_display:.2f}"
        
        except Exception as e:
            print(f"Runtime Error: {e}")
            raise e
        finally:
            self.is_streaming = False
            self.stop_signal = False 
            self.inference_engine.log_vram("Finished")
            
        if not self.stop_signal and last_frame is not None:
            yield last_timestamp, last_frame, None, self.history_synchronizer.get_chat_history(), f"{fps_display:.2f}"

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
        if self.inference_engine is not None:
            self.inference_engine.reset()
        return 0, None, None, [], "0.00", ""

    def create_interface(self):
        with gr.Blocks(title="Qwen2.5-VL Video Chat (Modular)") as demo:
            pil_frames_state = gr.State()

            with gr.Row():
                with gr.Column(scale=3):
                    gr_frame_display = gr.Image(label="Current Frame", interactive=False, height=400)
                    with gr.Row():
                        gr_time_display = gr.Number(label="Video Time (s)", value=0)
                        gr_fps_display = gr.Textbox(label="Inference FPS", value="0.00")
                        gr_ttft_display = gr.Textbox(label="Ask TTFT (s)", value="")
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
                    gr_frame_interval = gr.Slider(minimum=0.1, maximum=5.0, step=0.1, value=1.0, interactive=True, label="2. Frame Interval")
                    gr_start_button = gr.Button("3. Start Online Inference", variant="primary")

            gr_question_input.submit(
                self.generate_answer,
                inputs=[gr_question_input],
                outputs=[gr_chat_interface, gr_ttft_display],
                queue=True,
            )
            gr_start_button.click(self.start_chat, inputs=[gr_video_upload, gr_frame_interval, gr_chat_interface], outputs=[gr_time_display, gr_frame_display, pil_frames_state, gr_chat_interface, gr_fps_display])
            gr_pause_button.click(self.toggle_pause, inputs=[], outputs=[gr_pause_button, gr_chat_interface])
            gr_stop_button.click(
                self.stop_chat,
                inputs=[],
                outputs=[gr_time_display, gr_frame_display, pil_frames_state, gr_chat_interface, gr_fps_display, gr_ttft_display],
            )
        
        return demo
