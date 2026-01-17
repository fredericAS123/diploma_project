import gradio as gr
import threading
import time
import torch
from Qwen_inference import QwenInferenceWrapper

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
        
        # [数据缓存]
        self.cached_video_path = None 
        self.cached_video_data = None 

    def generate_answer(self, question, pil_frames_state):
        """手动提问：处理自动暂停逻辑"""
        if pil_frames_state is None or len(pil_frames_state) == 0:
             yield self.history_synchronizer.get_chat_history() + [gr.ChatMessage(role="assistant", content="请先播放视频")]
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
            
        yield self.history_synchronizer.get_chat_history() + [gr.ChatMessage(role="assistant", content=status_msg)]

        self.inference_engine.log_vram("Manual-Pre")
        start_t = time.perf_counter()
        
        # 上下文策略：只取最后一帧
        context_frames = pil_frames_state[-1:] if len(pil_frames_state) >= 1 else pil_frames_state
        
        try:
            # 调用封装好的推理引擎
            response = self.inference_engine.predict(context_frames, question, use_system_prompt=False)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            response = "Error: OOM (显存不足)。请尝试减少视频帧数。"
        
        cost = time.perf_counter() - start_t
        print(f"Manual Inference Latency: {cost:.4f}s")
        self.inference_engine.log_vram("Manual-Post")

        self.history_synchronizer.update("assistant", response)
        yield self.history_synchronizer.get_chat_history()

    def start_chat(self, video_path, frame_interval, current_history):
        """智能启动：缓存复用 + 修复 Stop/Restart"""
        if not video_path:
            raise gr.Error("Please upload a video file.")

        # 1. 强制重置标志位
        self.pause_event.set()
        self.stop_signal = False 

        # 2. 缓存检查
        if video_path == self.cached_video_path and self.cached_video_data is not None:
            print(f"⚡ Cache hit! Reusing video data for: {video_path}")
            pil_frames_all, original_frames, timestamps = self.cached_video_data
        else:
            print(f"🔄 New video detected. Loading via Inference Engine...")
            self.cached_video_data = None 
            torch.cuda.empty_cache()
            
            # 调用推理引擎加载视频
            pil_frames_all, original_frames, timestamps = self.inference_engine.load_video(
                video_path, fps=1 / frame_interval
            )
            self.cached_video_path = video_path
            self.cached_video_data = (pil_frames_all, original_frames, timestamps)

        self.history_synchronizer.reset()
        fps_display = 0.0
        
        print(f"🚀 开始流式推理，总帧数: {len(pil_frames_all)}")
        self.inference_engine.log_vram("Start-Loop")
        self.is_streaming = True
        
        try:
            for idx, (pil_frame, original_frame, timestamp) in enumerate(
                zip(pil_frames_all, original_frames, timestamps)
            ):
                if self.stop_signal:
                    print("🛑 Inference loop stopped by user.")
                    break

                if not self.pause_event.is_set():
                    self.pause_event.wait()
                    if self.stop_signal: break
                
                inference_start = time.perf_counter()
                
                # 调用推理引擎
                response = self.inference_engine.predict([pil_frame], "Report status.")
                
                self.history_synchronizer.update("assistant", f"[{timestamp:.1f}s] {response}")

                cost_time = time.perf_counter() - inference_start
                if cost_time > 0:
                    current_fps = 1.0 / cost_time
                    fps_display = 0.8 * fps_display + 0.2 * current_fps if idx > 0 else current_fps

                yield timestamp, original_frame, pil_frames_all[: idx + 1], self.history_synchronizer.get_chat_history(), f"{fps_display:.2f}"
        
        except Exception as e:
            print(f"Runtime Error: {e}")
            raise e
        finally:
            self.is_streaming = False
            self.stop_signal = False 
            self.inference_engine.log_vram("Finished")
            
        if not self.stop_signal:
            yield timestamps[-1], original_frames[-1], pil_frames_all, self.history_synchronizer.get_chat_history(), f"{fps_display:.2f}"

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
        return 0, None, None, [], "0.00"

    def create_interface(self):
        with gr.Blocks(title="Qwen2.5-VL Video Chat (Modular)") as demo:
            pil_frames_state = gr.State()

            with gr.Row():
                with gr.Column(scale=3):
                    gr_frame_display = gr.Image(label="Current Frame", interactive=False, height=400)
                    with gr.Row():
                        gr_time_display = gr.Number(label="Video Time (s)", value=0)
                        gr_fps_display = gr.Textbox(label="Inference FPS", value="0.00")
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

            gr_question_input.submit(self.generate_answer, inputs=[gr_question_input, pil_frames_state], outputs=gr_chat_interface, queue=True)
            gr_start_button.click(self.start_chat, inputs=[gr_video_upload, gr_frame_interval, gr_chat_interface], outputs=[gr_time_display, gr_frame_display, pil_frames_state, gr_chat_interface, gr_fps_display])
            gr_pause_button.click(self.toggle_pause, inputs=[], outputs=[gr_pause_button, gr_chat_interface])
            gr_stop_button.click(self.stop_chat, inputs=[], outputs=[gr_time_display, gr_frame_display, pil_frames_state, gr_chat_interface, gr_fps_display])
        
        return demo