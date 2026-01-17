import re
import threading
import gradio as gr
import torch
from PIL import Image
import numpy as np
import time

# --- 引入必要的库 ---
try:
    from decord import VideoReader, cpu
except ImportError:
    print("Error: 'decord' not found. Please run `pip install decord`")

try:
    from modelscope import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
except ImportError:
    # 备选方案：如果 modelscope 报错，尝试直接从 transformers 导入
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        from qwen_vl_utils import process_vision_info
    except ImportError:
        print("Error: libs not found. Please run: pip install modelscope qwen-vl-utils transformers")


MODEL_PATH = "Qwen/Qwen2.5-VL-3B-Instruct" 

class HistorySynchronizer:
    def __init__(self):
        self.chat_history = []
        self.frame_count = 0

    def get_chat_history(self):
        return self.chat_history

    def update(self, role, content):
        self.chat_history.append(gr.ChatMessage(role=role, content=str(content)))

    def set_history(self, history):
        self.chat_history = history

    def reset(self):
        self.chat_history = []
        self.frame_count = 0

# --- WebUI 类 ---
class VideoChatWebUI:
    def __init__(self, model_path=MODEL_PATH):
        """
        初始化：加载 Qwen2.5-VL 模型
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Qwen2.5-VL model from: {model_path} ...")
        
        try:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype="auto",
                device_map="auto",
            )
            self.processor = AutoProcessor.from_pretrained(model_path)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
            self.processor = None

        self.history_synchronizer = HistorySynchronizer()
        
        # [控制标志]
        self.pause_event = threading.Event()
        self.pause_event.set() 
        self.stop_signal = False  # [新增] 用于强制中断循环的标志
        
        # [线程安全]
        self.inference_lock = threading.Lock() 
        self.is_streaming = False            
        
        # [数据缓存]
        self.cached_video_path = None # [新增] 记录上次加载的视频路径
        self.cached_video_data = None # [新增] 存储 (pil_frames, original_frames, timestamps)
        
        self.log_vram("Init")


    def _load_video(self, video_path, fps=1):
        """
        加载视频并抽帧
        """
        vr = VideoReader(video_path, ctx=cpu(0))
        max_frame = len(vr) - 1
        frame_indices = np.arange(0, max_frame, int(vr.get_avg_fps() / fps))
        
        pil_frames = []
        original_frames = []
        
        print(f"Processing video, extracting {len(frame_indices)} frames...")
        for frame_index in frame_indices:
            img_np = vr[frame_index].asnumpy()
            img_pil = Image.fromarray(img_np)
            pil_frames.append(img_pil)
            original_frames.append(img_np)

        timestamps = frame_indices / vr.get_avg_fps()
        return pil_frames, original_frames, timestamps

    def log_vram(self, tag=""):
        """
        [工具] 打印当前显存占用
        """
        if torch.cuda.is_available():
            # 已分配：Tensor 实际占用的显存
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            # 已预留：PyTorch 向 OS 申请的总显存 (包含碎片)
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            print(f"📊 [VRAM-{tag}] Alloc: {allocated:.2f}GB | Reserved: {reserved:.2f}GB")

    def _run_qwen_inference(self, pil_frames, prompt_text, use_system_prompt=True):
        """
        [普通推理] 每次都重新计算，不使用 KV Cache
        [新增]使用 self.inference_lock 确保线程安全
        """
        if self.model is None:
            return "Error: Model not loaded."

        # 使用锁包裹推理过程，防止流式推理和手动推理撞车
        with self.inference_lock:
            # System Prompt
            SYSTEM_PROMPT = (
                        "You are a professional video surveillance AI. "
                        "When asked to 'Report status', follow these rules strictly:"
                        "\n1. If the scene is static, empty, or contains only insignificant background movements (like trees blowing), output EXACTLY: '[WAIT]'."
                        "\n2. ONLY output a description if there is a meaningful EVENT or ACTION happening."
                        "\n3. Be concise."
                        "\n4. Do not repeat previous information."
                    )

        content_list = []
        for img in pil_frames:
            content_list.append({"type": "image", "image": img})
        
        # 用户指令
        content_list.append({"type": "text", "text": prompt_text})

        # 根据参数决定是否包含系统提示
        if use_system_prompt:
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": SYSTEM_PROMPT}]
                },
                {
                    "role": "user",
                    "content": content_list
                }
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": content_list
                }
            ]

        # 2. 准备推理
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        # 3. 生成
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=128,
                temperature=0.1, # 低温度对于指令遵循很重要
                top_p=0.9
            )

        # 4. 解码
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return output_text

    # --- 事件处理函数 ---

    def generate_answer(self, question, pil_frames_state):
        """
        [修改] 手动提问：增加了自动暂停流式处理的逻辑
        """
        if pil_frames_state is None or len(pil_frames_state) == 0:
             yield self.history_synchronizer.get_chat_history() + [gr.ChatMessage(role="assistant", content="请先播放视频")]
             return 

        # [新增 3] 自动暂停机制
        # 如果当前正在流式播放且未暂停，则强制暂停，避免抢占显存
        auto_paused = False
        if self.is_streaming and self.pause_event.is_set():
            print("⚠️ Detected streaming active. Auto-pausing for manual question...")
            self.pause_event.clear() # 暂停流式循环
            auto_paused = True
            # 给一点时间让 start_chat 循环响应暂停
            time.sleep(0.1)

        self.history_synchronizer.update("user", question)
        
        # 提示用户如果有自动暂停
        status_msg = "Thinking..."
        if auto_paused:
            status_msg += " (Video Auto-Paused)"
            
        yield self.history_synchronizer.get_chat_history() + [gr.ChatMessage(role="assistant", content=status_msg)]

        # [显存监控] 提问前
        self.log_vram("Manual-Pre")

        start_t = time.perf_counter()
        
        # 修改：只传入最后一帧，避免显存溢出
        context_frames = pil_frames_state[-1:] if len(pil_frames_state) >= 2 else pil_frames_state
        
        try:
            # 这里的 _run_qwen_inference 内部已经有锁了，所以是安全的
            response = self._run_qwen_inference(context_frames, question, use_system_prompt=False)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            response = "Error: OOM (显存不足)。请尝试减少视频帧数。"
        
        cost = time.perf_counter() - start_t
        print(f"Manual Inference Latency: {cost:.4f}s")
        
        # [显存监控] 提问后
        self.log_vram("Manual-Post")

        self.history_synchronizer.update("assistant", response)
        yield self.history_synchronizer.get_chat_history()

    def start_chat(self, video_path, frame_interval, current_history):
        """
        [修改] 智能启动：支持缓存复用，并修复了Stop后无法Restart的Bug
        """
        if not video_path:
            raise gr.Error("Please upload a video file.")

        # 1. [关键修复] 强制重置暂停状态，防止 Stop 后的死锁
        self.pause_event.set()
        self.stop_signal = False # 重置停止标志

        # 2. [缓存优化] 检查是否可以复用视频数据
        if video_path == self.cached_video_path and self.cached_video_data is not None:
            print(f"⚡ Cache hit! Reusing video data for: {video_path}")
            pil_frames_all, original_frames, timestamps = self.cached_video_data
        else:
            print(f"🔄 New video detected. Loading: {video_path}")
            # 如果是新视频，清理旧缓存以防显存泄露
            self.cached_video_data = None 
            torch.cuda.empty_cache()
            
            pil_frames_all, original_frames, timestamps = self._load_video(
                video_path, fps=1 / frame_interval
            )
            # 更新缓存
            self.cached_video_path = video_path
            self.cached_video_data = (pil_frames_all, original_frames, timestamps)

        self.history_synchronizer.reset()
        fps_display = 0.0
        
        print(f"🚀 开始推理，总帧数: {len(pil_frames_all)}")
        self.log_vram("Start")

        self.is_streaming = True
        
        try:
            for idx, (pil_frame, original_frame, timestamp) in enumerate(
                zip(pil_frames_all, original_frames, timestamps)
            ):
                # [新增] 检查强制停止标志
                if self.stop_signal:
                    print("🛑 Inference loop stopped by user.")
                    break

                # 检查暂停状态
                if not self.pause_event.is_set():
                    self.pause_event.wait()
                    # 唤醒后再次检查停止标志（防止在暂停时点了停止）
                    if self.stop_signal: 
                        break
                
                inference_start = time.perf_counter()
                
                context_frames = [pil_frame] 
                prompt = "Report status."
                
                response = self._run_qwen_inference(context_frames, prompt)
                
                self.history_synchronizer.update("assistant", f"[{timestamp:.1f}s] {response}")

                inference_end = time.perf_counter()
                
                cost_time = inference_end - inference_start
                if cost_time > 0:
                    current_fps = 1.0 / cost_time
                    fps_display = 0.8 * fps_display + 0.2 * current_fps if idx > 0 else current_fps

                current_chat_history = self.history_synchronizer.get_chat_history()
                yield timestamp, original_frame, pil_frames_all[: idx + 1], current_chat_history, f"{fps_display:.2f}"
        
        except Exception as e:
            print(f"Runtime Error: {e}")
            raise e
        finally:
            self.is_streaming = False
            self.stop_signal = False # 恢复标志位
            self.log_vram("Finished")
            
        # 如果不是被强制停止的，才输出最后结果
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
        """
        [修改] 停止逻辑：设置标志位，并确保从暂停中唤醒以便退出
        """
        print("🛑 Stop command received.")
        self.stop_signal = True # 1. 设置停止标志
        self.pause_event.set()  # 2. [关键] 如果当前处于暂停等待中，必须唤醒它，它才能检测到 stop_signal 并 break
        
        # 稍微给一点时间让循环退出
        time.sleep(0.1)
        self.is_streaming = False
        
        # 返回重置 UI 的值
        return 0, None, None, [], "0.00"

    # --- UI 构建函数 ---
    def create_interface(self):
        with gr.Blocks(title="Qwen2.5-VL Video Chat (No KV Cache)") as demo:
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

if __name__ == "__main__":
    web_ui = VideoChatWebUI()
    demo = web_ui.create_interface()
    print("Launching WebUI (No KV Cache)...")
    demo.launch(server_name="0.0.0.0", server_port=6006, share=False, debug=True)