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
    print("Error: modelscope or qwen_vl_utils not found. Please run: pip install modelscope qwen-vl-utils")


MODEL_PATH = "Qwen/Qwen2.5-VL-3B-Instruct"  # correct with local model path


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
        初始化：加载 Qwen2.5-VL 模型 (使用 ModelScope)
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Qwen2.5-VL model from: {model_path} ...")
        
        try:
            # [修正] 使用 Qwen2_5_VLForConditionalGeneration 加载
            # 推荐开启 flash_attention_2 以节省显存加速推理 (AutoDL通常支持)
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype="auto",
                device_map="auto",
                # attn_implementation="flash_attention_2" # if supported, else comment
            )
            
            # 加载处理器
            # min_pixels / max_pixels 可按需设置，这里使用默认
            self.processor = AutoProcessor.from_pretrained(model_path)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("请检查 modelscope 是否安装，或显存是否足够。")
            self.model = None
            self.processor = None

        self.history_synchronizer = HistorySynchronizer()
        self.pause_event = threading.Event()
        self.pause_event.set() 

    def _load_video(self, video_path, fps=1):
        """
        加载视频并抽帧，返回 PIL 图片列表
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

    def _run_qwen_inference(self, pil_frames, prompt_text):
        """
        [核心推理] 增加 System Prompt 以实现静默控制
        """
        if self.model is None:
            return "Error: Model not loaded."

        # =================================================================
        # [核心修改 1] 定义强硬的 System Prompt (安保员人设)
        # =================================================================
        SYSTEM_PROMPT = (
            "You are a professional video surveillance AI. Your job is to monitor the stream."
            "\n\nRULES:"
            "\n1. If the scene is static, empty, or contains only insignificant background movements (like trees blowing), output EXACTLY: '[WAIT]'."
            "\n2. ONLY output a description if there is a meaningful EVENT or ACTION happening (e.g., a person enters, a car moves, someone falls)."
            "\n3. Be extremely concise. One sentence only."
            "\n4. Do not repeat previous information."
        )

        # 1. 构造 Messages
        # 这里的 content_list 是图像内容
        content_list = []
        for img in pil_frames:
            content_list.append({"type": "image", "image": img})
        
        # 用户指令
        content_list.append({"type": "text", "text": prompt_text})

        # [核心修改 2] 在 User 之前插入 System
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
        """手动提问"""
        if pil_frames_state is None or len(pil_frames_state) == 0:
             yield self.history_synchronizer.get_chat_history() + [gr.ChatMessage(role="assistant", content="请先播放视频")]
             return 

        self.history_synchronizer.update("user", question)
        current_history = self.history_synchronizer.get_chat_history()
        yield current_history + [gr.ChatMessage(role="assistant", content="Thinking...")]

        # 推理
        start_t = time.perf_counter()
        # 将所有帧喂给模型
        response = self._run_qwen_inference(pil_frames_state, question)
        cost = time.perf_counter() - start_t
        print(f"Manual Inference Latency: {cost:.4f}s")

        self.history_synchronizer.update("assistant", response)
        yield self.history_synchronizer.get_chat_history()

    def start_chat(self, video_path, frame_interval, current_history):
        """流式处理"""
        if not video_path:
            raise gr.Error("Please upload a video file.")

        pil_frames_all, original_frames, timestamps = self._load_video(
            video_path, fps=1 / frame_interval
        )

        self.history_synchronizer.reset()
        fps_display = 0.0

        for idx, (pil_frame, original_frame, timestamp) in enumerate(
            zip(pil_frames_all, original_frames, timestamps)
        ):
            if not self.pause_event.is_set():
                self.pause_event.wait()
            
            inference_start = time.perf_counter()

            # [自动解说逻辑]
            # 为了速度，只取最近的 1 帧进行描述
            context_frames = pil_frames_all[max(0, idx): idx+1] 
            prompt = "Describe this frame concisely."
            
            # 调用真实模型
            response = self._run_qwen_inference(context_frames, prompt)
            
            # 更新历史
            self.history_synchronizer.update("assistant", f"[{timestamp:.1f}s] {response}")

            inference_end = time.perf_counter()
            
            # 计算 FPS
            cost_time = inference_end - inference_start
            if cost_time > 0:
                current_fps = 1.0 / cost_time
                fps_display = 0.8 * fps_display + 0.2 * current_fps if idx > 0 else current_fps

            current_chat_history = self.history_synchronizer.get_chat_history()
            
            yield timestamp, original_frame, pil_frames_all[: idx + 1], current_chat_history, f"{fps_display:.2f}"

        yield timestamps[-1], original_frames[-1], pil_frames_all, self.history_synchronizer.get_chat_history(), f"{fps_display:.2f}"

    def toggle_pause(self):
        if self.pause_event.is_set():
            self.pause_event.clear()
            return "Resume Video", self.history_synchronizer.get_chat_history()
        else:
            self.pause_event.set()
            return "Pause Video", self.history_synchronizer.get_chat_history()

    def stop_chat(self):
        self.pause_event.clear()
        self.history_synchronizer.reset()
        return 0, None, None, [], "0.00"

    # --- UI 构建函数 ---
    def create_interface(self):
        with gr.Blocks(title="Qwen2.5-VL Online Video Chat") as demo:
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
                    gr_question_input = gr.Textbox(label="Manual Question")

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
    print("Launching WebUI...")
    # 启用公网端口 6006
    demo.launch(server_name="0.0.0.0", server_port=6006, share=False, debug=True)