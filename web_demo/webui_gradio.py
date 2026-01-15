import re
import threading
import gradio as gr
import torch
from PIL import Image
import numpy as np
import time
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

# 尝试导入 decord，如果没安装则提示
try:
    from decord import VideoReader, cpu
except ImportError:
    print("Error: 'decord' not found. Please run `pip install decord`")

# try:
#     from transformers import AutoTokenizer
#     from internvl.model.videochat_online import (
#         VideoChatOnline_IT,
#         InternVLChatConfig,
#     )
# except ImportError:
#     pass

# --- 1. 配置 ---
# model_name = "work_dirs/VideoChatOnline_Stage2"

# 图像预处理常量
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# --- 2. 辅助类与函数 ---

class HistorySynchronizer:
    def __init__(self):
        self.history = []
        self.chat_history = []
        self.frame_count = 0

    def set_history(self, history):
        self.history = history

    def get_history(self):
        return self.history
    
    def get_chat_history(self):
        return self.chat_history

    def clean_history_item(self, item):
        return re.sub(r"Frame\d+: <image>", "", item).strip()

    def update(self, new_msg):
        new_msg["content"] = self.clean_history_item(new_msg["content"])
        new_msg = gr.ChatMessage(role=new_msg["role"], content=new_msg["content"])
        if self.chat_history:
            self.chat_history.append(new_msg)
        else:
            self.chat_history = [new_msg]

    def reset(self):
        self.history = []
        self.chat_history = []
        self.frame_count = 0

# def build_transform(input_size):
#     transform = T.Compose([
#         T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
#         T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
#         T.ToTensor(),
#         T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
#     ])
#     return transform

# --- 3. WebUI 类封装 ---

class VideoChatWebUI:
    def __init__(self):
        """
        初始化 UI 类
        """
        # self.model_path = model_path   用的时候记得init里上传modelpath
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        # =========================================================================
        # [MOCK] 模型加载部分已注释，用于纯 UI 和 逻辑流程测试
        # =========================================================================
        print(f"!!! UI TEST MODE: Mocking model loading !!!")
        
        # 实际代码中取消注释以下部分：
        '''
        print("Loading model...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, add_eos_token=False, trust_remote_code=True, use_fast=False
        )
        config = InternVLChatConfig.from_pretrained(model_path, trust_remote_code=True)
        self.model = (
            VideoChatOnline_IT.from_pretrained(
                model_path,
                config=config,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            .eval()
            .to(self.device)
        )
        self.system_prompt = "Carefully watch the video..."
        self.model.system_message = self.system_prompt
        '''
        
        self.generation_config = dict(
            max_new_tokens=256, 
            do_sample=False, 
            num_beams=1, 
            temperature=0.95
        )

        # 内部状态管理
        self.history_synchronizer = HistorySynchronizer()
        self.pause_event = threading.Event()
        self.pause_event.set() # 默认非暂停状态
    @staticmethod
    def _build_transform(input_size):
        """
        [NEW] 这是一个静态方法，因为它不需要访问 self.model 或 self.tokenizer
        """
        MEAN = (0.485, 0.456, 0.406) # 或者引用 VideoChatWebUI.IMAGENET_MEAN
        STD = (0.229, 0.224, 0.225)
        
        transform = T.Compose([
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ])
        return transform
    
    def _load_video(self, video_path, fps=1, input_size=448):
        """内部视频加载处理函数"""
        vr = VideoReader(video_path, ctx=cpu(0))
        max_frame = len(vr) - 1
        frame_indices = np.arange(0, max_frame, int(vr.get_avg_fps() / fps))
        frames = []
        original_frames = []
        transform = self._build_transform(input_size)

        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].asnumpy()).convert("RGB")
            original_frames.append(np.array(img))
            frames.append(transform(img))

        pixel_values = torch.stack(frames)
        original_frames = np.stack(original_frames)
        timestamps = frame_indices / vr.get_avg_fps()

        return pixel_values, original_frames, timestamps

    # --- 事件处理函数 ---

    def generate_answer(self, question, video_frame_data):
        """
        主动交互函数：处理用户手动的任意问题 (如"车里有几个人?")
        """
        # 1. 边界检查：如果没有视频数据，直接返回提示
        if video_frame_data is None:
             yield self.history_synchronizer.get_chat_history() + [gr.ChatMessage(role="assistant", content="Error: Video data is missing. Please upload and start a video first.")]
             return 

        # 2. 构造 Prompt 前缀
        # 逻辑：将新出现的帧标记为 <image> 拼接到 prompt 前面
        # 例如: "Frame1: <image>\nFrame2: <image>\nDescribe the video."
        video_prefix = "".join([
            f"Frame{self.history_synchronizer.frame_count+i+1}: <image>\n"
            for i in range(len(video_frame_data[self.history_synchronizer.frame_count :]))
        ])
        
        # 更新类内部的帧计数器
        self.history_synchronizer.frame_count = len(video_frame_data)
        
        # 拼接完整问题
        full_question = video_prefix + question
        
        # 3. 更新 UI：立即显示用户的问题
        self.history_synchronizer.update({"role": "user", "content": question})
        current_chat_history = self.history_synchronizer.get_chat_history()
        
        # 4. 显示 "Generating..." 状态
        temp_chat = current_chat_history.copy()
        temp_chat.append(gr.ChatMessage(role="assistant", content="Generating..."))
        yield temp_chat 

        # ==================================================================================
        # [AREA 1: 真实模型推理] (测试UI时保持注释，正式运行时取消注释)
        # ==================================================================================
        """
        # [TODO: UNCOMMENT] 取消以下注释以启用真实推理
        
        # A. 将数据移动到显卡 (确保 self.model 已加载)
        pixel_values = video_frame_data.to(self.model.device).to(self.model.dtype)

        # B. 记录开始时间
        llm_start_time = time.perf_counter()
        
        # C. 调用 InternVL chat 接口
        llm_message, history = self.model.chat(
            self.tokenizer,
            pixel_values,
            full_question,
            self.generation_config,
            history=self.history_synchronizer.get_history(), # 传入之前累积的历史
            return_history=True,
            verbose=False,
        )
        print(f"LLM Latency: {time.perf_counter() - llm_start_time:.4f}s")
        """
        # ==================================================================================

        # ==================================================================================
        # [AREA 2: 模拟推理] (测试UI时使用，正式运行时注释掉或删除)
        # ==================================================================================
        llm_start_time = time.perf_counter()
        print(f"Mocking Inference for question: {question}")
        time.sleep(1.5) # 模拟模型思考了1.5秒
        
        llm_message = f"【模拟回复】这是一个测试回复。你刚才问了：'{question}'。真实模型未加载。"
        print(f"LLM Latency: {time.perf_counter() - llm_start_time:.4f}s")
        
        # 模拟获取历史 (在真实代码中 model.chat 会返回 history)
        history = self.history_synchronizer.get_history()
        
        # ==================================================================================

        # 5. 更新历史记录并返回最终结果
        self.history_synchronizer.set_history(history)
        self.history_synchronizer.update({"role": "assistant", "content": llm_message})

        yield self.history_synchronizer.get_chat_history()

    def start_chat(self, video_path, frame_interval, current_history):
        """
        核心流式处理循环：自动解说，"Describe frame" 或 "Any update?"。包含 FPS 计算和模拟推理
        """
        if not video_path:
            raise gr.Error("Please upload a video file.")

        # 1. 加载和预处理视频
        # (此时调用内部的 _load_video)
        pixel_values, original_frames, timestamps = self._load_video(
            video_path, fps=1 / frame_interval
        )

        # 2. 历史记录初始化
        if current_history:
            history = current_history
        else:
            self.history_synchronizer.reset()
            history = self.history_synchronizer.get_chat_history()

        # 3. 初始化 FPS 计算变量
        fps_display = 0.0

        # --- 逐帧循环 ---
        for idx, (frame, original_frame, timestamp) in enumerate(
            zip(pixel_values, original_frames, timestamps)
        ):
            # A. 暂停逻辑
            if not self.pause_event.is_set():
                self.pause_event.wait()
            
            # ==============================================================================
            # [推理区域] 计算推理时间 & 调用模型
            # ==============================================================================
            inference_start = time.perf_counter()

            # [TODO: 如果你希望模型自动解说，请在这里调用模型]
            # -----------------------------------------------------------
            # [AREA 1: 真实模型自动推理] (测试UI时保持注释)
            # -----------------------------------------------------------
            """
            # [TODO: UNCOMMENT] 取消以下注释以启用自动解说
            
            # 1. 构造自动 Prompt (例如: 描述当前帧发生的事)
            #    注意：你需要决定是每帧都问，还是每隔几帧问一次
            current_pixel_values = pixel_values[: idx + 1].to(self.model.device).to(self.model.dtype)
            
            # 构造增量 Prompt (只包含新帧) 或者 全量 Prompt
            # 这里简化为: 告诉模型现在是第几帧，请描述
            auto_prompt = f"Frame{idx+1}: <image>\nDescribe the current scene briefly."
            
            # 2. 调用模型
            response, new_history = self.model.chat(
                self.tokenizer,
                current_pixel_values,
                auto_prompt,
                self.generation_config,
                history=self.history_synchronizer.get_history(),
                return_history=True
            )
            
            # 3. 更新历史 (这样右侧聊天框就会自动跳出解说)
            self.history_synchronizer.set_history(new_history)
            self.history_synchronizer.update({"role": "assistant", "content": response})
            """
            
            # -----------------------------------------------------------
            # [AREA 2: 模拟推理] (仅用于测试 FPS 显示和 UI 流程)
            # -----------------------------------------------------------
            
            # 模拟模型推理耗时 0.5 秒
            time.sleep(0.5) 
            
            # -----------------------------------------------------------

            inference_end = time.perf_counter()
            
            # B. 计算 FPS (推理时间倒数)
            cost_time = inference_end - inference_start
            if cost_time > 0:
                current_fps = 1.0 / cost_time
                # 平滑滤波: 避免 FPS 数字跳动太快看不清
                # 逻辑: 80% 保持旧值, 20% 采纳新值
                fps_display = 0.8 * fps_display + 0.2 * current_fps if idx > 0 else current_fps

            # ==============================================================================

            # 获取最新的聊天记录 (可能是空的，也可能是 generate_answer 更新的，或者是上面自动推理更新的)
            current_chat_history = self.history_synchronizer.get_chat_history()
            
            # Yield 返回所有状态给前端
            # 顺序对应 outputs 列表: [Time, Frame, State, ChatHistory, FPS]
            yield timestamp, original_frame, pixel_values[: idx + 1], current_chat_history, f"{fps_display:.2f}"

            # C. 额外的播放延迟 (可选)
            # 如果你希望在模型推理之外再人为减慢播放速度，可以取消下面的注释
            # time.sleep(frame_interval) 

        # 循环结束，发送最后一帧状态
        yield timestamps[-1], original_frames[-1], pixel_values, self.history_synchronizer.get_chat_history(), f"{fps_display:.2f}"

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
        # 注意：Stop后重置 FPS 显示为 0.00
        return 0, None, None, [], "0.00"

    # --- UI 构建函数 ---

    def create_interface(self):
        """构建 Gradio 界面"""
        with gr.Blocks(title="VideoChat Online Test") as demo:
            # 隐形状态变量
            pixel_values_state = gr.State()

            # 布局
            with gr.Row():
                # --- 左侧：视频显示区 ---
                with gr.Column(scale=3):
                    gr_frame_display = gr.Image(
                        label="Current Frame (Model Input)", interactive=False, height=400
                    )
                    
                    # 信息显示行：时间 + FPS
                    with gr.Row():
                        gr_time_display = gr.Number(label="Video Time (s)", value=0)
                        # [NEW] FPS 显示组件
                        gr_fps_display = gr.Textbox(label="Inference FPS", value="0.00")
                    
                    with gr.Row():
                        gr_pause_button = gr.Button("Pause Video")
                        gr_stop_button = gr.Button("Stop Video", variant="stop")

                # --- 右侧：聊天区 ---
                with gr.Column(scale=2):
                    gr_chat_interface = gr.Chatbot(
                        label="Chat History", 
                        height=500
                    )
                    gr_question_input = gr.Textbox(
                        label="Manual Question (Ask anytime)"
                    )
            
            # --- 底部：控制区 ---
            with gr.Row():
                with gr.Column():
                    gr_video_upload = gr.Video(label="1. Upload Video")
                with gr.Column():
                    gr_frame_interval = gr.Slider(
                        minimum=0.1, maximum=2.0, step=0.1, value=0.5,
                        interactive=True, label="2. Frame Interval (Sampling Step)",
                    )
                    gr_start_button = gr.Button("3. Start Online Inference", variant="primary")

            # --- 事件绑定 ---

            # 1. 提问
            gr_question_input.submit(
                self.generate_answer,
                inputs=[gr_question_input, pixel_values_state],
                outputs=gr_chat_interface,
                queue=True,
            )

            # 2. 开始 (这里绑定了最多的输出，包括 FPS)
            gr_start_button.click(
                self.start_chat,
                inputs=[
                    gr_video_upload,
                    gr_frame_interval,
                    gr_chat_interface,
                ],
                outputs=[
                    gr_time_display,
                    gr_frame_display,
                    pixel_values_state,
                    gr_chat_interface,
                    gr_fps_display, # 更新 FPS
                ],
            )

            # 3. 暂停
            gr_pause_button.click(
                self.toggle_pause,
                inputs=[],
                outputs=[gr_pause_button, gr_chat_interface]
            )

            # 4. 停止
            gr_stop_button.click(
                self.stop_chat,
                inputs=[],
                outputs=[
                    gr_time_display,
                    gr_frame_display,
                    pixel_values_state,
                    gr_chat_interface,
                    gr_fps_display, # reset FPS
                ],
            )
        
        return demo


if __name__ == "__main__":

    web_ui = VideoChatWebUI()
    demo = web_ui.create_interface()
    print("Launching WebUI with public link...")
    # startup：（for autodl local port：6006）
    demo.launch(server_name="0.0.0.0", 
        server_port=6006, 
        share=False, 
        debug=True)
    print("WebUI will be available at: http://0.0.0.0:6006, please visit autodl 自定义服务详情")