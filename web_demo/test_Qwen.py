import torch
import numpy as np
import threading
from PIL import Image

# --- 引入必要的库 ---
try:
    from decord import VideoReader, cpu
except ImportError:
    print("Error: 'decord' not found. Please run `pip install decord`")

try:
    from modelscope import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
except ImportError:
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        from qwen_vl_utils import process_vision_info
    except ImportError:
        print("Error: libs not found. Please run: pip install modelscope qwen-vl-utils transformers")

class QwenInferenceWrapper:
    def __init__(self, model_path="Qwen/Qwen2.5-VL-3B-Instruct"):
        self.device = "cude" if torch.cuda.is_available() else "cpu"
        self.inference_lock = threading.Lock()  # 互斥锁：确保同一时间只有一个推理在跑
        self.model = None
        self.processor = None
        print(f"Loading Qwen2.5-VL model from : {model_path} ...")
        self.log_vram("Before Load")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype="auto",
                device_map="auto",
            )
        self.processor = AutoProcessor.from_pretrained(model_path)
        print("Model loaded successfully!")
        self.log_vram("After Load")

    def log_vram(self, tag=""):
        """打印当前显存占用"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            print(f"📊 [VRAM-{tag}] Alloc: {allocated:.2f}GB | Reserved: {reserved:.2f}GB")

    def load_video(self, video_path, fps=1):
        """加载视频并抽帧"""
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

    def predict(self, pil_frame, prompt_text, use_system_prompt=True):
        if self.model is None:
            return "Model not loaded."
        with self.inference_lock:
            content_list = []
            content_list.append({"type": "text", "text": prompt_text})
            content_list.append({"type": "image", "image": pil_frame})
            SYSTEM_PROMPT = (
                "You are an intelligent video stream analyst. Your job is to describe visual changes and contents in real-time. "
                "For each input frame, assess the visual information based on these priorities:\n"
                
                "1. **DETECTION (High Priority):** If ANY object (person, car, animal) or dynamic element (smoke, fire, water flow, light change) moves or appears, DESCRIBE it immediately. "
                "Do NOT wait for a 'meaningful event'. Simple movement or presence is enough.\n"
                
                "2. **SCENE CHANGE (Medium Priority):** If the camera angle moves (pans/tilts) or the scene transitions to a new location, briefly describe the new view (e.g., 'Camera pans to a street corner').\n"
                
                "3. **IGNORE (Low Priority):** Output EXACTLY '[WAIT]' ONLY if:\n"
                "   - The scene is completely static (no pixel changes).\n"
                "   - The image is blurry, black, or purely redundant background (e.g., just sky, just grass) with NO objects.\n"
                "   - You have just described this exact scene in the previous turn and nothing new has happened.\n"
                
                "**Output Format:**\n"
                "- Keep descriptions short (10-20 words).\n"
                "- Focus on verbs and nouns (e.g., 'A person walks into frame', 'Red car turns left', 'Camera tilts up to the sky').\n"
                "- NO intro/outro like 'I see...' or 'The image shows...'. Direct observation only."
            )
            if use_system_prompt:
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": content_list}  
                ]
            else:
                messages = [
                    {"role": "user", "content": content_list}  
                ]
            text_input = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            images_input, video_input = process_vision_info(messages)
            inputs = self.processor(
                text=text_input,
                images=images_input,
                videos=video_input,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            generate_ids = self.model.generate(
                **inputs,
                max_new_tokens=128,
                top_p = 0.9,
                temperature=0.1
            )
            generate_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generate_ids)]
            # generate_trimmed = generate_ids[ : , inputs.input_ids.shape[1]: ] 批量处理，当各批次输入长度不同时报错。
            output_text = self.processor.batch_decode(generate_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            return output_text[0] #batch size =1时也需要返回第0个元素，因为output_text是个list