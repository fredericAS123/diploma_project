import os
import sys
import threading
import torch
from transformers import AutoProcessor

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_CURRENT_DIR, os.pardir))
_TEMPORAL_DIR = os.path.join(_PROJECT_ROOT, "temporal_encoding")
if _TEMPORAL_DIR not in sys.path:
    sys.path.insert(0, _TEMPORAL_DIR)

from temporal_encoding.model.stream_qwen_model import StreamQwenModel
from temporal_encoding.model.video_stream_inference import VideoStreamingInference


class QwenInferenceWrapper:
    """Streaming backend adapter for Qwen2.5-VL with mRoPE absolute time."""

    def __init__(self, model_path: str = "/root/autodl-tmp/Qwen/Qwen2___5-VL-3B-Instruct", device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.inference_lock = threading.Lock()

        print(f"Loading StreamQwenModel from: {model_path} ...")

        if self.device.startswith("cuda"):
            self.model = StreamQwenModel.from_pretrained(
                model_path,
                torch_dtype="auto",
                device_map="auto",
            )
        else:
            self.model = StreamQwenModel.from_pretrained(
                model_path,
                torch_dtype="auto",
            ).to(self.device)

        self.processor = AutoProcessor.from_pretrained(model_path)
        self.engine = VideoStreamingInference(self.model, self.processor, device=self.device)
        print("✅ StreamQwenModel + VideoStreamingInference ready.")

    def log_vram(self, tag: str = ""):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            print(f"📊 [VRAM-{tag}] Alloc: {allocated:.2f}GB | Reserved: {reserved:.2f}GB")

    def reset(self):
        self.engine.reset()

    def process_frame(self, image, manual_time: int | float | None = None) -> str:
        """Encode a single frame into the persistent KV cache."""
        with self.inference_lock:
            return self.engine.append_frame(image, manual_time=manual_time)

    def ask_question(
        self,
        question: str,
        manual_time: int | float | None = None,
        max_new_tokens: int = 256,
        min_new_tokens: int = 1,
        do_sample: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.9,
        return_metrics: bool = False,
    ):
        """Ask a question based on the cached video memory."""
        with self.inference_lock:
            response, metrics = self.engine.ask(
                question,
                manual_time=manual_time,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                update_state=False,
            )
        if return_metrics:
            return response, metrics
        return response