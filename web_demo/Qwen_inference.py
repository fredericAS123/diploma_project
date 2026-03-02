"""
QwenInferenceWrapper — Web Demo 后端推理引擎封装

连接 temporal_encoding.model 流式推理引擎与 Gradio Web 前端。
已适配新的 API（无 manual_time，使用 append_video_chunk / ask / ask_choice）。
"""
import os
import sys
import time
import threading
import torch
from transformers import AutoProcessor

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_CURRENT_DIR, os.pardir))
_TEMPORAL_DIR = os.path.join(_PROJECT_ROOT, "temporal_encoding")
if _TEMPORAL_DIR not in sys.path:
    sys.path.insert(0, _TEMPORAL_DIR)

from model import StreamQwenModel, VideoStreamingInference, EvictionConfig


class QwenInferenceWrapper:
    """Streaming backend adapter for Qwen2.5-VL with automatic mRoPE position tracking."""

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
        """重置流式引擎（新视频开始时调用）。"""
        with self.inference_lock:
            self.engine.reset()

    def reset_with_eviction(
        self,
        enable_eviction: bool = False,
        max_cache_tokens: int = 150000,
        use_mid_anchors: bool = False,
        mid_retention_ratio: float = 0.2,
    ):
        """重置并以新的淘汰配置重建流式引擎。

        Args:
            enable_eviction: 是否启用 KV Cache 淘汰
            max_cache_tokens: 最大 cache token 数（淘汰阈值）
            use_mid_anchors: 是否启用中段锚点 (Level-2)
            mid_retention_ratio: Level-2 中段保留比例
        """
        with self.inference_lock:
            eviction_config = None
            if enable_eviction:
                # 默认采用 Level-1（sink+tail-window+边界保护），
                # 中段锚点作为可选增强开关，按任务场景决定是否启用。
                max_tok = int(max_cache_tokens)
                eviction_config = EvictionConfig(
                    max_cache_tokens=max_tok,
                    sink_size=0,       # auto-detect from first chunk
                    # 默认全部预算给尾窗；开启中段锚点时会改为保守尾窗比例
                    window_size=0 if not use_mid_anchors else int(max_tok * 0.65),
                    enable_temporal_sampling=bool(use_mid_anchors),
                    mid_retention_ratio=float(mid_retention_ratio),
                    # Level-1 边界连续性保护（即使回退到 Level-1 也更稳）
                    enable_sink_boundary_guard=True,
                    sink_boundary_guard_tokens=0,
                )
            self.engine = VideoStreamingInference(
                self.model, self.processor, device=self.device,
                eviction_config=eviction_config,
            )

    def process_frame(self, image, **kwargs) -> str:
        """
        编码单帧到持久 KV Cache。

        Args:
            image: PIL Image
            **kwargs: 兼容旧接口（忽略 manual_time 等废弃参数）
        """
        with self.inference_lock:
            return self.engine.append_frame(image)

    def process_video_chunk(self, frames, fps: float = 2.0) -> str:
        """
        编码多帧视频 chunk 到持久 KV Cache（推荐方式）。

        Args:
            frames: PIL Image 列表
            fps: 帧率
        """
        with self.inference_lock:
            return self.engine.append_video_chunk(frames, fps=fps)

    def ask_question(
        self,
        question: str,
        max_new_tokens: int = 256,
        min_new_tokens: int = 1,
        do_sample: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.9,
        update_state: bool = False,
        return_metrics: bool = False,
        **kwargs,
    ):
        """
        基于已缓存的视频记忆回答问题。

        Args:
            **kwargs: 兼容旧接口（忽略 manual_time 等废弃参数）
        """
        with self.inference_lock:
            response, metrics = self.engine.ask(
                question,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                update_state=update_state,
            )
        if return_metrics:
            return response, metrics
        return response

    def ask_question_stream(
        self,
        question: str,
        max_new_tokens: int = 256,
        min_new_tokens: int = 1,
        do_sample: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.9,
        update_state: bool = False,
    ):
        """流式问答：逐 token 返回增量文本。"""
        with self.inference_lock:
            for event in self.engine.ask_stream(
                question,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                update_state=update_state,
            ):
                yield event

    def ask_choice(self, question: str, choices: list[str]) -> str:
        """多选题评分，返回得分最高的选项。"""
        with self.inference_lock:
            return self.engine.ask_choice(question, choices)

    def get_cache_info(self) -> dict:
        """获取当前 KV Cache 状态信息。"""
        return self.engine.get_cache_info()

    def get_memory_breakdown(self, cache_info: dict | None = None) -> dict:
        """返回显存拆分估算（GB）：模型、KV、推理临时、总已分配/保留。"""
        if cache_info is None:
            cache_info = self.get_cache_info()

        if not torch.cuda.is_available():
            return {
                "model_gb": 0.0,
                "kv_cache_gb": float(cache_info.get("cache_memory_gb", 0.0)),
                "runtime_gb": 0.0,
                "allocated_gb": 0.0,
                "reserved_gb": 0.0,
            }

        allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved_gb = torch.cuda.memory_reserved() / (1024 ** 3)
        kv_gb = float(cache_info.get("cache_memory_gb", 0.0))

        # 参数+buffer 的显存占用估算
        ptr_seen = set()
        model_bytes = 0
        for t in list(self.model.parameters()) + list(self.model.buffers()):
            if not t.is_cuda:
                continue
            ptr = int(t.data_ptr())
            if ptr in ptr_seen:
                continue
            ptr_seen.add(ptr)
            model_bytes += t.nelement() * t.element_size()
        model_gb = model_bytes / (1024 ** 3)

        runtime_gb = max(0.0, allocated_gb - model_gb - kv_gb)
        return {
            "model_gb": round(model_gb, 3),
            "kv_cache_gb": round(kv_gb, 3),
            "runtime_gb": round(runtime_gb, 3),
            "allocated_gb": round(allocated_gb, 3),
            "reserved_gb": round(reserved_gb, 3),
        }

    def native_video_inference(
        self,
        frames,
        question: str,
        fps: float = 2.0,
        max_new_tokens: int = 256,
        min_new_tokens: int = 1,
        do_sample: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        """
        原生视频推理：一次性将所有帧+问题送入模型（非流式）。
        用于与流式推理进行 TTFT / VRAM 对比。

        Returns:
            (response_text, metrics_dict)
        """
        with self.inference_lock:
            # 保存当前流式状态（推理后恢复，避免破坏 streaming 引擎）
            saved_state = None
            if hasattr(self.model, 'stream_state'):
                saved_state = self.model.stream_state
            if hasattr(self.model, 'reset_stream_state'):
                self.model.reset_stream_state()

            # 重置 VRAM 峰值统计
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()

            system_prompt = (
                "You are a concise video analyst. Answer briefly and directly. "
                "Focus on visible facts only. Avoid speculation, avoid repetition. "
                "Strictly limit the response to at most 60 tokens."
            )

            messages = [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {"role": "user", "content": [
                    {"type": "video", "video": frames},
                    {"type": "text", "text": question},
                ]}
            ]

            text_prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            inputs = self.processor(
                text=[text_prompt],
                videos=[frames],
                padding=True,
                return_tensors="pt",
                videos_kwargs={"fps": fps},
            ).to(self.device)

            input_token_count = inputs.input_ids.shape[1]

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_start = time.perf_counter()

            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "min_new_tokens": min_new_tokens,
                "do_sample": do_sample,
            }
            if do_sample:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = top_p

            output_ids = self.model.generate(**inputs, **gen_kwargs)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_end = time.perf_counter()

            generated_ids = output_ids[:, input_token_count:]
            response = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]

            vram_peak_gb = 0.0
            if torch.cuda.is_available():
                vram_peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

            total_latency = t_end - t_start
            metrics = {
                "ttft": total_latency,
                "total_latency": total_latency,
                "vram_peak_gb": round(vram_peak_gb, 3),
                "num_frames": len(frames),
                "input_tokens": input_token_count,
                "output_tokens": generated_ids.shape[1],
            }

            print(
                f"📊 [Native] Frames={len(frames)}, InputTok={input_token_count}, "
                f"Latency={total_latency:.2f}s, VRAM Peak={vram_peak_gb:.2f}GB"
            )

            # 清理临时张量
            del inputs, output_ids, generated_ids
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 恢复流式状态
            if saved_state is not None:
                self.model.stream_state = saved_state
            elif hasattr(self.model, 'reset_stream_state'):
                self.model.reset_stream_state()

            return response, metrics

    def single_frame_inference(
        self,
        image,
        question: str,
        max_new_tokens: int = 256,
        min_new_tokens: int = 1,
        do_sample: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        """
        单帧图片推理：仅传入一帧图片 + 问题（image 模态）。
        用于对比「视频理解 vs 图片理解」以及「流式/原生 vs 单帧」的差异。

        Returns:
            (response_text, metrics_dict)
        """
        with self.inference_lock:
            # 保存当前流式状态
            saved_state = None
            if hasattr(self.model, 'stream_state'):
                saved_state = self.model.stream_state
            if hasattr(self.model, 'reset_stream_state'):
                self.model.reset_stream_state()

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()

            system_prompt = (
                "You are a concise video analyst. Answer briefly and directly. "
                "Focus on visible facts only. Avoid speculation, avoid repetition. "
                "Strictly limit the response to at most 60 tokens."
            )

            messages = [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ]}
            ]

            text_prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            inputs = self.processor(
                text=[text_prompt],
                images=[image],
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            input_token_count = inputs.input_ids.shape[1]

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_start = time.perf_counter()

            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "min_new_tokens": min_new_tokens,
                "do_sample": do_sample,
            }
            if do_sample:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = top_p

            output_ids = self.model.generate(**inputs, **gen_kwargs)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_end = time.perf_counter()

            generated_ids = output_ids[:, input_token_count:]
            response = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]

            vram_peak_gb = 0.0
            if torch.cuda.is_available():
                vram_peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

            total_latency = t_end - t_start
            metrics = {
                "ttft": total_latency,
                "total_latency": total_latency,
                "vram_peak_gb": round(vram_peak_gb, 3),
                "input_tokens": input_token_count,
                "output_tokens": generated_ids.shape[1],
            }

            print(
                f"📊 [Single-Frame] InputTok={input_token_count}, "
                f"Latency={total_latency:.2f}s, VRAM Peak={vram_peak_gb:.2f}GB"
            )

            del inputs, output_ids, generated_ids
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 恢复流式状态
            if saved_state is not None:
                self.model.stream_state = saved_state
            elif hasattr(self.model, 'reset_stream_state'):
                self.model.reset_stream_state()

            return response, metrics
