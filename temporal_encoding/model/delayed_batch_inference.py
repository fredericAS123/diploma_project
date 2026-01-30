"""
Delayed Batch Inference Engine - 新方案核心实现

核心思想：
1. 流式收集帧（add_frame）
2. 延迟到提问时才批量编码（ask）
3. 使用 video 模式享受 temporal merge 压缩
4. Vision Encoder 能看到所有帧，建立完整的跨帧注意力

优势：
- ✅ Vision Encoder 跨帧注意力：完整
- ✅ KV Cache 完整性：完整（不丢失历史）
- ✅ 实现难度：中等
- ✅ 显存控制：通过智能帧管理
"""

import torch
from transformers import AutoProcessor
import gc
import time
from typing import List, Dict, Optional, Tuple
from PIL import Image

from .smart_frame_manager import SmartFrameManager


class DelayedBatchInferenceEngine:
    """延迟批量编码推理引擎"""
    
    def __init__(
        self,
        model,
        processor,
        device: str = "cuda",
        star_memory_size: int = 20,
        stream_window_size: int = 20,
        max_pixels: int = 4 * 224 * 224,  # 借鉴Flash-VStream的低分辨率策略
        min_pixels: int = 4 * 28 * 28,
    ):
        """
        Args:
            model: Qwen2.5-VL 模型
            processor: 对应的 processor
            device: 设备
            star_memory_size: Star Memory 容量
            stream_window_size: Stream Memory 窗口大小
            max_pixels: 最大像素（用于 video 模式）
            min_pixels: 最小像素
        """
        self.model = model
        self.processor = processor
        self.device = device
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        
        # 智能帧管理器
        self.frame_manager = SmartFrameManager(
            star_memory_size=star_memory_size,
            stream_window_size=stream_window_size,
        )
        
        # KV Cache（在提问时生成）
        self.video_cache = None
        self.cache_is_valid = False
        
        # 记录模型数据类型
        try:
            self.model_dtype = next(model.parameters()).dtype
        except StopIteration:
            self.model_dtype = torch.float32
        
        # 统计信息
        self.encode_count = 0
        self.total_frames_processed = 0
        
        print(f"✅ DelayedBatchInferenceEngine Initialized")
        print(f"   📐 Max Pixels: {max_pixels} ({int(max_pixels**0.5)}x{int(max_pixels**0.5)})")
        print(f"   🎬 Strategy: 流式收集 + 延迟批量编码")
    
    def add_frame(self, frame: Image.Image, timestamp: float = None) -> str:
        """
        添加新帧（流式收集）
        
        Args:
            frame: PIL Image
            timestamp: 时间戳（秒）
        
        Returns:
            状态信息
        """
        if timestamp is None:
            timestamp = time.time()
        
        result = self.frame_manager.add_frame(frame, timestamp)
        self.total_frames_processed += 1
        
        # 标记 cache 失效（有新帧加入）
        self.cache_is_valid = False
        
        # 构建状态消息
        status = f"Frame #{self.total_frames_processed} added to Stream Memory"
        if result['added_to_star']:
            status += f" + Star Memory ({result['reason']})"
        
        status += f" | Star: {result['star_count']}, Stream: {result['stream_count']}"
        
        return status
    
    def _encode_all_frames(self) -> Dict[str, any]:
        """
        批量编码所有帧（内部方法，在提问时调用）
        
        使用 video 模式：
        - Vision Encoder 能看到所有帧
        - 享受 temporal merge 的压缩
        - 建立完整的跨帧注意力
        """
        # 获取所有需要编码的帧
        all_frames, metadata = self.frame_manager.get_all_frames()
        
        if not all_frames:
            return {'success': False, 'reason': 'no frames'}
        
        print(f"\n🔄 [编码 #{self.encode_count + 1}] 批量编码 {metadata['unique_frames']} 帧")
        print(f"   📊 Star: {metadata['star_frames']}, Stream: {metadata['stream_frames']}, "
              f"重叠: {metadata['overlap_frames']}")
        print(f"   📉 压缩比: {metadata['compression_ratio']:.2f}x "
              f"(从 {metadata['total_added']} 帧压缩到 {metadata['unique_frames']} 帧)")
        
        # 构建消息（使用 video 模式）
        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": all_frames,
                    "min_pixels": self.min_pixels,
                    "max_pixels": self.max_pixels,
                },
                {"type": "text", "text": "Watch this video stream."},
            ],
        }]
        
        # 应用 chat template
        text_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        
        # 处理输入
        t_start = time.time()
        inputs = self.processor(
            text=[text_prompt],
            videos=[all_frames],
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        
        # 转换数据类型
        if "pixel_values_videos" in inputs and inputs["pixel_values_videos"] is not None:
            inputs["pixel_values_videos"] = inputs["pixel_values_videos"].to(
                device=self.device,
                dtype=self.model_dtype,
            )
        
        # 编码（生成 KV Cache）
        with torch.inference_mode():
            outputs = self.model(
                **{k: v for k, v in inputs.items() if k not in ["attention_mask"]},
                attention_mask=inputs.get("attention_mask", None),
                past_key_values=None,  # 从零开始
                use_cache=True,
                output_hidden_states=False,
                logits_to_keep=1,
            )
        
        # 保存 KV Cache
        self.video_cache = self._detach_past(outputs.past_key_values)
        self.cache_is_valid = True
        self.encode_count += 1
        
        t_end = time.time()
        
        # 提取视觉 token 数量
        visual_tokens = self._extract_visual_tokens_from_inputs(inputs)
        cache_length = self._get_past_len(self.video_cache)
        
        print(f"   ✅ 编码完成！")
        print(f"   ⏱️  耗时: {t_end - t_start:.2f}s")
        print(f"   🎯 Visual Tokens: {visual_tokens}")
        print(f"   💾 KV Cache Length: {cache_length}")
        print(f"   📐 video_grid_thw: {inputs.get('video_grid_thw')}")
        
        return {
            'success': True,
            'frames_encoded': metadata['unique_frames'],
            'visual_tokens': visual_tokens,
            'cache_length': cache_length,
            'encoding_time': t_end - t_start,
            'video_grid_thw': inputs.get('video_grid_thw'),
        }
    
    def ask(
        self,
        question: str,
        max_new_tokens: int = 512,
        min_new_tokens: int = 1,
        do_sample: bool = True,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
    ) -> Tuple[str, Dict]:
        """
        提问（如果 cache 失效，会自动重新编码所有帧）
        
        Args:
            question: 问题文本
            max_new_tokens: 最大生成 token 数
            其他参数: 生成参数
        
        Returns:
            (answer, metrics)
        """
        t_total_start = time.time()
        
        # 1. 如果 cache 失效，重新编码所有帧
        encode_result = None
        if not self.cache_is_valid:
            encode_result = self._encode_all_frames()
            if not encode_result['success']:
                return f"Error: {encode_result['reason']}", {}
        
        # 2. 编码问题
        question_prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        question_inputs = self.processor.tokenizer(
            question_prompt,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(self.device)
        
        # 3. 构建完整的 attention mask
        past_len = self._get_past_len(self.video_cache)
        full_mask = self._build_full_attention_mask(question_inputs.attention_mask, past_len)
        
        cache_position = None
        if past_len and past_len > 0:
            cache_position = torch.arange(
                past_len,
                past_len + question_inputs.input_ids.shape[1],
                device=question_inputs.input_ids.device,
            )
        
        # 4. 生成回答
        t_gen_start = time.time()
        with torch.inference_mode():
            generated_ids = self.model.generate(
                input_ids=question_inputs.input_ids,
                attention_mask=full_mask,
                past_key_values=self.video_cache,
                cache_position=cache_position,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                use_cache=True,
            )
        t_gen_end = time.time()
        
        # 5. 解码输出
        output_ids = generated_ids[0, question_inputs.input_ids.shape[1]:]
        answer = self.processor.tokenizer.decode(output_ids, skip_special_tokens=True)
        
        t_total_end = time.time()
        
        # 6. 统计信息
        metrics = {
            'total_latency': t_total_end - t_total_start,
            'generation_latency': t_gen_end - t_gen_start,
            'output_tokens': len(output_ids),
        }
        
        if encode_result:
            metrics['encoding_latency'] = encode_result['encoding_time']
            metrics['frames_encoded'] = encode_result['frames_encoded']
            metrics['visual_tokens'] = encode_result['visual_tokens']
        
        return answer, metrics
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        frame_stats = self.frame_manager.get_statistics()
        
        return {
            'total_frames_added': self.total_frames_processed,
            'encode_count': self.encode_count,
            'cache_valid': self.cache_is_valid,
            **frame_stats,
        }
    
    def reset(self):
        """重置引擎"""
        self.frame_manager.reset()
        self.video_cache = None
        self.cache_is_valid = False
        self.encode_count = 0
        self.total_frames_processed = 0
        
        gc.collect()
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
        
        print("🔄 DelayedBatchInferenceEngine Reset.")
    
    # ========== 辅助方法 ==========
    
    def _detach_past(self, past_key_values):
        """分离 past_key_values 从计算图"""
        if past_key_values is None:
            return None
        if hasattr(past_key_values, "get_seq_length"):
            return past_key_values
        return tuple(tuple(p.detach() for p in layer) for layer in past_key_values)
    
    def _get_past_len(self, past_key_values):
        """获取 past_key_values 的长度"""
        if past_key_values is None:
            return 0
        if hasattr(past_key_values, "get_seq_length"):
            return past_key_values.get_seq_length()
        return past_key_values[0][0].shape[-2]
    
    def _build_full_attention_mask(self, attention_mask, past_len):
        """构建完整的 attention mask（包括 past）"""
        if past_len is None or past_len == 0:
            return attention_mask
        past_mask = torch.ones(
            (attention_mask.shape[0], past_len),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        return torch.cat([past_mask, attention_mask], dim=1)
    
    def _extract_visual_tokens_from_inputs(self, inputs):
        """从 processor 输出中提取 visual tokens 的数量"""
        if "video_grid_thw" not in inputs:
            return 0
        
        video_grid_thw = inputs["video_grid_thw"]
        merge_length = getattr(self.processor, "merge_size", 2) ** 2
        
        # video_grid_thw shape: (num_videos, 3) -> (T, H, W)
        num_video_tokens = (video_grid_thw[0].prod() // merge_length).item() if len(video_grid_thw) > 0 else 0
        return num_video_tokens
