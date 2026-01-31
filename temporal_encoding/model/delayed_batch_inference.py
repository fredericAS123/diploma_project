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
from transformers import AutoProcessor, TextIteratorStreamer
import gc
import time
from threading import Thread
from typing import List, Dict, Optional, Tuple, Generator
from PIL import Image

from .smart_frame_manager import SmartFrameManager
from .video_sampler import VideoSampler, validate_time_encoding


class DelayedBatchInferenceEngine:
    """延迟批量编码推理引擎 - 增加动态采样支持"""
    
    def __init__(
        self,
        model,
        processor,
        device: str = "cuda",
        star_memory_size: int = 20,
        stream_window_size: int = 20,
        max_pixels: int = 4 * 224 * 224,  # 借鉴Flash-VStream的低分辨率策略
        min_pixels: int = 4 * 28 * 28,
        # === 新增参数 ===
        target_fps: float = None,  # 目标采样频率，None表示不采样
        enable_absolute_time_encoding: bool = True,  # 是否启用绝对时间编码
        use_disk_cache: bool = True,  # 是否使用硬盘缓存（节省内存）
        max_sampled_frames: int = 48,  # 最大采样帧数（防止OOM，24GB显存建议48）
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
            target_fps: 目标采样频率
            enable_absolute_time_encoding: 是否启用绝对时间编码
            use_disk_cache: 是否使用硬盘缓存（推荐True，大幅节省内存）
            max_sampled_frames: 最大采样帧数（防止OOM）
        """
        self.model = model
        self.processor = processor
        self.device = device
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.target_fps = target_fps
        self.enable_absolute_time_encoding = enable_absolute_time_encoding
        self.use_disk_cache = use_disk_cache
        self.max_sampled_frames = max_sampled_frames
        
        # 智能帧管理器（支持硬盘缓存）
        self.frame_manager = SmartFrameManager(
            star_memory_size=star_memory_size,
            stream_window_size=stream_window_size,
            use_disk_cache=use_disk_cache,
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
        
        # 存储采样元数据
        self.last_sample_metadata = None
        
        # 从processor获取时间编码相关配置
        self.temporal_patch_size = getattr(
            processor, 'temporal_patch_size', 
            getattr(model.config.vision_config, 'temporal_patch_size', 2)
        )
        self.tokens_per_second = getattr(
            model.config.vision_config, 'tokens_per_second', 4
        )
        
        if target_fps is not None:
            self.video_sampler = VideoSampler(
                target_fps=target_fps,
                temporal_patch_size=self.temporal_patch_size,
                tokens_per_second=self.tokens_per_second,
                max_sampled_frames=max_sampled_frames,
            )
            print(f"   🎯 Target FPS: {target_fps}")
            print(f"   📊 Max Sampled Frames: {max_sampled_frames}")
            print(f"   ⏱️  Absolute Time Encoding: {enable_absolute_time_encoding}")
        else:
            self.video_sampler = None
        
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
        
        # 标记 cache 失效（有新帧加入，下次 ask 时需要重新编码）
        self.cache_is_valid = False
        
        # 构建状态消息
        status = f"Frame #{self.total_frames_processed} added to Stream Memory"
        if result['added_to_star']:
            status += f" + Star Memory ({result['reason']})"
        
        status += f" | Star: {result['star_count']}, Stream: {result['stream_count']}"
        
        return status
    
    def _encode_all_frames(self) -> Dict[str, any]:
        """
        批量编码所有帧（修改版 - 支持动态采样和时间编码）
        
        关键修复：使用 sample_from_timestamps 基于真实时间戳采样，
        而非 sample_frames 基于索引采样。这对于 Star+Stream 混合的
        稀疏帧序列至关重要，否则时间编码会完全错乱。
        """
        # ========== 强制释放显存（关键！）==========
        # 必须在编码前彻底释放旧的 KV cache，否则会 OOM
        if self.video_cache is not None:
            # 如果是 DynamicCache 类型，需要特殊处理
            if hasattr(self.video_cache, 'key_cache'):
                for layer_cache in self.video_cache.key_cache:
                    del layer_cache
                for layer_cache in self.video_cache.value_cache:
                    del layer_cache
            del self.video_cache
            self.video_cache = None
        
        # 多次调用 gc 确保彻底释放
        for _ in range(3):
            gc.collect()
        
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # 确保释放完成
            # 打印显存状态（调试用）
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"   🧹 显存释放后: 已分配 {allocated:.2f}GB, 已保留 {reserved:.2f}GB")
        
        # 获取所有需要编码的帧（包含时间戳）
        all_frames, timestamps, metadata = self.frame_manager.get_all_frames()
        
        if not all_frames:
            return {'success': False, 'reason': 'no frames'}
        
        # === 关键修复：使用基于时间戳的采样 ===
        second_per_grid_ts = None
        if self.video_sampler is not None:
            # 使用 sample_from_timestamps 而非 sample_frames
            # 这确保了稀疏帧（如 t=0s 的 Star 帧 + t=50~55s 的 Stream 帧）
            # 能正确反映真实的时间跨度
            all_frames, second_per_grid_t, sample_meta = self.video_sampler.sample_from_timestamps(
                frames=all_frames,
                timestamps=timestamps,
            )
            
            self.last_sample_metadata = sample_meta
            
            if self.enable_absolute_time_encoding:
                # 将 second_per_grid_t 转换为 tensor
                second_per_grid_ts = torch.tensor(
                    [second_per_grid_t], 
                    dtype=torch.float32,
                    device=self.device
                )
            
            print(f"   📹 采样: {sample_meta['original_frames']} → {sample_meta['sampled_frames']} 帧")
            print(f"   ⏱️  second_per_grid_t: {second_per_grid_t:.4f}s")
            print(f"   🕐 实际时间跨度: {sample_meta['video_duration']:.2f}s (从 t={metadata['min_timestamp']:.1f}s 到 t={metadata['max_timestamp']:.1f}s)")
        
        # 记录帧数（后面会删除 all_frames）
        frames_encoded_count = len(all_frames)
        
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
        
        # === 新增：注入 second_per_grid_ts ===
        if second_per_grid_ts is not None:
            inputs['second_per_grid_ts'] = second_per_grid_ts
        
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
        
        # 提取视觉 token 数量（在释放 inputs 之前）
        visual_tokens = self._extract_visual_tokens_from_inputs(inputs)
        cache_length = self._get_past_len(self.video_cache)
        video_grid_thw = inputs.get('video_grid_thw')
        
        # ========== 立即释放中间变量，回收显存 ==========
        del outputs
        del inputs
        del all_frames
        gc.collect()
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
        
        print(f"   ✅ 编码完成！")
        print(f"   ⏱️  耗时: {t_end - t_start:.2f}s")
        print(f"   🎯 Visual Tokens: {visual_tokens}")
        print(f"   💾 KV Cache Length: {cache_length}")
        print(f"   📐 video_grid_thw: {video_grid_thw}")
        
        result = {
            'success': True,
            'frames_encoded': frames_encoded_count,
            'visual_tokens': visual_tokens,
            'cache_length': cache_length,
            'encoding_time': t_end - t_start,
            'video_grid_thw': video_grid_thw,
        }
        
        if self.last_sample_metadata:
            result['sample_metadata'] = self.last_sample_metadata
        
        return result
    
    def _estimate_original_fps(self) -> float:
        """估算原始帧率（基于帧管理器的时间戳）"""
        # 从 SmartFrameManager 获取时间戳信息
        timestamps = []
        for entry in self.frame_manager.star_memory:
            timestamps.append(entry['timestamp'])
        for entry in self.frame_manager.stream_memory:
            timestamps.append(entry['timestamp'])
        
        if len(timestamps) < 2:
            return 30.0  # 默认30fps
        
        timestamps = sorted(timestamps)
        duration = timestamps[-1] - timestamps[0]
        
        if duration > 0:
            return len(timestamps) / duration
        return 30.0
    
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
        output_token_count = len(output_ids)
        
        # ========== 关键：释放 generate 产生的临时变量 ==========
        del generated_ids
        del question_inputs
        del full_mask
        if cache_position is not None:
            del cache_position
        gc.collect()
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
        
        t_total_end = time.time()
        
        # 6. 统计信息
        metrics = {
            'total_latency': t_total_end - t_total_start,
            'generation_latency': t_gen_end - t_gen_start,
            'output_tokens': output_token_count,
        }
        
        if encode_result:
            metrics['encoding_latency'] = encode_result['encoding_time']
            metrics['frames_encoded'] = encode_result['frames_encoded']
            metrics['visual_tokens'] = encode_result['visual_tokens']
        
        return answer, metrics
    
    def ask_stream(
        self,
        question: str,
        max_new_tokens: int = 512,
        min_new_tokens: int = 1,
        do_sample: bool = True,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        timeout: float = 60.0,
    ) -> Generator[str, None, Dict]:
        """
        流式提问（Token Streaming 输出）- 复用官方 TextIteratorStreamer 方案
        
        Args:
            question: 问题文本
            max_new_tokens: 最大生成 token 数
            timeout: streamer 超时时间
            其他参数: 生成参数
        
        Yields:
            str: 逐个生成的文本片段
        
        Returns:
            最后通过 generator.send(None) 或遍历完后可获取 metrics（实际通过属性）
        
        Usage:
            for text in engine.ask_stream(question):
                print(text, end='', flush=True)
            print()  # 换行
            # metrics 可通过 engine.last_stream_metrics 获取
        """
        t_total_start = time.time()
        
        # 1. 如果 cache 失效，重新编码所有帧
        encode_result = None
        if not self.cache_is_valid:
            encode_result = self._encode_all_frames()
            if not encode_result['success']:
                yield f"Error: {encode_result['reason']}"
                return
        
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
        
        # 4. 创建 Streamer（复用官方方案）
        streamer = TextIteratorStreamer(
            self.processor.tokenizer,
            timeout=timeout,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        
        # 5. 构建生成参数
        gen_kwargs = {
            'input_ids': question_inputs.input_ids,
            'attention_mask': full_mask,
            'past_key_values': self.video_cache,
            'cache_position': cache_position,
            'max_new_tokens': max_new_tokens,
            'min_new_tokens': min_new_tokens,
            'do_sample': do_sample,
            'temperature': temperature,
            'top_p': top_p,
            'repetition_penalty': repetition_penalty,
            'pad_token_id': self.processor.tokenizer.pad_token_id,
            'eos_token_id': self.processor.tokenizer.eos_token_id,
            'use_cache': True,
            'streamer': streamer,
        }
        
        # 6. 在单独线程中运行 generate（官方方案）
        t_gen_start = time.time()
        thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()
        
        # 7. 流式输出
        output_tokens = 0
        generated_text = ""
        for new_text in streamer:
            if new_text:
                output_tokens += 1  # 近似计数
                generated_text += new_text
                yield new_text
        
        # 8. 等待生成完成
        thread.join()
        t_gen_end = time.time()
        t_total_end = time.time()
        
        # ========== 关键：释放 generate 产生的临时变量 ==========
        del gen_kwargs
        del question_inputs
        del full_mask
        if cache_position is not None:
            del cache_position
        del streamer
        gc.collect()
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
        
        # 9. 保存 metrics 到实例属性（供调用者获取）
        self.last_stream_metrics = {
            'total_latency': t_total_end - t_total_start,
            'generation_latency': t_gen_end - t_gen_start,
            'output_tokens': output_tokens,
            'generated_text': generated_text,
        }
        
        if encode_result:
            self.last_stream_metrics['encoding_latency'] = encode_result['encoding_time']
            self.last_stream_metrics['frames_encoded'] = encode_result['frames_encoded']
            self.last_stream_metrics['visual_tokens'] = encode_result['visual_tokens']
    
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
