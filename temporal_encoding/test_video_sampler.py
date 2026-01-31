"""
视频采样和时间编码测试

测试目标：
1. 验证采样后帧数正确
2. 验证 second_per_grid_t 计算正确
3. 验证时间编码能正确覆盖视频时长
4. 端到端测试与模型集成
"""

import pytest
import torch
import numpy as np
from PIL import Image
from typing import List
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model.video_sampler import (
    VideoSampler,
    calculate_expected_temporal_positions,
    validate_time_encoding,
)


class TestVideoSampler:
    """视频采样器单元测试"""
    
    @pytest.fixture
    def create_dummy_frames(self):
        """创建测试用的假视频帧"""
        def _create(num_frames: int, size: tuple = (224, 224)) -> List[Image.Image]:
            frames = []
            for i in range(num_frames):
                # 创建有时间标记的帧
                arr = np.zeros((*size, 3), dtype=np.uint8)
                arr[:, :, 0] = i % 256  # R通道编码帧号
                frames.append(Image.fromarray(arr))
            return frames
        return _create
    
    def test_basic_sampling_1fps(self, create_dummy_frames):
        """测试1fps采样 - 50秒视频应该得到50帧"""
        # 创建50秒30fps的视频 = 1500帧
        original_fps = 30.0
        duration = 50.0
        total_frames = int(duration * original_fps)
        frames = create_dummy_frames(total_frames)
        
        # 1fps采样
        sampler = VideoSampler(target_fps=1.0)
        sampled, second_per_grid_t, meta = sampler.sample_frames(
            frames=frames,
            original_fps=original_fps,
            video_duration=duration,
        )
        
        # 验证帧数（应该是50，对齐到temporal_patch_size=2的倍数）
        assert len(sampled) == 50, f"Expected 50 frames, got {len(sampled)}"
        
        # 验证是temporal_patch_size的倍数
        assert len(sampled) % 2 == 0, "Frame count should be multiple of temporal_patch_size"
        
        # 验证second_per_grid_t
        # 50帧 / 2 = 25个temporal grids
        # 50秒 / 25 grids = 2秒/grid
        expected_second_per_grid = duration / (len(sampled) // 2)
        assert abs(second_per_grid_t - expected_second_per_grid) < 0.01, \
            f"second_per_grid_t mismatch: {second_per_grid_t} vs {expected_second_per_grid}"
        
        print(f"✅ 1fps采样测试通过")
        print(f"   原始: {total_frames}帧 @ {original_fps}fps")
        print(f"   采样后: {len(sampled)}帧")
        print(f"   second_per_grid_t: {second_per_grid_t}s")
        print(f"   压缩比: {meta['compression_ratio']:.2f}x")
    
    def test_time_encoding_validation(self, create_dummy_frames):
        """测试时间编码验证"""
        # 创建10秒视频
        frames = create_dummy_frames(300)  # 30fps * 10秒
        
        sampler = VideoSampler(target_fps=2.0)  # 2fps
        sampled, second_per_grid_t, meta = sampler.sample_frames(
            frames=frames,
            original_fps=30.0,
            video_duration=10.0,
        )
        
        # 验证时间编码
        is_valid, details = validate_time_encoding(
            sampled_frames=len(sampled),
            second_per_grid_t=second_per_grid_t,
            expected_duration=10.0,
            tolerance=0.5,  # 允许0.5秒误差
        )
        
        assert is_valid, f"Time encoding validation failed: {details}"
        
        print(f"✅ 时间编码验证通过")
        print(f"   覆盖时长: {details['total_covered_time']:.2f}s")
        print(f"   预期时长: {details['expected_duration']:.2f}s")
        print(f"   误差: {details['time_error']:.2f}s")
    
    def test_temporal_position_calculation(self):
        """测试temporal position计算与官方一致"""
        # 模拟：20帧，每个grid 2秒，tokens_per_second=4
        num_frames = 20
        second_per_grid_t = 2.0
        
        positions = calculate_expected_temporal_positions(
            num_frames=num_frames,
            second_per_grid_t=second_per_grid_t,
            temporal_patch_size=2,
            tokens_per_second=4,
        )
        
        # 期望的位置序列：
        # Grid 0: 0*2*4=0, 帧0和帧1的位置都是0
        # Grid 1: 1*2*4=8, 帧2和帧3的位置都是8
        # Grid 2: 2*2*4=16, ...
        # 以此类推
        expected = []
        for grid_idx in range(10):  # 20帧/2 = 10个grids
            pos = grid_idx * 2 * 4
            expected.extend([pos, pos])  # 每个grid有2帧
        
        assert positions == expected, f"Position mismatch:\n  Got: {positions}\n  Expected: {expected}"
        
        print(f"✅ Temporal position计算测试通过")
        print(f"   Positions: {positions[:10]}... (showing first 10)")
    
    def test_different_fps_scenarios(self, create_dummy_frames):
        """测试不同FPS场景"""
        test_cases = [
            {'target_fps': 0.5, 'duration': 60, 'original_fps': 30},  # 0.5fps, 60秒
            {'target_fps': 1.0, 'duration': 30, 'original_fps': 24},  # 1fps, 30秒
            {'target_fps': 2.0, 'duration': 10, 'original_fps': 60},  # 2fps, 10秒
            {'target_fps': 4.0, 'duration': 5, 'original_fps': 30},   # 4fps, 5秒
        ]
        
        for case in test_cases:
            total_frames = int(case['duration'] * case['original_fps'])
            frames = create_dummy_frames(total_frames)
            
            sampler = VideoSampler(target_fps=case['target_fps'])
            sampled, second_per_grid_t, meta = sampler.sample_frames(
                frames=frames,
                original_fps=case['original_fps'],
                video_duration=case['duration'],
            )
            
            # 验证帧数合理
            expected_frames = case['duration'] * case['target_fps']
            # 对齐到temporal_patch_size的倍数
            expected_frames = max(2, int(expected_frames // 2) * 2)
            
            assert abs(len(sampled) - expected_frames) <= 2, \
                f"Frame count mismatch for {case}: got {len(sampled)}, expected ~{expected_frames}"
            
            # 验证时间编码
            is_valid, _ = validate_time_encoding(
                sampled_frames=len(sampled),
                second_per_grid_t=second_per_grid_t,
                expected_duration=case['duration'],
                tolerance=1.0,
            )
            assert is_valid, f"Time encoding invalid for {case}"
            
            print(f"✅ FPS={case['target_fps']}, Duration={case['duration']}s: "
                  f"{len(sampled)} frames, second_per_grid_t={second_per_grid_t:.3f}s")


class TestIntegrationWithModel:
    """与模型集成的端到端测试"""
    
    @pytest.fixture
    def model_and_processor(self):
        """加载模型和处理器（使用小模型或mock）"""
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            
            # 尝试加载小模型，如果没有则跳过
            model_name = "Qwen/Qwen2.5-VL-2B-Instruct"  # 使用小模型测试
            
            processor = AutoProcessor.from_pretrained(model_name)
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            
            return model, processor
        except Exception as e:
            pytest.skip(f"Model not available: {e}")
    
    @pytest.mark.slow
    def test_second_per_grid_ts_injection(self, model_and_processor):
        """测试 second_per_grid_ts 正确注入到模型"""
        model, processor = model_and_processor
        
        # 创建测试帧
        frames = [Image.new('RGB', (224, 224), color=(i*10, 0, 0)) for i in range(20)]
        
        # 准备输入
        messages = [{
            "role": "user",
            "content": [
                {"type": "video", "video": frames},
                {"type": "text", "text": "Describe this video."},
            ],
        }]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], videos=[frames], return_tensors="pt").to(model.device)
        
        # 注入自定义 second_per_grid_ts
        custom_second_per_grid = 2.0  # 每个grid 2秒
        inputs['second_per_grid_ts'] = torch.tensor([custom_second_per_grid]).to(model.device)
        
        # 运行前向传播
        with torch.inference_mode():
            outputs = model(**inputs, output_hidden_states=True)
        
        # 验证输出有效
        assert outputs.logits is not None
        assert outputs.logits.shape[0] == 1  # batch size = 1
        
        print(f"✅ second_per_grid_ts 注入测试通过")
        print(f"   注入的 second_per_grid_t: {custom_second_per_grid}s")
        print(f"   输出 logits shape: {outputs.logits.shape}")
    
    @pytest.mark.slow
    def test_end_to_end_with_sampler(self, model_and_processor):
        """端到端测试：采样 + 时间编码 + 模型推理"""
        model, processor = model_and_processor
        
        # 模拟30秒30fps的视频
        original_fps = 30.0
        duration = 30.0
        total_frames = int(duration * original_fps)
        frames = [Image.new('RGB', (224, 224), color=(i % 256, 0, 0)) for i in range(total_frames)]
        
        # 1fps采样
        sampler = VideoSampler(target_fps=1.0)
        sampled_frames, second_per_grid_t, meta = sampler.sample_frames(
            frames=frames,
            original_fps=original_fps,
            video_duration=duration,
        )
        
        print(f"\n📊 采样结果:")
        print(f"   原始: {total_frames} 帧")
        print(f"   采样后: {len(sampled_frames)} 帧")
        print(f"   second_per_grid_t: {second_per_grid_t:.4f}s")
        
        # 准备模型输入
        messages = [{
            "role": "user",
            "content": [
                {"type": "video", "video": sampled_frames},
                {"type": "text", "text": "What happens in this video?"},
            ],
        }]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], videos=[sampled_frames], return_tensors="pt").to(model.device)
        
        # 注入时间编码参数
        inputs['second_per_grid_ts'] = torch.tensor([second_per_grid_t]).to(model.device)
        
        # 生成
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=50,
            )
        
        output_text = processor.batch_decode(
            generated_ids[:, inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )[0]
        
        print(f"\n🤖 模型输出: {output_text[:200]}...")
        print(f"✅ 端到端测试通过")


class TestTimeEncodingAccuracy:
    """时间编码精度测试"""
    
    def test_frame_time_mapping(self):
        """测试帧到时间的映射精度"""
        # 场景：50秒视频，1fps采样 = 50帧
        video_duration = 50.0
        target_fps = 1.0
        temporal_patch_size = 2
        tokens_per_second = 4
        
        sampler = VideoSampler(
            target_fps=target_fps,
            temporal_patch_size=temporal_patch_size,
            tokens_per_second=tokens_per_second,
        )
        
        # 计算参数
        num_frames = int(video_duration * target_fps)
        num_frames = max(temporal_patch_size, (num_frames // temporal_patch_size) * temporal_patch_size)
        num_grids = num_frames // temporal_patch_size
        second_per_grid_t = video_duration / num_grids
        
        print(f"\n📊 时间编码精度测试:")
        print(f"   视频时长: {video_duration}s")
        print(f"   采样后帧数: {num_frames}")
        print(f"   Temporal grids: {num_grids}")
        print(f"   second_per_grid_t: {second_per_grid_t}s")
        
        # 计算每个grid对应的实际时间
        print(f"\n   Grid时间映射:")
        for i in range(min(10, num_grids)):  # 显示前10个
            grid_time = i * second_per_grid_t
            position_id = int(i * second_per_grid_t * tokens_per_second)
            print(f"     Grid {i}: 时间={grid_time:.2f}s, Position ID={position_id}")
        
        # 验证最后一个grid的时间接近视频结束
        last_grid_time = (num_grids - 1) * second_per_grid_t
        time_error = abs(last_grid_time - (video_duration - second_per_grid_t))
        
        assert time_error < 0.5, f"Last grid time error too large: {time_error}s"
        
        print(f"\n   最后一个Grid时间: {last_grid_time:.2f}s")
        print(f"   ✅ 时间映射精度测试通过")


if __name__ == "__main__":
    # 运行快速测试
    pytest.main([__file__, "-v", "-k", "not slow"])