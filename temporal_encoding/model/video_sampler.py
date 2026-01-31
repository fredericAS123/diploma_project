"""
视频采样工具 - 支持动态采样频率和时间戳保持

核心功能：
1. 按指定FPS采样视频帧
2. 计算正确的 second_per_grid_ts 以保持绝对时间编码
3. 与官方 processor 无缝集成
"""

import math
from typing import List, Tuple, Optional, Union
from PIL import Image
import numpy as np


class VideoSampler:
    """视频采样器 - 支持动态采样频率并保持Qwen2.5-VL的绝对时间编码"""
    
    def __init__(
        self,
        target_fps: float = 1.0,
        temporal_patch_size: int = 2,
        tokens_per_second: int = 4,
        max_sampled_frames: Optional[int] = None,  # 限制最大帧数防止OOM，None表示不限制
    ):
        """
        Args:
            target_fps: 目标采样频率 (帧/秒)，例如 1.0 表示每秒采样1帧
            temporal_patch_size: Qwen2.5-VL的temporal patch大小，默认2
            tokens_per_second: Qwen2.5-VL配置中的tokens_per_second，默认4
            max_sampled_frames: 采样后的最大帧数，防止显存溢出（None表示不限制）
        """
        self.target_fps = target_fps
        self.temporal_patch_size = temporal_patch_size
        self.tokens_per_second = tokens_per_second
        self.max_sampled_frames = max_sampled_frames
    
    def sample_frames(
        self,
        frames: List[Image.Image],
        original_fps: float,
        video_duration: Optional[float] = None,
    ) -> Tuple[List[Image.Image], float, dict]:
        """
        按目标FPS采样视频帧，并计算正确的时间编码参数
        
        Args:
            frames: 原始视频帧列表
            original_fps: 原始视频帧率
            video_duration: 视频总时长(秒)，如果不提供则从frames和fps计算
            
        Returns:
            sampled_frames: 采样后的帧列表
            second_per_grid_t: 每个temporal grid对应的秒数（用于时间编码）
            metadata: 采样元数据
        """
        total_frames = len(frames)
        
        # 计算视频总时长
        if video_duration is None:
            video_duration = total_frames / original_fps
        
        # 计算目标帧数
        target_frame_count = max(1, int(video_duration * self.target_fps))
        
        # 确保帧数是 temporal_patch_size 的倍数（Qwen2.5-VL要求）
        target_frame_count = self._align_to_temporal_patch(target_frame_count)
        
        # 限制最大帧数（防止显存溢出）
        if (
            self.max_sampled_frames is not None
            and self.max_sampled_frames > 0
            and target_frame_count > self.max_sampled_frames
        ):
            target_frame_count = self._align_to_temporal_patch(self.max_sampled_frames)
        
        # 计算采样间隔
        if target_frame_count >= total_frames:
            # 如果目标帧数 >= 原始帧数，不采样
            sampled_frames = frames
            actual_fps = original_fps
        else:
            # 均匀采样
            indices = np.linspace(0, total_frames - 1, target_frame_count, dtype=int)
            sampled_frames = [frames[i] for i in indices]
            actual_fps = target_frame_count / video_duration
        
        # 计算 second_per_grid_t
        # 这是关键：每个temporal grid（temporal_patch_size帧）对应的实际秒数
        second_per_grid_t = self._calculate_second_per_grid_t(
            num_sampled_frames=len(sampled_frames),
            video_duration=video_duration,
        )
        
        metadata = {
            'original_frames': total_frames,
            'sampled_frames': len(sampled_frames),
            'original_fps': original_fps,
            'target_fps': self.target_fps,
            'actual_fps': actual_fps,
            'video_duration': video_duration,
            'second_per_grid_t': second_per_grid_t,
            'temporal_grids': len(sampled_frames) // self.temporal_patch_size,
            'compression_ratio': total_frames / len(sampled_frames),
        }
        
        return sampled_frames, second_per_grid_t, metadata
    
    def _align_to_temporal_patch(self, frame_count: int) -> int:
        """将帧数对齐到temporal_patch_size的倍数"""
        return max(
            self.temporal_patch_size,
            (frame_count // self.temporal_patch_size) * self.temporal_patch_size
        )
    
    def _calculate_second_per_grid_t(
        self,
        num_sampled_frames: int,
        video_duration: float,
    ) -> float:
        """
        计算每个temporal grid对应的秒数
        
        关键公式解析（来自官方 get_rope_index）：
        - temporal_position = grid_index * second_per_grid_t * tokens_per_second
        - grid_index 从 0 到 (num_temporal_grids - 1)
        - 我们希望最后一个grid的时间位置接近视频结束
        
        因此：
        - second_per_grid_t = video_duration / num_temporal_grids
        - 其中 num_temporal_grids = num_sampled_frames / temporal_patch_size
        """
        num_temporal_grids = num_sampled_frames // self.temporal_patch_size
        
        if num_temporal_grids <= 1:
            return video_duration
        
        # 每个grid对应的秒数
        second_per_grid_t = video_duration / num_temporal_grids
        
        return second_per_grid_t
    
    def sample_from_timestamps(
        self,
        frames: List[Image.Image],
        timestamps: List[float],
    ) -> Tuple[List[Image.Image], float, dict]:
        """
        根据时间戳采样帧（适用于非均匀采样的视频）
        
        Args:
            frames: 原始帧列表
            timestamps: 每帧对应的时间戳(秒)
            
        Returns:
            sampled_frames, second_per_grid_t, metadata
        """
        if len(frames) != len(timestamps):
            raise ValueError("frames和timestamps长度必须相同")
        
        video_duration = timestamps[-1] - timestamps[0]
        target_frame_count = max(1, int(video_duration * self.target_fps))
        target_frame_count = self._align_to_temporal_patch(target_frame_count)
        
        # 限制最大帧数（防止显存溢出）
        if (
            self.max_sampled_frames is not None
            and self.max_sampled_frames > 0
            and target_frame_count > self.max_sampled_frames
        ):
            target_frame_count = self._align_to_temporal_patch(self.max_sampled_frames)
        
        # 按目标时间点采样最近的帧
        target_times = np.linspace(timestamps[0], timestamps[-1], target_frame_count)
        sampled_frames = []
        
        for t in target_times:
            # 找到最接近目标时间的帧
            idx = np.argmin(np.abs(np.array(timestamps) - t))
            sampled_frames.append(frames[idx])
        
        second_per_grid_t = self._calculate_second_per_grid_t(
            num_sampled_frames=len(sampled_frames),
            video_duration=video_duration,
        )
        
        metadata = {
            'original_frames': len(frames),
            'sampled_frames': len(sampled_frames),
            'video_duration': video_duration,
            'target_fps': self.target_fps,
            'second_per_grid_t': second_per_grid_t,
            'temporal_grids': len(sampled_frames) // self.temporal_patch_size,
        }
        
        return sampled_frames, second_per_grid_t, metadata


def calculate_expected_temporal_positions(
    num_frames: int,
    second_per_grid_t: float,
    temporal_patch_size: int = 2,
    tokens_per_second: int = 4,
) -> List[int]:
    """
    计算预期的temporal position IDs（用于验证）
    
    这个函数模拟官方 get_rope_index 中的时间编码计算
    """
    num_grids = num_frames // temporal_patch_size
    
    positions = []
    for grid_idx in range(num_grids):
        # 每个grid内的所有帧共享相同的temporal position
        time_value = grid_idx * second_per_grid_t * tokens_per_second
        for _ in range(temporal_patch_size):
            positions.append(int(time_value))
    
    return positions


def validate_time_encoding(
    sampled_frames: int,
    second_per_grid_t: float,
    expected_duration: float,
    temporal_patch_size: int = 2,
    tokens_per_second: int = 4,
    tolerance: float = 0.1,
) -> Tuple[bool, dict]:
    """
    验证时间编码是否正确
    
    Args:
        sampled_frames: 采样后的帧数
        second_per_grid_t: 计算得到的second_per_grid_t
        expected_duration: 预期的视频总时长
        tolerance: 时间误差容忍度(秒)
        
    Returns:
        is_valid: 是否通过验证
        details: 验证详情
    """
    num_grids = sampled_frames // temporal_patch_size
    
    # 计算最后一个grid的时间位置
    last_grid_time = (num_grids - 1) * second_per_grid_t
    
    # 计算覆盖的总时长
    total_covered_time = num_grids * second_per_grid_t
    
    # 验证
    time_error = abs(total_covered_time - expected_duration)
    is_valid = time_error <= tolerance
    
    details = {
        'num_grids': num_grids,
        'last_grid_time_seconds': last_grid_time,
        'total_covered_time': total_covered_time,
        'expected_duration': expected_duration,
        'time_error': time_error,
        'tolerance': tolerance,
        'is_valid': is_valid,
    }
    
    return is_valid, details