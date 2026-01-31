"""
Smart Frame Manager - 借鉴 Flash-VStream 的双重记忆机制

核心思想：
1. Star Memory: 保留重要的关键帧（场景变化、高信息量）
2. Stream Memory: 保留最近的N帧（FIFO滑动窗口）
3. 在回答问题时，合并两种记忆进行批量编码

内存优化（参考官方web_demo_streaming）：
- 帧保存到硬盘临时目录，内存中只存路径
- 提问时才按需加载帧
"""

import numpy as np
from collections import deque
from typing import List, Dict, Optional, Tuple
from PIL import Image
import time
import tempfile
import os
import uuid
import shutil
import atexit


class SmartFrameManager:
    """智能帧管理器 - 双重记忆机制 + 硬盘缓存"""
    
    def __init__(
        self,
        star_memory_size: int = 20,           # Star Memory 最大容量
        stream_window_size: int = 20,         # Stream Memory 滑动窗口大小
        importance_threshold: float = 0.6,    # 重要性阈值
        scene_change_threshold: float = 0.3,  # 场景变化阈值
        use_disk_cache: bool = True,          # 是否使用硬盘缓存（节省内存）
        cache_dir: Optional[str] = None,      # 缓存目录，None则自动创建临时目录
    ):
        """
        Args:
            star_memory_size: 最多保留多少个重要帧
            stream_window_size: 滑动窗口大小
            importance_threshold: 超过此阈值才进入 Star Memory
            scene_change_threshold: 场景变化检测阈值
            use_disk_cache: 是否使用硬盘缓存（推荐True，大幅节省内存）
            cache_dir: 缓存目录路径
        """
        self.star_memory_size = star_memory_size
        self.stream_window_size = stream_window_size
        self.importance_threshold = importance_threshold
        self.scene_change_threshold = scene_change_threshold
        self.use_disk_cache = use_disk_cache
        
        # 硬盘缓存设置
        if use_disk_cache:
            if cache_dir is None:
                self.cache_dir = tempfile.mkdtemp(prefix="qwen_vl_frames_")
            else:
                self.cache_dir = cache_dir
                os.makedirs(cache_dir, exist_ok=True)
            # 注册退出时清理
            atexit.register(self._cleanup_cache)
        else:
            self.cache_dir = None
        
        # 双重记忆存储
        self.star_memory: List[Dict] = []           # 重要帧（长期记忆）
        self.stream_memory: deque = deque(maxlen=stream_window_size)  # 滑动窗口（短期记忆）
        
        # 状态追踪
        self.last_frame_array: Optional[np.ndarray] = None
        self.frame_count = 0
        self.total_frames_added = 0
        
        # 首帧始终加入 Star Memory
        self.first_frame_added = False
        
        print(f"✅ SmartFrameManager Initialized:")
        print(f"   📌 Star Memory: {star_memory_size} frames")
        print(f"   🌊 Stream Window: {stream_window_size} frames")
        print(f"   🎯 Importance Threshold: {importance_threshold}")
        if use_disk_cache:
            print(f"   💾 Disk Cache: {self.cache_dir}")
        else:
            print(f"   ⚠️  Memory Mode (no disk cache)")
    
    def _save_frame_to_disk(self, frame: Image.Image) -> str:
        """将帧保存到硬盘，返回文件路径"""
        filename = f"{uuid.uuid4()}.jpg"
        filepath = os.path.join(self.cache_dir, filename)
        frame.save(filepath, "JPEG", quality=90)
        return filepath
    
    def _load_frame_from_disk(self, filepath: str) -> Image.Image:
        """从硬盘加载帧"""
        return Image.open(filepath).convert("RGB")
    
    def _cleanup_cache(self):
        """清理缓存目录"""
        if self.cache_dir and os.path.exists(self.cache_dir):
            try:
                shutil.rmtree(self.cache_dir)
                print(f"🗑️  Cache cleaned: {self.cache_dir}")
            except Exception as e:
                print(f"⚠️  Failed to clean cache: {e}")
    
    def add_frame(self, frame: Image.Image, timestamp: float) -> Dict[str, any]:
        """
        添加新帧到管理器
        
        Args:
            frame: PIL Image
            timestamp: 时间戳（秒）
        
        Returns:
            添加结果统计
        """
        self.frame_count += 1
        self.total_frames_added += 1
        
        # 转换为numpy数组用于分析（降采样加速计算）
        frame_array = np.array(frame.resize((224, 224)))
        
        # 计算重要性分数
        importance_score = self._compute_importance(frame_array)
        
        # 检测场景变化
        is_scene_change = self._is_scene_change(frame_array)
        
        # 保存帧（硬盘或内存）
        if self.use_disk_cache:
            frame_data = self._save_frame_to_disk(frame)
        else:
            frame_data = frame  # 直接保存PIL对象
        
        # 构建帧信息
        frame_info = {
            'frame': frame_data,  # 路径或PIL对象
            'timestamp': timestamp,
            'frame_index': self.frame_count,
            'importance': importance_score,
            'is_scene_change': is_scene_change,
            'added_to_star': False,
        }
        
        # 1. 始终添加到 Stream Memory（滑动窗口）
        # 注意：deque 会自动移除旧帧，需要清理其硬盘文件
        if len(self.stream_memory) == self.stream_window_size and self.use_disk_cache:
            old_frame = self.stream_memory[0]
            # 只有不在star_memory中的帧才能删除
            if not old_frame['added_to_star'] and isinstance(old_frame['frame'], str):
                if os.path.exists(old_frame['frame']):
                    try:
                        os.remove(old_frame['frame'])
                    except:
                        pass
        
        self.stream_memory.append(frame_info)
        
        # 2. 判断是否加入 Star Memory
        added_to_star = False
        reason = None
        
        if not self.first_frame_added:
            # 首帧强制加入
            self._add_to_star_memory(frame_info)
            added_to_star = True
            reason = "首帧"
            self.first_frame_added = True
        elif is_scene_change:
            # 场景变化帧加入
            self._add_to_star_memory(frame_info)
            added_to_star = True
            reason = "场景变化"
        elif importance_score > self.importance_threshold:
            # 高重要性帧加入
            self._add_to_star_memory(frame_info)
            added_to_star = True
            reason = f"高重要性({importance_score:.2f})"
        
        # 更新状态
        self.last_frame_array = frame_array.copy()
        
        return {
            'added_to_star': added_to_star,
            'reason': reason,
            'importance': importance_score,
            'star_count': len(self.star_memory),
            'stream_count': len(self.stream_memory),
        }
    
    def _compute_importance(self, frame_array: np.ndarray) -> float:
        """
        计算帧的重要性分数
        
        综合考虑：
        1. 与上一帧的差异（运动/变化）
        2. 帧的信息量（纹理复杂度）
        """
        if self.last_frame_array is None:
            return 1.0  # 首帧最重要
        
        # 1. 帧间差异（运动检测）
        diff = np.abs(frame_array.astype(float) - self.last_frame_array.astype(float))
        motion_score = np.mean(diff) / 255.0
        
        # 2. 信息量（标准差作为复杂度度量）
        gray = np.mean(frame_array, axis=2)
        complexity_score = np.std(gray) / 128.0  # 归一化
        
        # 综合得分
        importance = 0.7 * motion_score + 0.3 * complexity_score
        return float(importance)
    
    def _is_scene_change(self, frame_array: np.ndarray) -> bool:
        """检测是否为场景变化帧"""
        if self.last_frame_array is None:
            return True  # 首帧视为场景变化
        
        # 计算帧间差异
        diff = np.abs(frame_array.astype(float) - self.last_frame_array.astype(float))
        change_ratio = np.mean(diff) / 255.0
        
        return change_ratio > self.scene_change_threshold
    
    def _add_to_star_memory(self, frame_info: Dict):
        """添加帧到 Star Memory，并维护容量限制"""
        frame_info['added_to_star'] = True
        self.star_memory.append(frame_info)
        
        # 如果超过容量，移除最不重要的帧（但保留首帧）
        if len(self.star_memory) > self.star_memory_size:
            # 首帧（frame_index=1）始终保留
            removable = [f for f in self.star_memory if f['frame_index'] > 1]
            
            if removable:
                # 按重要性排序，移除最不重要的
                removable.sort(key=lambda x: x['importance'])
                frame_to_remove = removable[0]
                self.star_memory.remove(frame_to_remove)
                
                # 如果该帧不在stream_memory中，删除其硬盘文件
                if self.use_disk_cache and isinstance(frame_to_remove['frame'], str):
                    in_stream = any(
                        f['frame'] == frame_to_remove['frame'] 
                        for f in self.stream_memory
                    )
                    if not in_stream and os.path.exists(frame_to_remove['frame']):
                        try:
                            os.remove(frame_to_remove['frame'])
                        except:
                            pass
    
    def get_all_frames(self) -> Tuple[List[Image.Image], List[float], Dict[str, any]]:
        """
        获取所有需要编码的帧（Star + Stream，去重并排序）
        
        Returns:
            (frames_list, timestamps_list, metadata)
            - frames_list: PIL Image 列表（按时间排序）
            - timestamps_list: 对应的时间戳列表（秒）
            - metadata: 统计信息
        """
        # 收集所有帧，用timestamp去重
        frame_dict = {}
        
        # 1. 添加 Star Memory
        for f in self.star_memory:
            ts = f['timestamp']
            if ts not in frame_dict:
                frame_dict[ts] = f
        
        # 2. 添加 Stream Memory
        for f in self.stream_memory:
            ts = f['timestamp']
            if ts not in frame_dict:
                frame_dict[ts] = f
        
        # 3. 按时间戳排序
        sorted_items = sorted(frame_dict.items(), key=lambda x: x[0])
        
        # 4. 提取帧列表和时间戳列表（如果是路径则加载）
        frames = []
        timestamps = []
        for ts, frame_info in sorted_items:
            frame_data = frame_info['frame']
            if isinstance(frame_data, str):
                # 从硬盘加载
                frames.append(self._load_frame_from_disk(frame_data))
            else:
                # 直接使用PIL对象
                frames.append(frame_data)
            timestamps.append(ts)
        
        # 5. 元数据统计
        star_count = len(self.star_memory)
        stream_count = len(self.stream_memory)
        unique_count = len(frames)
        overlap_count = star_count + stream_count - unique_count
        
        # 计算时间跨度
        time_span = timestamps[-1] - timestamps[0] if len(timestamps) >= 2 else 0.0
        
        metadata = {
            'star_frames': star_count,
            'stream_frames': stream_count,
            'unique_frames': unique_count,
            'overlap_frames': overlap_count,
            'total_added': self.total_frames_added,
            'compression_ratio': self.total_frames_added / unique_count if unique_count > 0 else 0,
            'time_span': time_span,
            'min_timestamp': timestamps[0] if timestamps else 0.0,
            'max_timestamp': timestamps[-1] if timestamps else 0.0,
        }
        
        return frames, timestamps, metadata
    
    def get_frame_paths(self) -> Tuple[List[str], Dict[str, any]]:
        """
        获取所有帧的文件路径（仅硬盘缓存模式可用）
        
        Returns:
            (paths_list, metadata)
        """
        if not self.use_disk_cache:
            raise ValueError("get_frame_paths() only available in disk cache mode")
        
        # 收集所有帧路径，用timestamp去重
        frame_dict = {}
        
        for f in self.star_memory:
            ts = f['timestamp']
            if ts not in frame_dict:
                frame_dict[ts] = f
        
        for f in self.stream_memory:
            ts = f['timestamp']
            if ts not in frame_dict:
                frame_dict[ts] = f
        
        sorted_items = sorted(frame_dict.items(), key=lambda x: x[0])
        paths = [item[1]['frame'] for item in sorted_items]
        
        metadata = {
            'star_frames': len(self.star_memory),
            'stream_frames': len(self.stream_memory),
            'unique_frames': len(paths),
            'total_added': self.total_frames_added,
        }
        
        return paths, metadata
    
    def get_statistics(self) -> Dict[str, any]:
        """获取统计信息"""
        # 收集唯一帧数（不加载帧）
        frame_dict = {}
        for f in self.star_memory:
            frame_dict[f['timestamp']] = f
        for f in self.stream_memory:
            if f['timestamp'] not in frame_dict:
                frame_dict[f['timestamp']] = f
        unique_count = len(frame_dict)
        
        return {
            'total_frames_added': self.total_frames_added,
            'star_memory_size': len(self.star_memory),
            'stream_memory_size': len(self.stream_memory),
            'unique_frames': unique_count,
            'compression_ratio': f"{self.total_frames_added / unique_count:.2f}x" if unique_count > 0 else "N/A",
            'disk_cache_enabled': self.use_disk_cache,
        }
    
    def reset(self):
        """重置管理器"""
        # 清理硬盘缓存
        if self.use_disk_cache and self.cache_dir:
            for f in self.star_memory:
                if isinstance(f['frame'], str) and os.path.exists(f['frame']):
                    try:
                        os.remove(f['frame'])
                    except:
                        pass
            for f in self.stream_memory:
                if isinstance(f['frame'], str) and os.path.exists(f['frame']):
                    try:
                        os.remove(f['frame'])
                    except:
                        pass
        
        self.star_memory.clear()
        self.stream_memory.clear()
        self.last_frame_array = None
        self.frame_count = 0
        self.total_frames_added = 0
        self.first_frame_added = False
        print("🔄 SmartFrameManager Reset.")
