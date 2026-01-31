"""
测试新方案：延迟批量编码 (Delayed Batch Inference)

测试流程：
1. 流式添加帧
2. 在不同时刻提问
3. 对比原生视频推理的结果
4. 测试动态采样与绝对时间编码
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image
import time
from pathlib import Path
from typing import TextIO

from temporal_encoding.model.delayed_batch_inference import DelayedBatchInferenceEngine
from temporal_encoding.model.video_sampler import validate_time_encoding


class TeeIO:
    """将输出同时写入多个流"""

    def __init__(self, *streams: TextIO):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return False


def capture_test_output(log_path: Path):
    """重定向 stdout/stderr 到文件（并保留控制台输出）"""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8")
    stdout_tee = TeeIO(sys.stdout, log_file)
    stderr_tee = TeeIO(sys.stderr, log_file)
    return log_file, stdout_tee, stderr_tee


def load_test_video_frames(video_source: str, max_frames: int = 50):
    """加载测试视频帧（支持帧目录或视频文件）"""
    source_path = Path(video_source)
    frames: list[Image.Image] = []

    if source_path.is_dir():
        frame_files = sorted(source_path.glob("*.jpg"))[:max_frames]
        for f in frame_files:
            frames.append(Image.open(f).convert("RGB"))
    elif source_path.is_file():
        try:
            import cv2
        except Exception as exc:
            raise RuntimeError("缺少 OpenCV，无法从视频文件提取帧") from exc

        cap = cv2.VideoCapture(str(source_path))
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频文件: {source_path}")

        try:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count <= 0:
                raise RuntimeError("视频帧数为 0，无法提取")

            if max_frames >= frame_count:
                indices = list(range(frame_count))
            else:
                step = frame_count / max_frames
                indices = [int(i * step) for i in range(max_frames)]

            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret and frame is not None:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(Image.fromarray(frame_rgb))
        finally:
            cap.release()
    else:
        raise RuntimeError(f"视频源不存在: {source_path}")

    print(f"✅ 加载 {len(frames)} 帧")
    return frames


def test_delayed_batch_inference():
    """测试延迟批量编码方案（含动态采样与绝对时间编码）"""
    print("="*80)
    print("测试：延迟批量编码方案 (Delayed Batch Inference)")
    print("="*80)
    
    # 1. 加载模型
    model_path = "/root/autodl-tmp/Qwen/Qwen2___5-VL-3B-Instruct"
    print(f"\n📦 加载模型: {model_path}")
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
    
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    
    # 2. 创建引擎（启用动态采样：1fps + 硬盘缓存）
    print("\n🚀 初始化 DelayedBatchInferenceEngine（启用 1fps 采样 + 绝对时间编码 + 硬盘缓存）")
    engine = DelayedBatchInferenceEngine(
        model=model,
        processor=processor,
        device="cuda",
        star_memory_size=20,      # Star Memory 容量
        stream_window_size=20,    # Stream Memory 窗口大小
        max_pixels=2 * 224 * 224, # 低分辨率策略
        target_fps=1.0,           # 动态采样：1fps
        enable_absolute_time_encoding=True,  # 启用绝对时间编码
        use_disk_cache=True,      # 启用硬盘缓存（节省内存）
    )
    
    # 3. 加载测试视频
    video_source = "/root/autodl-tmp/diploma/temporal_encoding/202208312002.mp4"
    frames = load_test_video_frames(video_source, max_frames=50)
    if not frames:
        raise RuntimeError("未加载到任何帧，请检查视频源")
    
    # 4. 流式添加帧（模拟50秒视频，每帧间隔1秒）
    print("\n" + "="*80)
    print("阶段 1：流式添加帧（模拟50秒视频）")
    print("="*80)
    
    video_duration = 50.0  # 模拟50秒视频
    frame_interval = video_duration / len(frames)
    
    for i, frame in enumerate(frames):
        timestamp = i * frame_interval
        status = engine.add_frame(frame, timestamp)
        
        # 每 10 帧打印一次状态
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(frames)}] t={timestamp:.1f}s | {status}")
    
    print(f"\n✅ 所有帧已添加，模拟视频时长: {video_duration}s")
    
    # 5. 查看统计信息
    stats = engine.get_statistics()
    print("\n📊 帧管理统计:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # 6. 提问测试（第一次会触发编码+采样）
    print("\n" + "="*80)
    print("阶段 2：提问测试（触发动态采样 + 绝对时间编码）")
    print("="*80)
    
    questions = [
        "请描述视频中的主要内容。",
        "视频中有什么人物或物体？",
        "视频中有什么场景变化？",
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n❓ 问题 {i}: {question}")
        answer, metrics = engine.ask(question, max_new_tokens=256)
        
        print(f"💬 回答: {answer}")
        print(f"📊 指标:")
        for key, value in metrics.items():
            if 'latency' in key or 'time' in key:
                print(f"  {key}: {value:.2f}s")
            else:
                print(f"  {key}: {value}")
        
        # 验证时间编码（仅第一次提问时有采样元数据）
        if i == 1 and engine.last_sample_metadata:
            meta = engine.last_sample_metadata
            print(f"\n📐 采样元数据验证:")
            print(f"  原始帧数: {meta['original_frames']}")
            print(f"  采样后帧数: {meta['sampled_frames']}")
            print(f"  second_per_grid_t: {meta['second_per_grid_t']:.4f}s")
            print(f"  temporal_grids: {meta['temporal_grids']}")
            # 计算压缩比（原始帧数 / 采样后帧数）
            compression_ratio = meta['original_frames'] / meta['sampled_frames'] if meta['sampled_frames'] > 0 else 0
            print(f"  压缩比: {compression_ratio:.2f}x")
            
            # 验证时间编码覆盖
            is_valid, details = validate_time_encoding(
                sampled_frames=meta['sampled_frames'],
                second_per_grid_t=meta['second_per_grid_t'],
                expected_duration=meta['video_duration'],
                tolerance=1.0,
            )
            print(f"  时间编码验证: {'✅ 通过' if is_valid else '❌ 失败'}")
            print(f"  覆盖时长: {details['total_covered_time']:.2f}s / {details['expected_duration']:.2f}s")
    
    # 7. 测试多次提问（cache复用）
    print("\n" + "="*80)
    print("阶段 3：Cache 复用测试（Token Streaming 输出）")
    print("="*80)
    
    for i in range(3):
        print(f"\n🔄 第 {i+1} 次提问（Streaming 输出，应该复用 cache）")
        question = f"这是第 {i+1} 个问题，请简要回答视频内容。"
        
        t_start = time.time()
        print(f"💬 回答: ", end="", flush=True)
        
        # 使用 streaming 输出
        for text in engine.ask_stream(question, max_new_tokens=128):
            print(text, end="", flush=True)
        print()  # 换行
        
        t_end = time.time()
        
        # 获取 metrics
        metrics = engine.last_stream_metrics
        print(f"⏱️  总耗时: {t_end - t_start:.2f}s")
        print(f"📊 输出tokens: {metrics.get('output_tokens', 'N/A')}")
        print(f"📊 编码耗时: {metrics.get('encoding_latency', 'N/A (cache复用)')}")
    
    # 8. 添加新帧后再提问
    print("\n" + "="*80)
    print("阶段 4：添加新帧 + 重新编码")
    print("="*80)
    
    # 添加 10 个新帧
    print("\n➕ 添加 10 个新帧...")
    for i in range(10):
        frame = frames[i % len(frames)]  # 复用已有帧
        timestamp = video_duration + i * 1.0  # 继续累加时间戳
        status = engine.add_frame(frame, timestamp)
    
    print(f"\n✅ 新帧已添加，新视频时长: {video_duration + 10}s")
    
    # 再次提问（会触发重新编码）
    print(f"\n❓ 添加新帧后提问:")
    question = "现在视频有更新，请描述最新的内容。"
    answer, metrics = engine.ask(question, max_new_tokens=256)
    
    print(f"💬 回答: {answer}")
    print(f"📊 指标:")
    for key, value in metrics.items():
        if 'latency' in key or 'time' in key:
            print(f"  {key}: {value:.2f}s")
        else:
            print(f"  {key}: {value}")
    
    # 再次验证时间编码
    if engine.last_sample_metadata:
        meta = engine.last_sample_metadata
        print(f"\n📐 更新后采样元数据:")
        print(f"  采样后帧数: {meta['sampled_frames']}")
        print(f"  second_per_grid_t: {meta['second_per_grid_t']:.4f}s")
        print(f"  视频时长: {meta['video_duration']:.2f}s")
    
    # 9. 最终统计
    print("\n" + "="*80)
    print("最终统计")
    print("="*80)
    
    final_stats = engine.get_statistics()
    print(f"\n📊 最终帧管理统计:")
    for key, value in final_stats.items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*80)
    print("✅ 测试完成!")
    print("="*80)


def test_native_vs_delayed():
    """对比原生推理与延迟批量编码（含动态采样）"""
    print("="*80)
    print("对比测试：原生推理 vs 延迟批量编码（1fps采样 + 绝对时间编码）")
    print("="*80)
    
    model_path = "/root/autodl-tmp/Qwen/Qwen2___5-VL-3B-Instruct"
    
    # 加载模型
    print(f"\n📦 加载模型: {model_path}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
    
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    
    # 加载测试视频
    video_source = "/root/autodl-tmp/diploma/temporal_encoding/202208312002.mp4"
    frames = load_test_video_frames(video_source, max_frames=30)
    if not frames:
        raise RuntimeError("未加载到任何帧，请检查视频源")
    
    # 模拟30秒视频
    video_duration = 30.0
    question = "请详细描述视频中的主要内容和场景。"
    
    # 1. 原生推理
    print("\n" + "="*80)
    print("方法 1：原生视频推理（无采样）")
    print("="*80)
    
    messages = [{
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": frames,
                "max_pixels": 4 * 224 * 224,
            },
            {"type": "text", "text": question},
        ],
    }]
    
    text_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    t_start = time.time()
    inputs = processor(
        text=[text_prompt],
        videos=[frames],
        padding=True,
        return_tensors="pt",
    ).to("cuda")
    
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
        )
    
    native_answer = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].split("assistant\n")[-1]
    
    t_end = time.time()
    native_time = t_end - t_start
    
    print(f"💬 原生回答: {native_answer}")
    print(f"⏱️  耗时: {native_time:.2f}s")
    print(f"📐 输入帧数: {len(frames)}")
    
    # 2. 延迟批量编码（启用1fps采样）
    print("\n" + "="*80)
    print("方法 2：延迟批量编码（1fps采样 + 绝对时间编码）")
    print("="*80)
    
    engine = DelayedBatchInferenceEngine(
        model=model,
        processor=processor,
        device="cuda",
        star_memory_size=20,
        stream_window_size=20,
        max_pixels=2 * 224 * 224,
        target_fps=1.0,  # 1fps采样
        enable_absolute_time_encoding=True,
        use_disk_cache=True,  # 启用硬盘缓存
    )
    
    # 添加所有帧（带时间戳）
    frame_interval = video_duration / len(frames)
    for i, frame in enumerate(frames):
        engine.add_frame(frame, i * frame_interval)
    
    # 提问
    t_start = time.time()
    delayed_answer, metrics = engine.ask(question, max_new_tokens=256)
    t_end = time.time()
    delayed_time = t_end - t_start
    
    print(f"💬 延迟编码回答: {delayed_answer}")
    print(f"⏱️  耗时: {delayed_time:.2f}s")
    print(f"📊 详细指标:")
    for key, value in metrics.items():
        if 'latency' in key or 'time' in key:
            print(f"  {key}: {value:.2f}s")
        else:
            print(f"  {key}: {value}")
    
    # 显示采样信息
    if engine.last_sample_metadata:
        meta = engine.last_sample_metadata
        print(f"\n📐 采样信息:")
        print(f"  原始帧数: {meta['original_frames']} -> 采样后: {meta['sampled_frames']}")
        print(f"  second_per_grid_t: {meta['second_per_grid_t']:.4f}s")
        print(f"  temporal_grids: {meta['temporal_grids']}")
        # 计算压缩比
        compression_ratio = meta['original_frames'] / meta['sampled_frames'] if meta['sampled_frames'] > 0 else 0
        print(f"  压缩比: {compression_ratio:.2f}x")
        
        # 验证时间编码
        is_valid, details = validate_time_encoding(
            sampled_frames=meta['sampled_frames'],
            second_per_grid_t=meta['second_per_grid_t'],
            expected_duration=meta['video_duration'],
            tolerance=1.0,
        )
        print(f"  时间编码验证: {'✅ 通过' if is_valid else '❌ 失败'}")
    
    # 3. 结果对比
    print("\n" + "="*80)
    print("结果对比")
    print("="*80)
    
    print(f"\n原生回答:\n{native_answer}\n")
    print(f"延迟编码回答:\n{delayed_answer}\n")
    
    # 对比指标
    len_ratio = len(delayed_answer) / len(native_answer) if len(native_answer) > 0 else 0
    speedup = native_time / delayed_time if delayed_time > 0 else 0
    
    print(f"📏 长度比: {len_ratio:.2f}")
    print(f"⚡ 速度对比: 原生 {native_time:.2f}s vs 延迟编码 {delayed_time:.2f}s")
    print(f"   {'加速' if speedup > 1 else '减速'}: {abs(speedup - 1) * 100:.1f}%")
    
    print("\n✅ 对比测试完成!")


def test_sparse_time_encoding_accuracy():
    """
    稀疏帧时间编码精度测试
    
    关键测试场景：
    - Star Memory: t=0s 的关键帧
    - Stream Memory: t=50~55s 的最近帧
    - 验证 second_per_grid_t 反映的是 ~55s 的真实时间跨度，而非 5s
    
    这是验证时间编码正确性的核心测试。
    """
    print("="*80)
    print("稀疏帧时间编码精度测试 (Sparse Frame Time Encoding)")
    print("="*80)
    print("\n🎯 测试目标：验证 Star+Stream 混合帧的时间编码正确反映真实时间跨度")
    
    # 不需要真实模型，只测试采样逻辑
    from temporal_encoding.model.video_sampler import VideoSampler, validate_time_encoding
    from temporal_encoding.model.smart_frame_manager import SmartFrameManager
    
    # 1. 创建帧管理器（使用内存模式简化测试）
    print("\n📦 初始化 SmartFrameManager (内存模式)")
    frame_manager = SmartFrameManager(
        star_memory_size=10,
        stream_window_size=10,
        use_disk_cache=False,  # 测试用内存模式
    )
    
    # 2. 模拟稀疏帧场景
    print("\n" + "="*80)
    print("阶段 1：构造稀疏帧场景")
    print("="*80)
    
    # 创建测试帧（简单的纯色图像）
    def create_test_frame(color_value: int) -> Image.Image:
        return Image.new('RGB', (224, 224), color=(color_value, color_value, color_value))
    
    # 场景：
    # - t=0s: 首帧（自动进入 Star Memory）
    # - t=50s ~ t=55s: 最近的 Stream Memory 帧
    
    # 添加首帧 (t=0s) - 会进入 Star Memory
    print(f"\n➕ 添加首帧 @ t=0.0s (Star Memory)")
    frame_manager.add_frame(create_test_frame(0), timestamp=0.0)
    
    # 添加中间的一些关键帧（模拟场景变化）
    print(f"➕ 添加场景变化帧 @ t=25.0s (Star Memory)")
    # 人为制造场景变化（大幅度颜色变化）
    frame_manager.add_frame(create_test_frame(200), timestamp=25.0)
    
    # 添加 Stream Memory 帧 (t=50s ~ t=55s)
    print(f"➕ 添加 Stream Memory 帧 @ t=50.0s ~ t=55.0s")
    for i in range(6):  # 6帧，t=50, 51, 52, 53, 54, 55
        t = 50.0 + i
        frame_manager.add_frame(create_test_frame(100 + i), timestamp=t)
    
    # 3. 获取帧和时间戳
    print("\n" + "="*80)
    print("阶段 2：获取帧并验证时间戳")
    print("="*80)
    
    frames, timestamps, metadata = frame_manager.get_all_frames()
    
    print(f"\n📊 帧管理器状态:")
    print(f"   Star Memory: {metadata['star_frames']} 帧")
    print(f"   Stream Memory: {metadata['stream_frames']} 帧")
    print(f"   唯一帧数: {metadata['unique_frames']} 帧")
    print(f"   时间跨度: {metadata['time_span']:.2f}s (从 t={metadata['min_timestamp']:.1f}s 到 t={metadata['max_timestamp']:.1f}s)")
    
    print(f"\n📋 时间戳列表: {timestamps}")
    
    # 验证时间戳确实覆盖了大范围
    assert metadata['min_timestamp'] == 0.0, "最小时间戳应为 0.0s"
    assert metadata['max_timestamp'] == 55.0, "最大时间戳应为 55.0s"
    assert metadata['time_span'] == 55.0, "时间跨度应为 55.0s"
    print(f"   ✅ 时间戳验证通过")
    
    # 4. 使用 sample_from_timestamps 进行采样
    print("\n" + "="*80)
    print("阶段 3：基于时间戳采样 (1fps)")
    print("="*80)
    
    sampler = VideoSampler(target_fps=1.0)
    sampled_frames, second_per_grid_t, sample_meta = sampler.sample_from_timestamps(
        frames=frames,
        timestamps=timestamps,
    )
    
    print(f"\n📐 采样结果:")
    print(f"   原始帧数: {sample_meta['original_frames']}")
    print(f"   采样后帧数: {sample_meta['sampled_frames']}")
    print(f"   视频时长: {sample_meta['video_duration']:.2f}s")
    print(f"   second_per_grid_t: {second_per_grid_t:.4f}s")
    print(f"   temporal_grids: {sample_meta['temporal_grids']}")
    
    # 5. 关键验证：second_per_grid_t 应反映 55s 的时间跨度
    print("\n" + "="*80)
    print("阶段 4：关键验证 - 时间编码精度")
    print("="*80)
    
    # 核心断言：时间编码应覆盖 55 秒
    is_valid, details = validate_time_encoding(
        sampled_frames=sample_meta['sampled_frames'],
        second_per_grid_t=second_per_grid_t,
        expected_duration=sample_meta['video_duration'],
        tolerance=1.0,
    )
    
    print(f"\n🔍 时间编码验证:")
    print(f"   Temporal Grids: {details['num_grids']}")
    print(f"   最后一个 Grid 时间: {details['last_grid_time_seconds']:.2f}s")
    print(f"   覆盖总时长: {details['total_covered_time']:.2f}s")
    print(f"   预期时长: {details['expected_duration']:.2f}s")
    print(f"   时间误差: {details['time_error']:.2f}s")
    print(f"   验证结果: {'✅ 通过' if is_valid else '❌ 失败'}")
    
    # 关键断言
    assert is_valid, f"时间编码验证失败: {details}"
    assert sample_meta['video_duration'] >= 50.0, \
        f"video_duration 应 >= 50s，实际: {sample_meta['video_duration']}"
    
    # 验证 second_per_grid_t 合理性
    # 55秒视频，1fps采样 = ~54帧（对齐到偶数）= 27 grids
    # second_per_grid_t ≈ 55 / 27 ≈ 2.03s
    expected_grids = sample_meta['sampled_frames'] // 2
    expected_second_per_grid = 55.0 / expected_grids
    
    print(f"\n📊 second_per_grid_t 合理性检查:")
    print(f"   预期 grids: {expected_grids}")
    print(f"   预期 second_per_grid_t: {expected_second_per_grid:.4f}s")
    print(f"   实际 second_per_grid_t: {second_per_grid_t:.4f}s")
    
    # 允许一定误差
    assert abs(second_per_grid_t - expected_second_per_grid) < 0.5, \
        f"second_per_grid_t 偏差过大: 预期 {expected_second_per_grid:.4f}s, 实际 {second_per_grid_t:.4f}s"
    
    print(f"   ✅ 验证通过")
    
    # 6. 对比错误做法（基于索引采样）
    print("\n" + "="*80)
    print("阶段 5：对比演示 - 错误做法 vs 正确做法")
    print("="*80)
    
    # 错误做法：使用 sample_frames（基于索引）
    wrong_sampler = VideoSampler(target_fps=1.0)
    _, wrong_second_per_grid, wrong_meta = wrong_sampler.sample_frames(
        frames=frames,
        original_fps=len(frames) / 55.0,  # 错误地假设均匀分布
        video_duration=55.0,
    )
    
    print(f"\n❌ 错误做法 (sample_frames - 基于索引):")
    print(f"   这种方法假设帧是均匀分布的，会把 t=0s 和 t=25s 的帧")
    print(f"   当作相邻帧处理，导致时间感知错乱")
    print(f"   second_per_grid_t: {wrong_second_per_grid:.4f}s")
    
    print(f"\n✅ 正确做法 (sample_from_timestamps - 基于时间戳):")
    print(f"   这种方法尊重真实时间戳，正确反映帧之间的时间间隔")
    print(f"   second_per_grid_t: {second_per_grid_t:.4f}s")
    
    # 最终总结
    print("\n" + "="*80)
    print("测试总结")
    print("="*80)
    print(f"""
🎯 核心验证点:
   1. ✅ 时间戳正确返回 (get_all_frames 返回 timestamps)
   2. ✅ sample_from_timestamps 正确处理稀疏帧
   3. ✅ second_per_grid_t 反映真实时间跨度 (~{second_per_grid_t:.2f}s/grid)
   4. ✅ 模型将感知到 55 秒的时间流逝

🔑 关键修复:
   - 使用 sample_from_timestamps 替代 sample_frames
   - 确保 Star+Stream 混合帧的时间编码正确
   - 模型现在能正确理解"t=0s 到 t=55s 中间有大段时间流逝"
""")
    
    print("✅ 稀疏帧时间编码测试完成!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="测试延迟批量编码方案")
    parser.add_argument(
        "--mode",
        choices=["basic", "compare", "sparse"],
        default="basic",
        help="测试模式：basic（基础测试）, compare（对比测试）, sparse（稀疏帧时间编码测试）"
    )
    
    args = parser.parse_args()

    log_path = Path(__file__).with_name("test_delayed_batch_inference_output.txt")
    log_file, stdout_tee, stderr_tee = capture_test_output(log_path)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    try:
        sys.stdout = stdout_tee
        sys.stderr = stderr_tee

        if args.mode == "basic":
            test_delayed_batch_inference()
        elif args.mode == "compare":
            test_native_vs_delayed()
        elif args.mode == "sparse":
            test_sparse_time_encoding_accuracy()
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()
        print(f"✅ 测试输出已保存到: {log_path}")
