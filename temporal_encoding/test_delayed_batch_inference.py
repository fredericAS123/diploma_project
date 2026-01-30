"""
测试新方案：延迟批量编码 (Delayed Batch Inference)

测试流程：
1. 流式添加帧
2. 在不同时刻提问
3. 对比原生视频推理的结果
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image
import time
from pathlib import Path

from temporal_encoding.model.delayed_batch_inference import DelayedBatchInferenceEngine


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
    """测试延迟批量编码方案"""
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
    
    # 2. 创建引擎
    print("\n🚀 初始化 DelayedBatchInferenceEngine")
    engine = DelayedBatchInferenceEngine(
        model=model,
        processor=processor,
        device="cuda",
        star_memory_size=20,      # Star Memory 容量
        stream_window_size=20,    # Stream Memory 窗口大小
        max_pixels=2 * 224 * 224, # 低分辨率策略
    )
    
    # 3. 加载测试视频
    video_source = "/root/autodl-tmp/diploma/temporal_encoding/202208312002.mp4"
    frames = load_test_video_frames(video_source, max_frames=50)
    if not frames:
        raise RuntimeError("未加载到任何帧，请检查视频源")
    
    # 4. 流式添加帧
    print("\n" + "="*80)
    print("阶段 1：流式添加帧")
    print("="*80)
    
    for i, frame in enumerate(frames):
        timestamp = i * 0.5  # 假设每帧间隔 0.5 秒
        status = engine.add_frame(frame, timestamp)
        
        # 每 10 帧打印一次状态
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(frames)}] {status}")
    
    print(f"\n✅ 所有帧已添加")
    
    # 5. 查看统计信息
    stats = engine.get_statistics()
    print("\n📊 帧管理统计:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # 6. 提问测试（第一次会触发编码）
    print("\n" + "="*80)
    print("阶段 2：提问测试")
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
    
    # 7. 测试多次提问（cache复用）
    print("\n" + "="*80)
    print("阶段 3：Cache 复用测试")
    print("="*80)
    
    for i in range(3):
        print(f"\n🔄 第 {i+1} 次提问（应该复用 cache）")
        question = f"这是第 {i+1} 个问题，请简要回答视频内容。"
        
        t_start = time.time()
        answer, metrics = engine.ask(question, max_new_tokens=128)
        t_end = time.time()
        
        print(f"💬 回答: {answer}")
        print(f"⏱️  总耗时: {t_end - t_start:.2f}s")
        print(f"📊 编码耗时: {metrics.get('encoding_latency', 'N/A (cache复用)')}")
    
    # 8. 添加新帧后再提问
    print("\n" + "="*80)
    print("阶段 4：添加新帧 + 重新编码")
    print("="*80)
    
    # 添加 10 个新帧
    print("\n➕ 添加 10 个新帧...")
    for i in range(10):
        frame = frames[i % len(frames)]  # 复用已有帧
        timestamp = len(frames) * 0.5 + i * 0.5
        status = engine.add_frame(frame, timestamp)
    
    print(f"\n✅ 新帧已添加")
    
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
    """对比原生推理与延迟批量编码"""
    print("="*80)
    print("对比测试：原生推理 vs 延迟批量编码")
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
    
    question = "请详细描述视频中的主要内容和场景。"
    
    # 1. 原生推理
    print("\n" + "="*80)
    print("方法 1：原生视频推理")
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
    
    print(f"💬 原生回答: {native_answer}")
    print(f"⏱️  耗时: {t_end - t_start:.2f}s")
    
    # 2. 延迟批量编码
    print("\n" + "="*80)
    print("方法 2：延迟批量编码")
    print("="*80)
    
    engine = DelayedBatchInferenceEngine(
        model=model,
        processor=processor,
        device="cuda",
        star_memory_size=20,
        stream_window_size=20,
        max_pixels=2 * 224 * 224,
    )
    
    # 添加所有帧
    for i, frame in enumerate(frames):
        engine.add_frame(frame, i * 0.5)
    
    # 提问
    t_start = time.time()
    delayed_answer, metrics = engine.ask(question, max_new_tokens=256)
    t_end = time.time()
    
    print(f"💬 延迟编码回答: {delayed_answer}")
    print(f"⏱️  耗时: {t_end - t_start:.2f}s")
    print(f"📊 详细指标:")
    for key, value in metrics.items():
        if 'latency' in key or 'time' in key:
            print(f"  {key}: {value:.2f}s")
        else:
            print(f"  {key}: {value}")
    
    # 3. 相似度比较
    print("\n" + "="*80)
    print("结果对比")
    print("="*80)
    
    print(f"\n原生回答:\n{native_answer}\n")
    print(f"延迟编码回答:\n{delayed_answer}\n")
    
    # 简单的相似度估计（基于长度）
    len_ratio = len(delayed_answer) / len(native_answer) if len(native_answer) > 0 else 0
    print(f"📏 长度比: {len_ratio:.2f}")
    
    print("\n✅ 对比测试完成!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="测试延迟批量编码方案")
    parser.add_argument(
        "--mode",
        choices=["basic", "compare"],
        default="basic",
        help="测试模式：basic（基础测试）, compare（对比测试）"
    )
    
    args = parser.parse_args()
    
    if args.mode == "basic":
        test_delayed_batch_inference()
    elif args.mode == "compare":
        test_native_vs_delayed()
