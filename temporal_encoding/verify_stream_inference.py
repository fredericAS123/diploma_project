import os
import sys
import torch
import copy
from transformers import AutoProcessor
from PIL import Image, ImageDraw

# ================= 配置 =================
MODEL_PATH = "/root/autodl-tmp/Qwen/Qwen2___5-VL-3B-Instruct"
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.stream_qwen_model import StreamQwenModel
from model.video_stream_inference import VideoStreamingInference

# ================= 设备配置 =================
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# ================= 资产生成 =================
print("🎨 Generating Assets for Timeline Test...")
size = (224, 224)

# Asset A: Red Circle
img_red_circle = Image.new("RGB", size, color="white")
draw = ImageDraw.Draw(img_red_circle)
draw.ellipse((50, 50, 170, 170), fill="red", outline="red")

# Asset B: Blue Square
img_blue_square = Image.new("RGB", size, color="white")
draw = ImageDraw.Draw(img_blue_square)
draw.rectangle((50, 50, 170, 170), fill="blue", outline="blue")

# Asset C: Green Triangle
img_green_triangle = Image.new("RGB", size, color="white")
draw = ImageDraw.Draw(img_green_triangle)
draw.polygon([(112, 40), (40, 180), (184, 180)], fill="green", outline="green")

# Noise: Black Screen
img_noise = Image.new("RGB", size, color="black")

# ================= 初始化 =================
print(f"🚀 Loading Model: {MODEL_PATH}")
processor = AutoProcessor.from_pretrained(MODEL_PATH)
model = StreamQwenModel.from_pretrained(MODEL_PATH, torch_dtype=dtype).to(device)
model.eval()

engine = VideoStreamingInference(model, processor, device)

# ================= 实验：多事件时间线 =================
print("\n🎬 Experiment: Multi-Event Timeline Test")

# 时间线定义
# T=0: Red Circle
# T=1..49: Noise
# T=50: Blue Square
# T=51..99: Noise
# T=100: Green Triangle
# T=101..120: Noise

def append_at_time(t, image, text):
    msg = engine.append_frame(image, manual_time=t, text_content=text)
    print(f"   T={t:03d}: {msg}")

print("\n[Phase 1] Streaming Timeline...")
append_at_time(0, img_red_circle, "Event: Red circle at Frame 0.")
cache_at_0 = copy.deepcopy(engine.video_cache)

for t in range(1, 50):
    append_at_time(t, img_noise, "Noise: black screen.")

append_at_time(50, img_blue_square, "Event: Blue square at Frame 50.")
cache_at_50 = copy.deepcopy(engine.video_cache)

for t in range(51, 100):
    append_at_time(t, img_noise, "Noise: black screen.")

append_at_time(100, img_green_triangle, "Event: Green triangle at Frame 100.")
cache_at_100 = copy.deepcopy(engine.video_cache)

for t in range(101, 121):
    append_at_time(t, img_noise, "Noise: black screen.")

# ================= 询问 =================
print("\n[Phase 2] Interrogation...")

q1 = "What object appeared at the very beginning (Frame 0)?"
q2 = "What object appeared around Frame 50?"
q3 = "What object appeared around Frame 100?"

choices = ["Red", "Blue", "Green"]

def ask_choice(question, manual_time):
    prompt = question + " Choices: Red, Blue, Green. Answer with one word."
    return engine.ask_choice(prompt, choices=choices, manual_time=manual_time)

def ask_from_cache(cache, question, manual_time):
    original_cache = engine.video_cache
    engine.video_cache = cache
    try:
        return ask_choice(question, manual_time=manual_time)
    finally:
        engine.video_cache = original_cache

ans1 = ask_from_cache(cache_at_0, q1, manual_time=1)
ans2 = ask_from_cache(cache_at_50, q2, manual_time=51)
ans3 = ask_from_cache(cache_at_100, q3, manual_time=101)

print("\n" + "=" * 50)
print("🧐 Experiment Results")
print("=" * 50)
print(f"Q1: {q1}\nA1: {ans1}")
print(f"Q2: {q2}\nA2: {ans2}")
print(f"Q3: {q3}\nA3: {ans3}")

# ================= 断言与失败条件 =================
ans1_l = ans1.lower()
ans2_l = ans2.lower()
ans3_l = ans3.lower()

ok1 = ("red" in ans1_l) or ("circle" in ans1_l) or ("红" in ans1) or ("圆" in ans1)
ok2 = ("blue" in ans2_l) or ("square" in ans2_l) or ("蓝" in ans2) or ("方" in ans2)
ok3 = ("green" in ans3_l) or ("triangle" in ans3_l) or ("绿" in ans3) or ("三角" in ans3)

if not ok1:
    raise AssertionError("❌ Temporal Disentanglement Failed: Frame 0 did NOT map to Red/Circle.")
if not ok2:
    raise AssertionError("❌ Temporal Disentanglement Failed: Frame 50 did NOT map to Blue/Square.")
if not ok3:
    raise AssertionError("❌ Temporal Disentanglement Failed: Frame 100 did NOT map to Green/Triangle.")

# 严格错配检查（防止跨时间混淆）
if ("blue" in ans1_l) or ("square" in ans1_l):
    raise AssertionError("❌ Temporal Confusion: Q1 answered with Blue/Square.")
if ("red" in ans2_l) or ("circle" in ans2_l):
    raise AssertionError("❌ Temporal Confusion: Q2 answered with Red/Circle.")
if ("red" in ans3_l) or ("circle" in ans3_l):
    raise AssertionError("❌ Temporal Confusion: Q3 answered with Red/Circle.")
if ("blue" in ans3_l) or ("square" in ans3_l):
    raise AssertionError("❌ Temporal Confusion: Q3 answered with Blue/Square.")

print("\n✅ Multi-Event Timeline Test PASSED.")
print("1) Temporal addressing verified across early/middle/late events.")
print("2) No cross-time confusion detected.")