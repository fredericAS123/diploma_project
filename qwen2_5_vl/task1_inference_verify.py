import os
import sys
import traceback

import cv2
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

MODEL_PATH = "/root/autodl-tmp/Qwen/Qwen2___5-VL-3B-Instruct"
VIDEO_PATH = "/root/autodl-tmp/diploma/temporal_encoding/202208312002.mp4"
DEVICE = "cuda"

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
FRAME_PATH = os.path.join(OUTPUT_DIR, "verification_frame_5s.jpg")
RESULT_PATH = os.path.join(OUTPUT_DIR, "task1_result.txt")

PROMPT = "这位选手的名字？她属于哪位导师的战队？请直接说出答案，不要添加其他描述性语言。"


def extract_frame_at_time(video_path: str, time_sec: float, output_path: str) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000.0)
        success, frame = cap.read()
        if not success or frame is None:
            raise RuntimeError(f"Failed to read frame at {time_sec}s")
        ok = cv2.imwrite(output_path, frame)
        if not ok:
            raise RuntimeError(f"Failed to save frame to: {output_path}")
    finally:
        cap.release()


def main() -> int:
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        if not os.path.isfile(VIDEO_PATH):
            print(f"Video not found: {VIDEO_PATH}")
            return 1

        extract_frame_at_time(VIDEO_PATH, 5.0, FRAME_PATH)
        if not os.path.isfile(FRAME_PATH):
            print(f"Frame was not saved: {FRAME_PATH}")
            return 1

        processor = AutoProcessor.from_pretrained(
            MODEL_PATH,
            min_pixels=256 * 256,
            max_pixels=1024 * 1024,
        )
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model.eval()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": VIDEO_PATH},
                    {"type": "text", "text": PROMPT},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(DEVICE)

        with torch.no_grad():
            generate_ids = model.generate(**inputs, max_new_tokens=256)

        input_len = inputs["input_ids"].shape[1]
        response = processor.batch_decode(
            generate_ids[:, input_len:],
            skip_special_tokens=True,
        )[0]

        print(response)
        with open(RESULT_PATH, "w", encoding="utf-8") as f:
            f.write(response.strip() + "\n")

        if not os.path.isfile(RESULT_PATH):
            print(f"Result file was not saved: {RESULT_PATH}")
            return 1

        return 0
    except Exception as exc:
        print("Task 1 failed:")
        print(str(exc))
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
