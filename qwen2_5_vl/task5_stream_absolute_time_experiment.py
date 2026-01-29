import os
import sys
import traceback
from typing import List, Tuple

import cv2
import torch
from PIL import Image
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL_PATH = "/root/autodl-tmp/Qwen/Qwen2___5-VL-3B-Instruct"
VIDEO_PATH = "/root/autodl-tmp/temporal_encoding/202208312002.mp4"
ALT_VIDEO_PATH = "/root/autodl-tmp/diploma/temporal_encoding/202208312002.mp4"
DEVICE = "cuda"

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
REPORT_PATH = os.path.join(OUTPUT_DIR, "task5_stream_absolute_time_report.txt")


def _ensure_stream_import():
    current_dir = os.path.dirname(__file__)
    candidate_paths = [
        os.path.join(current_dir, "..", "temporal_encoding", "model"),
        "/root/autodl-tmp/diploma/temporal_encoding/model",
    ]
    for path in candidate_paths:
        abs_path = os.path.abspath(path)
        if os.path.isdir(abs_path) and abs_path not in sys.path:
            sys.path.insert(0, abs_path)

    from stream_qwen_model import StreamQwenModel  # type: ignore
    from video_stream_inference import VideoStreamingInference  # type: ignore

    return StreamQwenModel, VideoStreamingInference


def _resolve_video_path() -> str:
    if os.path.isfile(VIDEO_PATH):
        return VIDEO_PATH
    if os.path.isfile(ALT_VIDEO_PATH):
        return ALT_VIDEO_PATH
    raise FileNotFoundError(f"Video not found: {VIDEO_PATH} or {ALT_VIDEO_PATH}")


def extract_frames_percent(video_path: str, percents: List[float]) -> Tuple[List[Image.Image], List[float], float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frames: List[Image.Image] = []
    timestamps: List[float] = []
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if fps <= 0 or frame_count <= 0:
            raise RuntimeError("Invalid video metadata (fps or frame count).")
        duration = frame_count / fps

        for p in percents:
            ts = max(0.0, min(duration, duration * p))
            cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000.0)
            success, frame = cap.read()
            if not success or frame is None:
                raise RuntimeError(f"Failed to read frame at {ts:.2f}s")
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
            timestamps.append(ts)
    finally:
        cap.release()

    return frames, timestamps, duration


def _normalize_position_ids(position_ids: torch.Tensor) -> torch.Tensor:
    if position_ids.dim() == 3 and position_ids.shape[0] == 1:
        position_ids = position_ids[0]
    if position_ids.dim() == 3 and position_ids.shape[0] == 3 and position_ids.shape[1] == 1:
        return position_ids[:, 0, :]
    if position_ids.dim() == 2:
        if position_ids.shape[0] == 3:
            return position_ids
        if position_ids.shape[1] == 3:
            return position_ids.transpose(0, 1)
    raise RuntimeError(f"Unsupported position_ids shape: {tuple(position_ids.shape)}")


def _get_text_len_before_video(input_ids: List[int], vision_start_id: int, video_token_id: int) -> int:
    for idx, tid in enumerate(input_ids):
        if tid == vision_start_id and idx + 1 < len(input_ids) and input_ids[idx + 1] == video_token_id:
            return idx + 1
    for idx, tid in enumerate(input_ids):
        if tid == video_token_id:
            return idx
    raise RuntimeError("Unable to determine text length before video tokens.")


def _get_text_len_before_image(input_ids: List[int], vision_start_id: int, image_token_id: int) -> int:
    for idx, tid in enumerate(input_ids):
        if tid == vision_start_id and idx + 1 < len(input_ids) and input_ids[idx + 1] == image_token_id:
            return idx + 1
    for idx, tid in enumerate(input_ids):
        if tid == image_token_id:
            return idx
    raise RuntimeError("Unable to determine text length before image tokens.")


def _find_video_span(input_ids: List[int], vision_start_id: int, video_token_id: int) -> Tuple[int, int]:
    for idx, tid in enumerate(input_ids):
        if tid == vision_start_id and idx + 1 < len(input_ids) and input_ids[idx + 1] == video_token_id:
            start = idx + 1
            end = start + 1
            while end < len(input_ids) and input_ids[end] == video_token_id:
                end += 1
            return start, end
    for idx, tid in enumerate(input_ids):
        if tid == video_token_id:
            start = idx
            end = start + 1
            while end < len(input_ids) and input_ids[end] == video_token_id:
                end += 1
            return start, end
    raise RuntimeError("Unable to locate video token span.")


def _find_image_span(input_ids: List[int], vision_start_id: int, image_token_id: int) -> Tuple[int, int]:
    for idx, tid in enumerate(input_ids):
        if tid == vision_start_id and idx + 1 < len(input_ids) and input_ids[idx + 1] == image_token_id:
            start = idx + 1
            end = start + 1
            while end < len(input_ids) and input_ids[end] == image_token_id:
                end += 1
            return start, end
    for idx, tid in enumerate(input_ids):
        if tid == image_token_id:
            start = idx
            end = start + 1
            while end < len(input_ids) and input_ids[end] == image_token_id:
                end += 1
            return start, end
    raise RuntimeError("Unable to locate image token span.")


def _span_ranges(position_ids: torch.Tensor, span: Tuple[int, int]) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    position_ids = _normalize_position_ids(position_ids)
    start, end = span
    t_ids = position_ids[0, start:end]
    h_ids = position_ids[1, start:end]
    w_ids = position_ids[2, start:end]
    return (int(t_ids.min().item()), int(t_ids.max().item())), (int(h_ids.min().item()), int(h_ids.max().item())), (int(w_ids.min().item()), int(w_ids.max().item()))


def _format_ranges(label: str, ranges: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]) -> str:
    t_range, h_range, w_range = ranges
    return f"{label} | T: {t_range[0]}-{t_range[1]} | H: {h_range[0]}-{h_range[1]} | W: {w_range[0]}-{w_range[1]}"


def main() -> int:
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        video_path = _resolve_video_path()

        StreamQwenModel, VideoStreamingInference = _ensure_stream_import()

        processor = AutoProcessor.from_pretrained(
            MODEL_PATH,
            min_pixels=256 * 256,
            max_pixels=1024 * 1024,
        )
        model = StreamQwenModel.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model.eval()

        # Native video inputs (fps=1.0)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path},
                    {"type": "text", "text": "Describe the video."},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        try:
            native_inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                fps=1.0,
                padding=True,
                return_tensors="pt",
            )
        except TypeError:
            native_inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )

        input_ids = native_inputs["input_ids"]
        attention_mask = native_inputs.get("attention_mask")
        video_grid_thw = native_inputs.get("video_grid_thw")
        second_per_grid_ts = native_inputs.get("second_per_grid_ts")
        if video_grid_thw is None:
            raise RuntimeError("video_grid_thw missing from native inputs.")
        grid = video_grid_thw
        if grid.dim() == 3 and grid.shape[0] == 1:
            grid = grid[0]
        temporal_patches = int(grid[0, 0].item())
        height_patches = int(grid[0, 1].item())
        width_patches = int(grid[0, 2].item())

        position_ids, _ = model.model.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=None,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            attention_mask=attention_mask,
        )

        input_ids_list = input_ids[0].tolist()
        video_span = _find_video_span(
            input_ids_list,
            model.config.vision_start_token_id,
            model.config.video_token_id,
        )

        spatial_merge = model.config.vision_config.spatial_merge_size
        llm_grid_h = height_patches // spatial_merge
        llm_grid_w = width_patches // spatial_merge
        tokens_per_t = llm_grid_h * llm_grid_w

        fractions = [0.1, 0.5, 0.9]
        frames, frame_times, duration = extract_frames_percent(video_path, fractions)

        native_ranges = []
        for frac in fractions:
            t_index = int(round((temporal_patches - 1) * frac))
            start = video_span[0] + t_index * tokens_per_t
            end = start + tokens_per_t
            native_ranges.append(_span_ranges(position_ids, (start, end)))

        native_text_len = _get_text_len_before_video(
            input_ids_list,
            model.config.vision_start_token_id,
            model.config.video_token_id,
        )

        streamer = VideoStreamingInference(model, processor, device=DEVICE)
        streamer.set_video_meta(
            duration_sec=duration,
            temporal_patches=temporal_patches,
            native_fps=1.0,
            text_len=native_text_len,
        )

        report_lines: List[str] = ["Stream Absolute Time Experiment"]
        report_lines.append(f"temporal_patches={temporal_patches}, tokens_per_t={tokens_per_t}")
        report_lines.append("")
        report_lines.append("Native Video (reference)")
        for idx, ranges in enumerate(native_ranges, start=1):
            report_lines.append(_format_ranges(f"Sample {idx}", ranges))

        report_lines.append("")
        report_lines.append("Streaming Single-Frame (manual_time)")

        stream_ranges = []
        for idx, (img, ts, frac) in enumerate(zip(frames, frame_times, fractions), start=1):
            msg = streamer.append_frame(img, frame_time_sec=ts, frame_frac=frac)
            manual_time = streamer.last_manual_time

            image_prompt = "Analyze this image: <image>"
            image_inputs = processor(
                text=[processor.apply_chat_template([
                    {"role": "user", "content": [{"type": "text", "text": image_prompt}, {"type": "image", "image": img}]}
                ], tokenize=False, add_generation_prompt=True)],
                images=[img],
                padding=True,
                return_tensors="pt",
            )
            image_inputs = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in image_inputs.items()}
            img_position_ids, _ = model.model.get_rope_index(
                input_ids=image_inputs["input_ids"],
                image_grid_thw=image_inputs.get("image_grid_thw"),
                video_grid_thw=None,
                attention_mask=image_inputs.get("attention_mask"),
            )
            img_position_ids = model._apply_time_shift(
                img_position_ids,
                image_inputs["input_ids"],
                int(manual_time),
            )
            img_span = _find_image_span(
                image_inputs["input_ids"][0].tolist(),
                model.config.vision_start_token_id,
                model.config.image_token_id,
            )
            ranges = _span_ranges(img_position_ids, img_span)
            stream_ranges.append(ranges)
            report_lines.append(_format_ranges(f"Frame {idx} (manual_time={manual_time})", ranges))

        report_lines.append("")
        report_lines.append("Comparison (T only)")
        all_t_match = True
        for idx, (native_r, stream_r) in enumerate(zip(native_ranges, stream_ranges), start=1):
            t_match = native_r[0] == stream_r[0]
            all_t_match = all_t_match and t_match
            report_lines.append(
                f"Sample {idx} | Native T: {native_r[0][0]}-{native_r[0][1]} | "
                f"Stream T: {stream_r[0][0]}-{stream_r[0][1]} | match={t_match}"
            )

        report_lines.append("")
        report_lines.append("Analysis Summary")
        if all_t_match:
            report_lines.append("Absolute time encoding matched native video on T axis.")
        else:
            report_lines.append("Absolute time encoding did not fully match native video on T axis.")

        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines) + "\n")

        print(f"Saved report to: {REPORT_PATH}")
        return 0
    except Exception as exc:
        print("Task 5 failed:")
        print(str(exc))
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
