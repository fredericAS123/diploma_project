import os
import sys
import traceback
from typing import List, Tuple, Dict

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
REPORT_PATH = os.path.join(OUTPUT_DIR, "task3_mrope_report.txt")


def _ensure_stream_model_import():
    current_dir = os.path.dirname(__file__)
    candidate_paths = [
        current_dir,
        os.path.join(current_dir, "..", "temporal_encoding", "model"),
        "/root/autodl-tmp/diploma/temporal_encoding/model",
    ]
    for path in candidate_paths:
        abs_path = os.path.abspath(path)
        if os.path.isdir(abs_path) and abs_path not in sys.path:
            sys.path.insert(0, abs_path)

    try:
        from stream_qwen_model import StreamQwenModel  # type: ignore
        return StreamQwenModel
    except Exception as exc:
        raise RuntimeError("Failed to import StreamQwenModel.") from exc


def _resolve_video_path() -> str:
    if os.path.isfile(VIDEO_PATH):
        return VIDEO_PATH
    if os.path.isfile(ALT_VIDEO_PATH):
        return ALT_VIDEO_PATH
    raise FileNotFoundError(f"Video not found: {VIDEO_PATH} or {ALT_VIDEO_PATH}")


def extract_frames_percent(video_path: str, percents: List[float]) -> List[Image.Image]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frames: List[Image.Image] = []
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
    finally:
        cap.release()

    return frames


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


def _get_position_ids(model, model_inputs: dict, manual_time: int | None = None) -> torch.Tensor:
    input_ids = model_inputs.get("input_ids")
    attention_mask = model_inputs.get("attention_mask")
    image_grid_thw = model_inputs.get("image_grid_thw")
    video_grid_thw = model_inputs.get("video_grid_thw")
    second_per_grid_ts = model_inputs.get("second_per_grid_ts")

    backbone = getattr(model, "model", None)
    if backbone is None or not hasattr(backbone, "get_rope_index"):
        raise RuntimeError("Model backbone does not support get_rope_index.")

    position_ids, _ = backbone.get_rope_index(
        input_ids=input_ids,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        second_per_grid_ts=second_per_grid_ts,
        attention_mask=attention_mask,
    )

    if manual_time is not None:
        position_ids = model._apply_time_shift(position_ids, input_ids, int(manual_time))

    return position_ids


def _get_image_grid_thw(model_inputs: dict) -> torch.Tensor:
    grid = model_inputs.get("image_grid_thw")
    if grid is None:
        raise RuntimeError("image_grid_thw not found in model inputs.")
    if grid.dim() == 3 and grid.shape[0] == 1:
        grid = grid[0]
    return grid


def _find_image_token_spans(
    input_ids: List[int],
    image_grid_thw: torch.Tensor,
    vision_start_id: int,
    image_token_id: int,
) -> List[Tuple[int, int]]:
    num_images = int(image_grid_thw.shape[0])
    spans: List[Tuple[int, int]] = []

    if vision_start_id is not None:
        start_positions = [i for i, tid in enumerate(input_ids) if tid == vision_start_id]
        for pos in start_positions:
            start = pos + 1
            if start >= len(input_ids):
                continue
            if input_ids[start] != image_token_id:
                continue
            end = start + 1
            while end < len(input_ids) and input_ids[end] == image_token_id:
                end += 1
            spans.append((start, end))
        if len(spans) >= num_images:
            return spans[:num_images]

    runs: List[Tuple[int, int]] = []
    i = 0
    while i < len(input_ids):
        if input_ids[i] == image_token_id:
            start = i
            i += 1
            while i < len(input_ids) and input_ids[i] == image_token_id:
                i += 1
            runs.append((start, i))
        else:
            i += 1
    if len(runs) >= num_images:
        return runs[:num_images]

    if spans:
        return spans

    raise RuntimeError("Unable to locate image token spans in input_ids.")


def _span_ranges(position_ids: torch.Tensor, span: Tuple[int, int]) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    position_ids = _normalize_position_ids(position_ids)
    start, end = span
    t_ids = position_ids[0, start:end]
    h_ids = position_ids[1, start:end]
    w_ids = position_ids[2, start:end]
    t_range = (int(t_ids.min().item()), int(t_ids.max().item()))
    h_range = (int(h_ids.min().item()), int(h_ids.max().item()))
    w_range = (int(w_ids.min().item()), int(w_ids.max().item()))
    return t_range, h_range, w_range


def _format_ranges(label: str, ranges: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]) -> str:
    t_range, h_range, w_range = ranges
    return (
        f"{label} | T: {t_range[0]}-{t_range[1]} | "
        f"H: {h_range[0]}-{h_range[1]} | W: {w_range[0]}-{w_range[1]}"
    )


def _build_inputs(processor, images: List[Image.Image], prompt: str) -> dict:
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
            + [{"type": "image", "image": img} for img in images],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    model_inputs = processor(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )
    return model_inputs


def _to_device(model_inputs: dict, device: str) -> dict:
    for key, value in list(model_inputs.items()):
        if isinstance(value, torch.Tensor):
            model_inputs[key] = value.to(device)
    return model_inputs


def _compare_ranges(
    a_ranges: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]],
    b_ranges: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]],
) -> str:
    (_, a_h, a_w) = (a_ranges[0], a_ranges[1], a_ranges[2])
    (_, b_h, b_w) = (b_ranges[0], b_ranges[1], b_ranges[2])

    if a_h == b_h and a_w == b_w:
        return "H/W match (full shift)"

    if b_h[1] < a_h[0] or b_w[1] < a_w[0]:
        return "H/W reset detected"

    if (a_h[0] - b_h[0]) > 10 and (a_w[0] - b_w[0]) > 10:
        return "H/W reset likely"

    return "H/W mismatch (partial shift)"


def main() -> int:
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        video_path = _resolve_video_path()
        frames = extract_frames_percent(video_path, [0.1, 0.5, 0.9])

        StreamQwenModel = _ensure_stream_model_import()

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

        report_lines: List[str] = []
        mode_a_ranges: Dict[int, Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]] = {}
        mode_b_ranges: Dict[int, Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]] = {}
        t_start_map: Dict[int, int] = {}

        prompt_multi = "Analyze these three images: <image> <image> <image>"
        model_inputs_multi = _build_inputs(processor, frames, prompt_multi)
        model_inputs_multi = _to_device(model_inputs_multi, DEVICE)
        with torch.no_grad():
            _ = model.forward(**model_inputs_multi)
        position_ids_multi = _get_position_ids(model, model_inputs_multi)

        image_grid_thw_multi = _get_image_grid_thw(model_inputs_multi)
        input_ids_multi = model_inputs_multi["input_ids"][0].tolist()
        spans_multi = _find_image_token_spans(
            input_ids_multi,
            image_grid_thw_multi,
            model.config.vision_start_token_id,
            model.config.image_token_id,
        )

        report_lines.append("Mode A: Multi-Image Reference")
        for idx, span in enumerate(spans_multi, start=1):
            ranges = _span_ranges(position_ids_multi, span)
            mode_a_ranges[idx] = ranges
            t_start_map[idx] = ranges[0][0]
            report_lines.append(_format_ranges(f"Image {idx}", ranges))

        report_lines.append("")

        report_lines.append("Mode B: Streaming Simulation (manual_time)")
        for idx, img in enumerate(frames, start=1):
            prompt_single = "Analyze this image: <image>"
            model_inputs_single = _build_inputs(processor, [img], prompt_single)
            model_inputs_single = _to_device(model_inputs_single, DEVICE)
            manual_time = t_start_map[idx]
            with torch.no_grad():
                _ = model.forward(**model_inputs_single, manual_time=manual_time)
            position_ids_single = _get_position_ids(model, model_inputs_single, manual_time=manual_time)
            image_grid_thw_single = _get_image_grid_thw(model_inputs_single)
            input_ids_single = model_inputs_single["input_ids"][0].tolist()
            spans_single = _find_image_token_spans(
                input_ids_single,
                image_grid_thw_single,
                model.config.vision_start_token_id,
                model.config.image_token_id,
            )
            if not spans_single:
                raise RuntimeError("No image token span found for single-image input.")
            ranges = _span_ranges(position_ids_single, spans_single[0])
            mode_b_ranges[idx] = ranges
            report_lines.append(_format_ranges(f"Image {idx}", ranges))

        report_lines.append("")
        report_lines.append("Comparison")
        for idx in mode_a_ranges:
            a_ranges = mode_a_ranges[idx]
            b_ranges = mode_b_ranges.get(idx)
            if b_ranges is None:
                report_lines.append(f"Image {idx} | missing Mode B data")
                continue
            verdict = _compare_ranges(a_ranges, b_ranges)
            report_lines.append(
                f"Image {idx} | Mode A H/W: {a_ranges[1][0]}-{a_ranges[1][1]}, {a_ranges[2][0]}-{a_ranges[2][1]} | "
                f"Mode B H/W: {b_ranges[1][0]}-{b_ranges[1][1]}, {b_ranges[2][0]}-{b_ranges[2][1]} | {verdict}"
            )

        report_lines.append("")
        report_lines.append("Analysis Summary")
        reset_flags = [
            _compare_ranges(mode_a_ranges[i], mode_b_ranges[i]).startswith("H/W reset")
            for i in mode_a_ranges
            if i in mode_b_ranges
        ]
        if reset_flags and any(reset_flags):
            summary = (
                "Detected Spatial Reset: Mode B H/W ranges are significantly smaller than Mode A. "
                "This suggests StreamQwenModel applies time shift without fully shifting spatial indices, "
                "so the 3D-RoPE behavior does not fully match the multi-image reference."
            )
        else:
            summary = (
                "No Spatial Reset: Mode B H/W ranges match Mode A, indicating full 3D-RoPE alignment "
                "when using manual_time in StreamQwenModel."
            )
        report_lines.append(summary)

        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines) + "\n")

        if not os.path.isfile(REPORT_PATH):
            print(f"Report file was not saved: {REPORT_PATH}")
            return 1

        print(f"Saved report to: {REPORT_PATH}")
        return 0
    except Exception as exc:
        print("Task 3 failed:")
        print(str(exc))
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
