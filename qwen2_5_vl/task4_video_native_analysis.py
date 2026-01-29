import os
import sys
import traceback
from typing import List, Tuple

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

MODEL_PATH = "/root/autodl-tmp/Qwen/Qwen2___5-VL-3B-Instruct"
VIDEO_PATH = "/root/autodl-tmp/temporal_encoding/202208312002.mp4"
ALT_VIDEO_PATH = "/root/autodl-tmp/diploma/temporal_encoding/202208312002.mp4"
DEVICE = "cuda"

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
REPORT_PATH = os.path.join(OUTPUT_DIR, "task4_mrope_report.txt")


def _resolve_video_path() -> str:
    if os.path.isfile(VIDEO_PATH):
        return VIDEO_PATH
    if os.path.isfile(ALT_VIDEO_PATH):
        return ALT_VIDEO_PATH
    raise FileNotFoundError(f"Video not found: {VIDEO_PATH} or {ALT_VIDEO_PATH}")


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


def _find_video_token_span(
    input_ids: List[int],
    vision_start_id: int,
    video_token_id: int,
) -> Tuple[int, int]:
    start_positions = [i for i, tid in enumerate(input_ids) if tid == vision_start_id]
    for pos in start_positions:
        start = pos + 1
        if start >= len(input_ids):
            continue
        if input_ids[start] != video_token_id:
            continue
        end = start + 1
        while end < len(input_ids) and input_ids[end] == video_token_id:
            end += 1
        return start, end

    i = 0
    while i < len(input_ids):
        if input_ids[i] == video_token_id:
            start = i
            i += 1
            while i < len(input_ids) and input_ids[i] == video_token_id:
                i += 1
            return start, i
        i += 1

    raise RuntimeError("Unable to locate video token span in input_ids.")


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


def main() -> int:
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        video_path = _resolve_video_path()

        processor = AutoProcessor.from_pretrained(
            MODEL_PATH,
            min_pixels=256 * 256,
            max_pixels=1024 * 1024,
        )
        try:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                MODEL_PATH,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
            )
            print("flash_attention_2 enabled.")
        except ImportError:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                MODEL_PATH,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            print("flash_attention_2 unavailable; falling back to default attention.")
        model.eval()

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

        inputs = None
        used_fps = False
        try:
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                fps=1.0,
                padding=True,
                return_tensors="pt",
            )
            used_fps = True
        except TypeError:
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )

        if inputs is None:
            raise RuntimeError("Failed to build model inputs for video.")

        if "pixel_values_videos" in inputs:
            print(f"pixel_values_videos shape: {tuple(inputs['pixel_values_videos'].shape)}")
        if "video_grid_thw" in inputs:
            print(f"video_grid_thw: {inputs['video_grid_thw'].tolist()}")
        print(f"fps=1.0 applied: {used_fps}")

        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")
        video_grid_thw = inputs.get("video_grid_thw")
        second_per_grid_ts = inputs.get("second_per_grid_ts")

        if input_ids is None or video_grid_thw is None:
            raise RuntimeError("Missing input_ids or video_grid_thw in model inputs.")

        position_ids, _ = model.model.get_rope_index(
            input_ids=input_ids,
            video_grid_thw=video_grid_thw,
            image_grid_thw=None,
            second_per_grid_ts=second_per_grid_ts,
            attention_mask=attention_mask,
        )

        input_id_list = input_ids[0].tolist()
        video_span = _find_video_token_span(
            input_id_list,
            model.config.vision_start_token_id,
            model.config.video_token_id,
        )

        grid = video_grid_thw
        if grid.dim() == 3 and grid.shape[0] == 1:
            grid = grid[0]
        if grid.dim() != 2 or grid.shape[0] < 1:
            raise RuntimeError(f"Unexpected video_grid_thw shape: {tuple(video_grid_thw.shape)}")

        temporal_patches = int(grid[0, 0].item())
        height_patches = int(grid[0, 1].item())
        width_patches = int(grid[0, 2].item())

        spatial_merge = model.config.vision_config.spatial_merge_size
        llm_grid_h = height_patches // spatial_merge
        llm_grid_w = width_patches // spatial_merge
        tokens_per_t = llm_grid_h * llm_grid_w

        print(f"temporal_patches={temporal_patches}, tokens_per_t={tokens_per_t}")

        fractions = [0.1, 0.5, 0.9]
        report_lines: List[str] = ["Native Video Analysis"]

        for idx, frac in enumerate(fractions, start=1):
            t_index = int(round((temporal_patches - 1) * frac))
            t_index = max(0, min(temporal_patches - 1, t_index))
            start = video_span[0] + t_index * tokens_per_t
            end = start + tokens_per_t
            ranges = _span_ranges(position_ids, (start, end))
            report_lines.append(_format_ranges(f"Sample {idx} (approx {int(frac*100)}%)", ranges))

        report_lines.append("")
        report_lines.append("Analysis Summary")
        t_values = []
        h_values = []
        w_values = []
        for line in report_lines:
            if line.startswith("Sample"):
                parts = line.split("|")
                t_range = parts[1].strip().split(":")[1].strip().split("-")
                h_range = parts[2].strip().split(":")[1].strip().split("-")
                w_range = parts[3].strip().split(":")[1].strip().split("-")
                t_values.append((int(t_range[0]), int(t_range[1])))
                h_values.append((int(h_range[0]), int(h_range[1])))
                w_values.append((int(w_range[0]), int(w_range[1])))

        is_monotonic = (
            t_values[0][0] <= t_values[1][0] <= t_values[2][0]
            and h_values[0][0] <= h_values[1][0] <= h_values[2][0]
            and w_values[0][0] <= w_values[1][0] <= w_values[2][0]
        )

        if is_monotonic:
            report_lines.append(
                "Global Monotonic Increase observed: T/H/W indices increase across time, matching native behavior."
            )
        else:
            report_lines.append(
                "Non-monotonic behavior detected: T/H/W indices do not strictly increase across time."
            )

        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines) + "\n")

        if not os.path.isfile(REPORT_PATH):
            print(f"Report file was not saved: {REPORT_PATH}")
            return 1

        print(f"Saved report to: {REPORT_PATH}")
        return 0
    except Exception as exc:
        print("Task 4 failed:")
        print(str(exc))
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
