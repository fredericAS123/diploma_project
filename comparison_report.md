# StreamUAV-QA Experiment Comparison

## Overview

- Generated at: 2026-03-21T20:19:54+0800
- Output root: /root/autodl-tmp/streamVQA/datasets/stream_uav_qa_experiment_comparison

- Note: the `fps=4, chunk=4` row comes from the completed rescue run with `max_cache_tokens=130000`.

## Params: streaming_fps0_25

- kv_cache_config.max_cache_tokens: 150000
- l1_config.chunk_size: 1
- l1_config.dataset_json: /root/autodl-tmp/streamVQA/datasets/stream_uav_qa_correct_pool_mode_split/stream_uav_eval_l1_single_frame.json
- l1_config.fps: 1.0
- l1_config.mode: single_frame
- l23_config.chunk_sizes: 2, 4
- l23_config.dataset_json: /root/autodl-tmp/streamVQA/datasets/stream_uav_qa_correct_pool_mode_split/stream_uav_eval_l23_history_prefix.json
- l23_config.fps: 0.25
- l23_config.mode: history_prefix
- model_path: /root/autodl-tmp/Qwen/Qwen2___5-VL-3B-Instruct
- output_root: /root/autodl-tmp/streamVQA/datasets/stream_uav_qa_correct_pool_stream3b_eval
- sequences_root: /root/autodl-tmp/streamVQA/VisDrone2019-VID-train/VisDrone2019-VID-train/sequences
- ttft_source: choice_prefill

## Params: streaming_fps1_0

- evaluation_type: streaming_3b
- experiment_name: stream_uav_qa_correct_pool_stream3b_eval_fps1
- kv_cache_config.max_cache_tokens: 150000
- l1_config.append_frame_as_video: True
- l1_config.chunk_size: 1
- l1_config.dataset_json: /root/autodl-tmp/streamVQA/datasets/stream_uav_qa_correct_pool_mode_split/stream_uav_eval_l1_single_frame.json
- l1_config.fps: 1.0
- l1_config.fps_semantics: append_frame_video_fps
- l1_config.input_mode: video
- l1_config.mode: single_frame_video
- l23_config.chunk_sizes: 2, 4
- l23_config.dataset_json: /root/autodl-tmp/streamVQA/datasets/stream_uav_qa_correct_pool_mode_split/stream_uav_eval_l23_history_prefix.json
- l23_config.fps: 1.0
- l23_config.fps_semantics: append_video_chunk_fps
- l23_config.mode: history_prefix
- model_path: /root/autodl-tmp/Qwen/Qwen2___5-VL-3B-Instruct
- output_root: /root/autodl-tmp/streamVQA/datasets/stream_uav_qa_correct_pool_stream3b_eval_fps1
- sequences_root: /root/autodl-tmp/streamVQA/VisDrone2019-VID-train/VisDrone2019-VID-train/sequences
- ttft_source: choice_prefill

## Params: streaming_fps4_0

- branch.branch_name: l1_single_frame_video_chunk1
- branch.chunk_size: 1
- branch.dataset_json: /root/autodl-tmp/streamVQA/datasets/stream_uav_qa_correct_pool_mode_split/stream_uav_eval_l1_single_frame.json
- branch.fps: 4.0
- branch.l1_input_mode: video
- branch.mode: single_frame_video
- branch.output_dir: /root/autodl-tmp/streamVQA/datasets/stream_uav_qa_correct_pool_stream3b_eval_fps4/l1_single_frame_video_chunk1
- evaluation_type: streaming_3b
- experiment_name: stream_uav_qa_correct_pool_stream3b_eval_fps4
- kv_cache_config.max_cache_tokens: 150000
- l1_config.append_frame_as_video: True
- l1_config.chunk_size: 1
- l1_config.dataset_json: /root/autodl-tmp/streamVQA/datasets/stream_uav_qa_correct_pool_mode_split/stream_uav_eval_l1_single_frame.json
- l1_config.fps: 4.0
- l1_config.fps_semantics: append_frame_video_fps
- l1_config.input_mode: video
- l1_config.mode: single_frame_video
- l23_config.chunk_sizes: 2, 4
- l23_config.dataset_json: /root/autodl-tmp/streamVQA/datasets/stream_uav_qa_correct_pool_mode_split/stream_uav_eval_l23_history_prefix.json
- l23_config.fps: 4.0
- l23_config.fps_semantics: append_video_chunk_fps
- l23_config.mode: history_prefix
- model_path: /root/autodl-tmp/Qwen/Qwen2___5-VL-3B-Instruct
- output_root: /root/autodl-tmp/streamVQA/datasets/stream_uav_qa_correct_pool_stream3b_eval_fps4
- sequences_root: /root/autodl-tmp/streamVQA/VisDrone2019-VID-train/VisDrone2019-VID-train/sequences
- ttft_source: choice_prefill

## Params: streaming_fps4_chunk4_rescue

- evaluation_type: streaming_3b
- experiment_name: stream_uav_qa_correct_pool_stream3b_eval_fps4_chunk4_resume130k_v2
- kv_cache_config.max_cache_tokens: 130000
- l1_config.append_frame_as_video: True
- l1_config.chunk_size: 1
- l1_config.dataset_json: /root/autodl-tmp/streamVQA/datasets/stream_uav_qa_correct_pool_mode_split/stream_uav_eval_l1_single_frame.json
- l1_config.fps: 4.0
- l1_config.fps_semantics: append_frame_video_fps
- l1_config.input_mode: video
- l1_config.mode: single_frame_video
- l23_config.chunk_sizes: 4
- l23_config.dataset_json: /root/autodl-tmp/streamVQA/datasets/stream_uav_qa_correct_pool_mode_split/stream_uav_eval_l23_history_prefix.json
- l23_config.fps: 4.0
- l23_config.fps_semantics: append_video_chunk_fps
- l23_config.mode: history_prefix
- model_path: /root/autodl-tmp/Qwen/Qwen2___5-VL-3B-Instruct
- output_root: /root/autodl-tmp/streamVQA/datasets/stream_uav_qa_correct_pool_stream3b_eval_fps4_chunk4_resume130k_v2
- sequences_root: /root/autodl-tmp/streamVQA/VisDrone2019-VID-train/VisDrone2019-VID-train/sequences
- ttft_source: choice_prefill

## Params: native_single_frame

- baseline_config.choice_scoring: option_letter_logprob
- baseline_config.input_mode: single_image_current_frame_only
- baseline_config.query_frame_selection: timestamp_clamp_to_current_frame
- evaluation_type: qwen2_5_vl_native_single_frame
- experiment_name: stream_uav_qa_correct_pool_qwen25vl_single_frame_eval
- l1_config.chunk_size: 1
- l1_config.dataset_json: /root/autodl-tmp/streamVQA/datasets/stream_uav_qa_correct_pool_mode_split/stream_uav_eval_l1_single_frame.json
- l1_config.mode: single_frame_native
- l23_config.chunk_size: 1
- l23_config.dataset_json: /root/autodl-tmp/streamVQA/datasets/stream_uav_qa_correct_pool_mode_split/stream_uav_eval_l23_history_prefix.json
- l23_config.mode: current_frame_only_native
- model_path: /root/autodl-tmp/Qwen/Qwen2___5-VL-3B-Instruct
- output_root: /root/autodl-tmp/streamVQA/datasets/stream_uav_qa_correct_pool_qwen25vl_single_frame_eval
- sequences_root: /root/autodl-tmp/streamVQA/VisDrone2019-VID-train/VisDrone2019-VID-train/sequences
- ttft_source: choice_prefill

## Params: api_video_prefix

- api_config.api_key_env: DASHSCOPE_API_KEY
- api_config.base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
- api_config.request_timeout: 300.0
- api_config.stream: True
- api_config.ttft_source: stream_first_delta
- dataset_json: /root/autodl-tmp/streamVQA/datasets/stream_uav_qa_correct_pool_mode_split/stream_uav_eval_l23_history_prefix.json
- evaluation_type: qwen2_5_vl_api_video_prefix
- experiment_name: stream_uav_qa_correct_pool_qwen25vl_api_video_prefix_eval
- l1_policy: reused_from_native_single_frame_baseline
- l23_config.chunk_size: None
- l23_config.dataset_json: /root/autodl-tmp/streamVQA/datasets/stream_uav_qa_correct_pool_mode_split/stream_uav_eval_l23_history_prefix.json
- l23_config.fps_semantics: api_video_input_fps
- l23_config.history_prefix_fps: 0.25
- l23_config.mode: history_prefix_api_video
- model_path: qwen2.5-vl-3b-instruct
- output_root: /root/autodl-tmp/streamVQA/datasets/stream_uav_qa_correct_pool_qwen25vl_api_video_prefix_eval
- sequences_root: /root/autodl-tmp/streamVQA/VisDrone2019-VID-train/VisDrone2019-VID-train/sequences

## Streaming L1

| setting | protocol | input_mode | fps | accuracy | avg_ttft_ms |
| --- | --- | --- | ---: | ---: | ---: |
| fps0.25 | single_frame | n/a | 1.0 | 66.67% | 30.5 |
| fps1.0 | single_frame_video | video | 1.0 | 59.80% | 31.1 |
| fps4.0 | single_frame_video | video | 4.0 | 52.94% | 31.3 |

## Streaming L2/L3

| difficulty | chunk | fps0.25 acc | fps1.0 acc | fps4.0 acc | delta 1.0-0.25 | delta 4.0-0.25 | ttft@0.25 | ttft@1.0 | ttft@4.0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| L2 | 2 | 55.51% | 40.81% | 39.34% | -14.71 pp | -16.18 pp | 41.5 | 82.2 | 223.9 |
| L2 | 4 | 55.51% | 41.54% | 39.71% | -13.97 pp | -15.81 pp | 41.5 | 82.1 | 211.4 |
| L3 | 2 | 37.33% | 30.67% | 30.67% | -6.67 pp | -6.67 pp | 50.3 | 115.1 | 305.7 |
| L3 | 4 | 38.67% | 33.33% | 33.33% | -5.33 pp | -5.33 pp | 50.7 | 114.7 | 278.5 |

## Native Single-Frame Baseline

| difficulty | accuracy | avg_ttft_ms | branch |
| --- | ---: | ---: | --- |
| L1 | 61.27% | 993.7 | l1_single_frame_native |
| L2 | 42.65% | 980.5 | l23_current_frame_native |
| L3 | 29.33% | 1034.5 | l23_current_frame_native |

## API Native Video Prefix Baseline

- L1 is reused from the native single-frame baseline and is not rerun here.
- API TTFT uses `stream_first_delta`, so it is not directly comparable to local `choice_prefill` TTFT.

| difficulty | accuracy | avg_ttft_ms | ttft_source | branch |
| --- | ---: | ---: | --- | --- |
| L2 | 57.72% | 7681.9 | stream_first_delta | l23_history_prefix_api_video_prefix |
| L3 | 46.67% | 9230.8 | stream_first_delta | l23_history_prefix_api_video_prefix |

## Best Streaming vs Native

| difficulty | best streaming setting | best streaming acc | native acc | delta | best streaming ttft | native ttft |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| L1 | fps0.25 (single_frame, input=n/a) | 66.67% | 61.27% | +5.39 pp | 30.5 | 993.7 |
| L2 | fps=0.25 chunk=2 | 55.51% | 42.65% | +12.87 pp | 41.5 | 980.5 |
| L3 | fps=0.25 chunk=4 | 38.67% | 29.33% | +9.33 pp | 50.7 | 1034.5 |

## Best Streaming vs API Video Prefix

| difficulty | best streaming setting | best streaming acc | api video acc | delta | best streaming ttft | api video ttft |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| L2 | fps=0.25 chunk=2 | 55.51% | 57.72% | -2.21 pp | 41.5 | 7681.9 |
| L3 | fps=0.25 chunk=4 | 38.67% | 46.67% | -8.00 pp | 50.7 | 9230.8 |

## Conclusions

- L2 best streaming accuracy comes from fps=0.25 with chunk=2, reaching 55.51%.
- L3 best streaming accuracy comes from fps=0.25 with chunk=4, reaching 38.67%.
- At fps=1.0, chunk=4 has the higher average L2/L3 accuracy (37.44%).
- At fps=4.0, chunk=4 has the higher average L2/L3 accuracy (36.52%).
- L1 is best with fps0.25 using protocol=single_frame and input_mode=n/a, at 66.67%.
- L1 best streaming vs native delta is +5.39 pp (66.67% vs 61.27%).
- L2 best streaming vs native delta is +12.87 pp (55.51% vs 42.65%).
- L3 best streaming vs native delta is +9.33 pp (38.67% vs 29.33%).
- For L2, the best local streaming setting trails the API video-prefix baseline by -2.21 pp (55.51% vs 57.72%).
- For L3, the best local streaming setting trails the API video-prefix baseline by -8.00 pp (38.67% vs 46.67%).
- The API native video-prefix baseline is more accurate than the best completed local streaming setup on at least one of L2/L3, so the final readout is an accuracy-latency tradeoff rather than a strict streaming-memory win.
