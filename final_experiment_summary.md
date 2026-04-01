# StreamUAV-QA Streaming Sweep Summary

## Experiment Status

- Baseline `fps=0.25`, `KV cache=150000`: complete
- Streaming `fps=1`, `KV cache=150000`: complete
- Streaming `fps=4`: `L1` and `chunk2` complete at `KV cache=150000`; `chunk4` rescue status: complete
- Native Qwen2.5-VL current-frame single-image baseline: complete
- API native prefix-video baseline via DashScope: complete

## Parameter Notes

- Baseline: `L1=image-mode chunk1 fps=1`, `L2/L3=history_prefix chunk2,4 fps=0.25`, `max_cache_tokens=150000`
- FPS=1: `L1=single-frame-video chunk1 fps=1`, `L2/L3=history_prefix chunk2,4 fps=1`, `max_cache_tokens=150000`
- FPS=4 completed branches: `L1=single-frame-video chunk1 fps=4`, `chunk2=history_prefix fps=4`, `max_cache_tokens=150000`
- FPS=4 chunk4 rescue: `history_prefix chunk4 fps=4`, `max_cache_tokens=130000`, resumed from the original partial outputs
- Native baseline: current-frame-only single-image inputs, no streaming cache, same correct-pool multiple-choice scoring
- API video baseline: prefix video clip via DashScope OpenAI-compatible API, `history_prefix_fps=0.25`, TTFT measured as `stream_first_delta`

## Interpretation Notes

- Baseline `L1` is a current-frame-only `image-mode` protocol. It does not append the whole video prefix and does not use temporal memory from earlier frames.
- Therefore, baseline `L1` beating the native single-frame baseline should be read as a protocol advantage of the local streaming wrapper on single-frame QA, not as evidence of temporal understanding.
- The temporal-memory claim should be judged mainly from `L2/L3`, where the streaming runs can ingest prefix history while the native single-frame baseline cannot.
- After adding the API native prefix-video baseline, the fair comparison becomes: local streaming vs native single-frame tests temporal-context value; local streaming vs API prefix-video tests whether explicit streaming cache beats native prefix-video understanding.

## TTFT Note

- Local streaming/native TTFT uses `choice_prefill`.
- API video TTFT uses `stream_first_delta`, so treat it as a separate latency notion.

## Core Results

| Setting | Branch | Difficulty | Accuracy | Avg TTFT | Delta vs baseline |
| --- | --- | --- | --- | --- | --- |
| baseline fps=0.25 | L1 | L1 | 66.67% | 30.5 ms | baseline |
| fps=1 | L1 video | L1 | 59.80% | 31.1 ms | -6.86 pp |
| fps=4 | L1 video | L1 | 52.94% | 31.3 ms | -13.73 pp |
| native single-frame | L1 | L1 | 61.27% | 993.7 ms | -5.39 pp |
| baseline fps=0.25 | chunk2 | L2 | 55.51% | 41.5 ms | baseline |
| fps=1 | chunk2 | L2 | 40.81% | 82.2 ms | -14.71 pp |
| fps=4 chunk2 | chunk2 | L2 | 39.34% | 223.9 ms | -16.18 pp |
| native single-frame | current-frame | L2 | 42.65% | 980.5 ms | -12.87 pp |
| API native video prefix | current-frame-prefix-video | L2 | 57.72% | 7681.9 ms | +2.21 pp |
| baseline fps=0.25 | chunk2 | L3 | 37.33% | 50.3 ms | baseline |
| fps=1 | chunk2 | L3 | 30.67% | 115.1 ms | -6.67 pp |
| fps=4 chunk2 | chunk2 | L3 | 30.67% | 305.7 ms | -6.67 pp |
| native single-frame | current-frame | L3 | 29.33% | 1034.5 ms | -8.00 pp |
| API native video prefix | current-frame-prefix-video | L3 | 46.67% | 9230.8 ms | +9.33 pp |
| baseline fps=0.25 | chunk4 | L2 | 55.51% | 41.5 ms | baseline |
| fps=1 | chunk4 | L2 | 41.54% | 82.1 ms | -13.97 pp |
| fps=4 chunk4 | chunk4 | L2 | 39.71% | 211.4 ms | -15.81 pp |
| native single-frame | current-frame | L2 | 42.65% | 980.5 ms | -12.87 pp |
| API native video prefix | current-frame-prefix-video | L2 | 57.72% | 7681.9 ms | +2.21 pp |
| baseline fps=0.25 | chunk4 | L3 | 38.67% | 50.7 ms | baseline |
| fps=1 | chunk4 | L3 | 33.33% | 114.7 ms | -5.33 pp |
| fps=4 chunk4 | chunk4 | L3 | 33.33% | 278.5 ms | -5.33 pp |
| native single-frame | current-frame | L3 | 29.33% | 1034.5 ms | -9.33 pp |
| API native video prefix | current-frame-prefix-video | L3 | 46.67% | 9230.8 ms | +8.00 pp |

## Conclusions

1. Raising streaming FPS from `0.25` to `1` consistently hurt `L2/L3` accuracy while also pushing TTFT from tens of milliseconds to roughly `80-115 ms`.
2. The completed `fps=4` branches are even less favorable: `L1` falls further, `chunk2` accuracy stays below the baseline, and TTFT rises into the hundreds of milliseconds.
3. The native single-frame baseline is much slower than the streaming setup, with TTFT around `1 s`, and it underperforms the baseline streaming run on `L2/L3`, which supports the claim that streaming memory is contributing useful temporal context.
4. Baseline `L1` outperforming the native single-frame baseline should not be interpreted as a temporal-memory gain, because the baseline `L1` itself is also single-frame; it is better explained as a protocol advantage of the local streaming wrapper.
5. The API native video-prefix baseline beats the best completed local streaming setting on `L2` by +2.21 pp (57.72% vs 55.51%) and on `L3` by +8.00 pp (46.67% vs 38.67%).
6. The best completed local configuration remains the original baseline: `L1=image-mode`, `L2/L3 history_prefix`, `fps=0.25`, `chunk2/4`, `max_cache_tokens=150000`.
7. This does not make API TTFT directly better: API video TTFT uses `stream_first_delta` and lands in the multi-second range, so the result is an accuracy-over-latency tradeoff, not a direct latency win.
8. The `fps=4 chunk4` rescue completed only after lowering `max_cache_tokens` to `130000`, so it is not a strict apples-to-apples comparison with the `150000` runs.

## Output Files

- `baseline merged summary`: `/root/autodl-tmp/streamVQA/datasets/stream_uav_qa_correct_pool_stream3b_eval/merged_summary.json`
- `fps1 merged summary`: `/root/autodl-tmp/streamVQA/datasets/stream_uav_qa_correct_pool_stream3b_eval_fps1/merged_summary.json`
- `fps4 L1 summary`: `/root/autodl-tmp/streamVQA/datasets/stream_uav_qa_correct_pool_stream3b_eval_fps4/l1_single_frame_video_chunk1/summary.json`
- `fps4 chunk2 summary`: `/root/autodl-tmp/streamVQA/datasets/stream_uav_qa_correct_pool_stream3b_eval_fps4/l23_history_prefix_chunk2/summary.json`
- `fps4 chunk4 rescue summary`: `/root/autodl-tmp/streamVQA/datasets/stream_uav_qa_correct_pool_stream3b_eval_fps4_chunk4_resume130k_v2/l23_history_prefix_chunk4/summary.json`
- `native merged summary`: `/root/autodl-tmp/streamVQA/datasets/stream_uav_qa_correct_pool_qwen25vl_single_frame_eval/merged_summary.json`
- `api video merged summary`: `/root/autodl-tmp/streamVQA/datasets/stream_uav_qa_correct_pool_qwen25vl_api_video_prefix_eval/merged_summary.json`

