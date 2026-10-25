# Qwen3-VL ICL prompts

The zero-, two- and four-shot prompt files retain the exact source strings.
`settings.json` records the source inference settings for Qwen3-VL-32B-Instruct
and Qwen3-VL-32B-Thinking. `response_schema.json` describes the requested fields.

The source constructs one user message: prompt text first, then labeled example
videos, then the target video. Demonstrations are prefixed with
`Example N: labeled scratching.` or `Example N: labeled nonscratching.` The
target prefix is `Now classify the following target video.` Two-shot uses one
example per class; four-shot uses two per class, with scratching examples first.
The zero-shot request contains only the prompt and target. Four-shot examples
in the source runner are taken from the source-training portion of its manifest.

Video parts request 29 frames and `max_pixels=100352`. Generation uses
`do_sample=False`, with 256 new tokens for Instruct or 4096 for Thinking. The
source removes a `<think>...</think>` block and validates the extracted JSON.
The confidence value is a requested self-report, not a calibrated probability.

Model loading, dataset manifests, video processing and batch evaluation are
outside this prompt-only artifact. This folder makes no separate performance
claim for the Instruct and Thinking variants.
