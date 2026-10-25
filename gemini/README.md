# Gemini 3.1 Flash-Lite prompts

The files contain the exact zero-, two- and four-shot prompts from source
scripts configured for `gemini-3.1-flash-lite`. The few-shot files are the
unaltered output of the source prompt builder with `total_shots=2` and `4`.
The requested output fields are documented in [response_schema.json](response_schema.json).

[settings.json](settings.json) records the model identifier, 24 FPS video
metadata, temperature 0, JSON output and message order. In zero-shot requests,
the target video precedes the prompt. In few-shot requests, the prompt comes
first, followed by labeled demonstrations and then the target introduction
and target video. The exact demonstration and target prefixes are included.

For each query, demonstrations are retrieved from the **training partition**
using cosine nearest neighbors over 768-dimensional `gemini-embedding-2`
video embeddings. Two-shot inference uses one example per class; four-shot
inference uses two per class. Scratching examples come first, followed by
nonscratching examples, each in nearest-first order within its class. Evaluation
clips must not be used as demonstration candidates.

This folder provides prompts and their request/retrieval contract. API clients,
credentials, video files, embedding generation, dataset sampling and result
aggregation are outside this minimal release. The settings describe the
included 3.1 configuration; they are not a run log or evidence of paper scores.
