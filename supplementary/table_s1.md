# Table S1. Detailed benchmark results

This table reports accuracy and scratching-class (class-1) F1, expressed as
percentages, for seven models under two evaluation protocols and six
preprocessing configurations, with both in-lab and external test results.

Downloads: [per-configuration CSV](table_s1.csv) and
[configuration-mean CSV](table_s1_configuration_means.csv).

## Evaluation and reporting

Trainable models use the full test sets. Realistic contains 4,964 in-lab
clips (980 scratching) and 18,616 external clips (1,428 scratching). Curated
contains 9,972 in-lab clips (4,986 scratching) and 14,228 external clips
(7,114 scratching).

Frozen MLLMs use four-shot ICL and three repeated balanced subsamples per
configuration, each targeting 200 test clips per domain (100 per class).
Their cells show mean ± sample SD over the three repeats. The sample SD
describes subsampling variability
and is not a confidence interval across animals or laboratories. These
subsets differ in membership and prevalence from the trainable-model full
tests, so cross-family comparisons are descriptive.

Means give each preprocessing configuration equal weight. They are not
pooled clip-level F1. MLLM repeat estimates are averaged within each
configuration first. Values are displayed to two decimal places; CSVs
retain additional numerical precision where available.

## Configuration means corresponding to manuscript Table II

The F1 means correspond to manuscript Table II. Each accuracy mean averages
the six detailed accuracy values for one model and protocol.

| Model | Protocol | In-lab accuracy | In-lab F1 | External accuracy | External F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| Scratch-AID | Realistic | 98.39 | 95.98 | 94.88 | 66.31 |
| CNNTransformer (ResNet-transformer) | Realistic | 98.64 | 96.60 | 95.72 | 71.20 |
| VideoMAE | Realistic | 98.81 | 97.00 | 97.14 | 82.49 |
| VideoSwin | Realistic | 98.81 | 97.03 | 96.44 | 78.12 |
| InternVideo-Next | Realistic | 98.81 | 96.99 | 97.05 | 81.19 |
| Gemini 3.1 Flash-Lite | Realistic | 78.31 | 77.50 | 75.67 | 75.07 |
| Qwen3-VL | Realistic | 80.53 | 80.15 | 75.69 | 72.43 |
| Scratch-AID | Curated | 99.67 | 99.67 | 87.10 | 85.40 |
| CNNTransformer (ResNet-transformer) | Curated | 99.68 | 99.68 | 87.71 | 85.87 |
| VideoMAE | Curated | 99.92 | 99.92 | 95.21 | 95.04 |
| VideoSwin | Curated | 99.76 | 99.76 | 93.33 | 93.00 |
| InternVideo-Next | Curated | 99.86 | 99.86 | 95.41 | 95.26 |
| Gemini 3.1 Flash-Lite | Curated | 78.05 | 77.07 | 74.64 | 73.19 |
| Qwen3-VL | Curated | 75.64 | 78.41 | 76.28 | 76.26 |

## Realistic: Trainable models — full test sets

| Model | Crop | Appearance | In-lab accuracy | In-lab F1 | External accuracy | External F1 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Scratch-AID | Per-Frame | None | 98.45 | 96.14 | 95.61 | 65.19 |
| Scratch-AID | Per-Frame | Gray | 98.43 | 96.00 | 95.18 | 60.71 |
| Scratch-AID | Per-Frame | Color Jitter | 98.19 | 95.50 | 94.05 | 66.12 |
| Scratch-AID | Motion-Gradient | None | 98.75 | 96.88 | 95.03 | 67.62 |
| Scratch-AID | Motion-Gradient | Gray | 98.33 | 95.86 | 93.39 | 62.20 |
| Scratch-AID | Motion-Gradient | Color Jitter | 98.19 | 95.52 | 96.05 | 76.01 |
| CNNTransformer (ResNet-transformer) | Per-Frame | None | 98.43 | 96.06 | 95.18 | 69.27 |
| CNNTransformer (ResNet-transformer) | Per-Frame | Gray | 98.51 | 96.28 | 95.86 | 66.90 |
| CNNTransformer (ResNet-transformer) | Per-Frame | Color Jitter | 98.45 | 96.15 | 96.73 | 79.06 |
| CNNTransformer (ResNet-transformer) | Motion-Gradient | None | 98.83 | 97.07 | 96.00 | 70.21 |
| CNNTransformer (ResNet-transformer) | Motion-Gradient | Gray | 98.91 | 97.26 | 96.22 | 72.97 |
| CNNTransformer (ResNet-transformer) | Motion-Gradient | Color Jitter | 98.71 | 96.75 | 94.33 | 68.81 |
| VideoMAE | Per-Frame | None | 98.79 | 96.94 | 97.06 | 81.98 |
| VideoMAE | Per-Frame | Gray | 98.75 | 96.87 | 97.14 | 82.41 |
| VideoMAE | Per-Frame | Color Jitter | 98.73 | 96.82 | 97.03 | 82.00 |
| VideoMAE | Motion-Gradient | None | 98.93 | 97.32 | 97.21 | 83.00 |
| VideoMAE | Motion-Gradient | Gray | 98.67 | 96.65 | 97.22 | 82.81 |
| VideoMAE | Motion-Gradient | Color Jitter | 98.97 | 97.42 | 97.17 | 82.73 |
| VideoSwin | Per-Frame | None | 98.91 | 97.24 | 96.29 | 75.89 |
| VideoSwin | Per-Frame | Gray | 98.75 | 96.86 | 97.05 | 80.30 |
| VideoSwin | Per-Frame | Color Jitter | 98.85 | 97.11 | 96.34 | 77.55 |
| VideoSwin | Motion-Gradient | None | 98.99 | 97.48 | 95.99 | 76.14 |
| VideoSwin | Motion-Gradient | Gray | 98.61 | 96.54 | 97.20 | 82.71 |
| VideoSwin | Motion-Gradient | Color Jitter | 98.77 | 96.92 | 95.78 | 76.11 |
| InternVideo-Next | Per-Frame | None | 99.03 | 97.54 | 96.79 | 79.00 |
| InternVideo-Next | Per-Frame | Gray | 98.85 | 97.07 | 97.13 | 81.83 |
| InternVideo-Next | Per-Frame | Color Jitter | 98.73 | 96.80 | 97.07 | 81.55 |
| InternVideo-Next | Motion-Gradient | None | 98.77 | 96.90 | 97.06 | 81.04 |
| InternVideo-Next | Motion-Gradient | Gray | 98.79 | 96.95 | 97.08 | 81.72 |
| InternVideo-Next | Motion-Gradient | Color Jitter | 98.69 | 96.71 | 97.17 | 81.99 |

## Realistic: Frozen MLLMs — balanced subsamples, four-shot ICL

| Model | Crop | Appearance | In-lab accuracy | In-lab F1 | External accuracy | External F1 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Gemini 3.1 Flash-Lite | Per-Frame | None | 81.17 ± 1.53 | 80.76 ± 1.19 | 79.67 ± 0.29 | 79.07 ± 1.28 |
| Gemini 3.1 Flash-Lite | Per-Frame | Gray | 77.50 ± 3.46 | 76.32 ± 3.95 | 74.33 ± 1.04 | 71.77 ± 1.79 |
| Gemini 3.1 Flash-Lite | Per-Frame | Color Jitter | 76.00 ± 3.91 | 75.38 ± 4.86 | 76.50 ± 1.80 | 77.15 ± 2.60 |
| Gemini 3.1 Flash-Lite | Motion-Gradient | None | 79.50 ± 4.36 | 79.01 ± 4.72 | 79.51 ± 4.01 | 79.22 ± 5.94 |
| Gemini 3.1 Flash-Lite | Motion-Gradient | Gray | 78.17 ± 4.54 | 76.88 ± 4.91 | 69.33 ± 1.76 | 67.35 ± 1.56 |
| Gemini 3.1 Flash-Lite | Motion-Gradient | Color Jitter | 77.50 ± 2.18 | 76.64 ± 2.75 | 74.67 ± 1.76 | 75.85 ± 2.07 |
| Qwen3-VL | Per-Frame | None | 80.83 ± 3.55 | 79.84 ± 5.45 | 76.33 ± 2.25 | 72.35 ± 2.87 |
| Qwen3-VL | Per-Frame | Gray | 80.33 ± 2.47 | 80.33 ± 3.64 | 72.50 ± 5.77 | 69.73 ± 5.32 |
| Qwen3-VL | Per-Frame | Color Jitter | 78.83 ± 5.03 | 79.06 ± 5.49 | 75.00 ± 1.80 | 72.11 ± 1.29 |
| Qwen3-VL | Motion-Gradient | None | 82.67 ± 2.84 | 82.04 ± 4.63 | 78.00 ± 3.77 | 74.95 ± 4.75 |
| Qwen3-VL | Motion-Gradient | Gray | 80.17 ± 4.62 | 80.03 ± 5.64 | 75.67 ± 2.02 | 72.22 ± 2.58 |
| Qwen3-VL | Motion-Gradient | Color Jitter | 80.33 ± 2.93 | 79.61 ± 2.76 | 76.67 ± 3.55 | 73.22 ± 3.63 |

## Curated: Trainable models — full test sets

| Model | Crop | Appearance | In-lab accuracy | In-lab F1 | External accuracy | External F1 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Scratch-AID | Per-Frame | None | 99.70 | 99.70 | 81.71 | 79.07 |
| Scratch-AID | Per-Frame | Gray | 99.74 | 99.74 | 85.29 | 82.90 |
| Scratch-AID | Per-Frame | Color Jitter | 99.65 | 99.65 | 91.23 | 90.72 |
| Scratch-AID | Motion-Gradient | None | 99.82 | 99.82 | 86.96 | 85.27 |
| Scratch-AID | Motion-Gradient | Gray | 99.72 | 99.72 | 84.37 | 81.71 |
| Scratch-AID | Motion-Gradient | Color Jitter | 99.41 | 99.41 | 93.03 | 92.71 |
| CNNTransformer (ResNet-transformer) | Per-Frame | None | 99.79 | 99.79 | 78.32 | 72.54 |
| CNNTransformer (ResNet-transformer) | Per-Frame | Gray | 99.56 | 99.56 | 88.47 | 87.18 |
| CNNTransformer (ResNet-transformer) | Per-Frame | Color Jitter | 99.61 | 99.61 | 93.12 | 92.87 |
| CNNTransformer (ResNet-transformer) | Motion-Gradient | None | 99.64 | 99.64 | 87.54 | 86.00 |
| CNNTransformer (ResNet-transformer) | Motion-Gradient | Gray | 99.84 | 99.84 | 87.05 | 85.34 |
| CNNTransformer (ResNet-transformer) | Motion-Gradient | Color Jitter | 99.62 | 99.62 | 91.75 | 91.29 |
| VideoMAE | Per-Frame | None | 99.90 | 99.90 | 94.87 | 94.67 |
| VideoMAE | Per-Frame | Gray | 99.90 | 99.90 | 95.29 | 95.12 |
| VideoMAE | Per-Frame | Color Jitter | 99.91 | 99.91 | 95.40 | 95.26 |
| VideoMAE | Motion-Gradient | None | 99.95 | 99.95 | 95.40 | 95.23 |
| VideoMAE | Motion-Gradient | Gray | 99.92 | 99.92 | 95.34 | 95.19 |
| VideoMAE | Motion-Gradient | Color Jitter | 99.93 | 99.93 | 94.98 | 94.78 |
| VideoSwin | Per-Frame | None | 99.80 | 99.80 | 90.45 | 89.64 |
| VideoSwin | Per-Frame | Gray | 99.71 | 99.71 | 93.70 | 93.37 |
| VideoSwin | Per-Frame | Color Jitter | 99.58 | 99.58 | 94.69 | 94.56 |
| VideoSwin | Motion-Gradient | None | 99.83 | 99.83 | 91.96 | 91.62 |
| VideoSwin | Motion-Gradient | Gray | 99.92 | 99.92 | 93.99 | 93.70 |
| VideoSwin | Motion-Gradient | Color Jitter | 99.72 | 99.72 | 95.22 | 95.11 |
| InternVideo-Next | Per-Frame | None | 99.83 | 99.83 | 94.57 | 94.34 |
| InternVideo-Next | Per-Frame | Gray | 99.78 | 99.78 | 95.38 | 95.25 |
| InternVideo-Next | Per-Frame | Color Jitter | 99.90 | 99.90 | 95.33 | 95.17 |
| InternVideo-Next | Motion-Gradient | None | 99.78 | 99.78 | 94.83 | 94.61 |
| InternVideo-Next | Motion-Gradient | Gray | 99.93 | 99.93 | 96.53 | 96.47 |
| InternVideo-Next | Motion-Gradient | Color Jitter | 99.92 | 99.92 | 95.85 | 95.72 |

## Curated: Frozen MLLMs — balanced subsamples, four-shot ICL

| Model | Crop | Appearance | In-lab accuracy | In-lab F1 | External accuracy | External F1 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Gemini 3.1 Flash-Lite | Per-Frame | None | 80.00 ± 5.41 | 79.53 ± 5.90 | 75.50 ± 1.80 | 73.93 ± 3.38 |
| Gemini 3.1 Flash-Lite | Per-Frame | Gray | 76.67 ± 4.37 | 75.26 ± 5.28 | 74.17 ± 3.79 | 71.31 ± 5.25 |
| Gemini 3.1 Flash-Lite | Per-Frame | Color Jitter | 76.67 ± 3.62 | 75.00 ± 4.68 | 77.00 ± 5.27 | 75.92 ± 6.72 |
| Gemini 3.1 Flash-Lite | Motion-Gradient | None | 78.83 ± 4.25 | 78.01 ± 4.79 | 76.83 ± 2.47 | 76.18 ± 2.12 |
| Gemini 3.1 Flash-Lite | Motion-Gradient | Gray | 75.78 ± 5.58 | 74.44 ± 5.79 | 70.17 ± 2.84 | 67.88 ± 2.59 |
| Gemini 3.1 Flash-Lite | Motion-Gradient | Color Jitter | 80.33 ± 2.75 | 80.15 ± 3.76 | 74.17 ± 1.89 | 73.93 ± 2.21 |
| Qwen3-VL | Per-Frame | None | 74.67 ± 3.33 | 76.28 ± 8.02 | 75.83 ± 5.30 | 76.05 ± 8.81 |
| Qwen3-VL | Per-Frame | Gray | 75.67 ± 3.06 | 79.41 ± 1.02 | 72.67 ± 4.80 | 73.46 ± 7.12 |
| Qwen3-VL | Per-Frame | Color Jitter | 72.83 ± 2.36 | 76.13 ± 2.80 | 75.67 ± 5.25 | 75.23 ± 10.06 |
| Qwen3-VL | Motion-Gradient | None | 77.83 ± 0.29 | 79.86 ± 3.39 | 79.00 ± 4.77 | 78.78 ± 8.57 |
| Qwen3-VL | Motion-Gradient | Gray | 77.50 ± 3.04 | 80.63 ± 0.86 | 76.17 ± 1.26 | 76.92 ± 4.21 |
| Qwen3-VL | Motion-Gradient | Color Jitter | 75.33 ± 2.47 | 78.17 ± 2.85 | 78.33 ± 5.03 | 77.11 ± 10.68 |

## Model names

CNNTransformer is the center-frame implementation called ResNet-transformer
in the manuscript's overview and results. Gemini is 3.1 Flash-Lite;
Qwen3-VL is the 32B-Instruct variant.

See [implementation details](implementation_details.md) for model settings
and [Table S2](table_s2.md) for Curated-to-Realistic checkpoint transfer.
