# Table S2. Curated-to-Realistic checkpoint transfer

Curated-trained, Curated-validation-selected checkpoints are applied to the
Realistic test clips without retraining. The comparison uses checkpoints
trained and selected under Realistic, evaluated on the same test domain
and preprocessing configuration. F1 is scratching-class (class-1) F1 (%).

[Download CSV](table_s2.csv), including accuracy, F1 and both test domains.

Differences are Curated-trained minus Realistic-trained, in percentage points.
Means and differences use unrounded stored values before display rounding.

## External Realistic test

| Model | Crop | Appearance | Curated-trained F1 | Realistic-trained F1 | Difference |
| --- | --- | --- | ---: | ---: | ---: |
| VideoSwin | Per-Frame | None | 71.98 | 75.89 | -3.91 |
| VideoSwin | Per-Frame | Gray | 79.33 | 80.30 | -0.97 |
| VideoSwin | Per-Frame | Color Jitter | 71.21 | 77.55 | -6.33 |
| VideoSwin | Motion-Gradient | None | 57.19 | 76.14 | -18.95 |
| VideoSwin | Motion-Gradient | Gray | 79.95 | 82.71 | -2.76 |
| VideoSwin | Motion-Gradient | Color Jitter | 70.74 | 76.11 | -5.37 |
| **VideoSwin: mean** | All six | All six | 71.73 | 78.12 | -6.38 |
| InternVideo-Next | Per-Frame | None | 79.64 | 79.00 | 0.64 |
| InternVideo-Next | Per-Frame | Gray | 78.67 | 81.83 | -3.17 |
| InternVideo-Next | Per-Frame | Color Jitter | 78.15 | 81.55 | -3.40 |
| InternVideo-Next | Motion-Gradient | None | 81.63 | 81.04 | 0.59 |
| InternVideo-Next | Motion-Gradient | Gray | 78.67 | 81.72 | -3.05 |
| InternVideo-Next | Motion-Gradient | Color Jitter | 80.35 | 81.99 | -1.63 |
| **InternVideo-Next: mean** | All six | All six | 79.52 | 81.19 | -1.67 |

## In-lab Realistic test

| Model | Crop | Appearance | Curated-trained F1 | Realistic-trained F1 | Difference |
| --- | --- | --- | ---: | ---: | ---: |
| VideoSwin | Per-Frame | None | 97.22 | 97.24 | -0.02 |
| VideoSwin | Per-Frame | Gray | 96.64 | 96.86 | -0.22 |
| VideoSwin | Per-Frame | Color Jitter | 96.16 | 97.11 | -0.95 |
| VideoSwin | Motion-Gradient | None | 96.95 | 97.48 | -0.53 |
| VideoSwin | Motion-Gradient | Gray | 96.68 | 96.54 | 0.14 |
| VideoSwin | Motion-Gradient | Color Jitter | 96.69 | 96.92 | -0.23 |
| **VideoSwin: mean** | All six | All six | 96.72 | 97.03 | -0.30 |
| InternVideo-Next | Per-Frame | None | 95.53 | 97.54 | -2.01 |
| InternVideo-Next | Per-Frame | Gray | 96.05 | 97.07 | -1.02 |
| InternVideo-Next | Per-Frame | Color Jitter | 95.63 | 96.80 | -1.17 |
| InternVideo-Next | Motion-Gradient | None | 95.76 | 96.90 | -1.14 |
| InternVideo-Next | Motion-Gradient | Gray | 95.90 | 96.95 | -1.05 |
| InternVideo-Next | Motion-Gradient | Color Jitter | 96.29 | 96.71 | -0.42 |
| **InternVideo-Next: mean** | All six | All six | 95.86 | 96.99 | -1.13 |

These contrasts compare complete protocol-specific pipelines; they do not
isolate class balance from clip construction, training volume or validation
selection. The six-configuration means are descriptive, not biological
confidence intervals.

[Back to supplementary materials](README.md).
