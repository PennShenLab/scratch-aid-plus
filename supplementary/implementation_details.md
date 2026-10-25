# Supplementary implementation details

These notes accompany the Scratch-AID+ benchmark and provide the per-model
implementation details omitted from the main manuscript for space. The settings
below describe the reported experiments; model-construction examples in the code
folders describe the supplied classes and do not reproduce the benchmark by
themselves. See [Table S1](table_s1.md) for detailed results and evaluation scope.

## Shared protocol and evaluation

The binary classes are `0 = nonscratching` and `1 = scratching`. The benchmark
constructs 29-frame source clips and evaluates two protocols separately:

- **Realistic** uses sequential, non-overlapping clips, labeled by the center
  frame, retaining naturally sparse scratching events.
- **Curated** uses overlapping, class-balanced clips. Positive clips contain at
  least 15 consecutive scratching frames; negatives are sampled at least 45
  frames away from a scratching event.

Each protocol has six preprocessing configurations: Per-Frame or Motion-Gradient
cropping, crossed with None, Gray or Color Jitter. Per-Frame cropping independently
selects the lowest-mean-intensity 256 × 256 window in each frame. Motion-Gradient
cropping uses one window across the clip, chosen using motion and spatial-gradient
scores. None retains RGB, Gray converts to grayscale, and Color Jitter perturbs
brightness, contrast, saturation and hue. Model-specific resizing, channel
conversion and temporal sampling follow preprocessing.

Trainable models are evaluated on the full in-lab and external test sets. Frozen
MLLMs use three repeated balanced subsamples per configuration, with 200 test
clips per domain and repeat (100 per class); the sampled training pools also
contain 200 clips (100 per class). Demonstrations come from the training partition.
The reported benchmark uses four-shot ICL. Zero- and two-shot prompts are retained
as additional supported prompting conditions.

Checkpoint selection is model-specific, as stated below. The primary endpoint is
scratching-class (class-1) F1; accuracy is secondary. Configuration means weight
all six configurations equally. For MLLMs, each configuration is first summarized
over the three subsampling repeats. These means do not pool predictions across
configurations or estimate uncertainty across independent animals or laboratories.

## VideoMAE

Core implementation: [videomae/README.md](../videomae/README.md).

Kinetics-400-pretrained VideoMAE is fully fine-tuned on 29-frame, 224 × 224 clips
with temporal stride 1, mean pooling and a binary linear head. Training uses
AdamW for 7 epochs, batch size 16, learning rate 6.25 × 10⁻⁵ and weight decay
0.05. Linear warm-up is followed by cosine learning-rate decay. Layer-wise
learning-rate decay is 0.75, and stochastic depth is 0.1.

Training augmentations are RandAugment, random erasing with probability 0.25,
Mixup with alpha 0.8, CutMix with alpha 1.0, and label smoothing 0.1. The checkpoint
with the highest validation accuracy is selected. Testing uses a single center
crop.

## CNNTransformer / ResNet-transformer

Core implementation: [resnet_transformer/README.md](../resnet_transformer/README.md).
`CNNTransformer` is the class called CNN-transformer in the implementation notes
and ResNet-transformer in the manuscript's model overview and results.

An ImageNet-pretrained ResNet18 encodes grayscale frames; the temporal transformer
processes the sequence and passes the feature at zero-based index `T // 2` to a
binary linear classifier. The supplied model accepts variable `T` and does not
sample, pad or resample frames. Use the architecture settings matching the trained
checkpoint; constructor defaults alone do not establish the experiment settings.

The reported training setup uses 224 × 224 inputs, 20 epochs, Adam, weighted
cross-entropy and multi-step learning-rate decay. The checkpoint with the highest
validation scratching-class F1 is selected.

## VideoSwin

Core configuration: [video_swin/README.md](../video_swin/README.md).

Swin-B is fully fine-tuned. Each input contains 16 consecutive frames drawn from a
29-frame clip. Training uses random temporal sampling, random resized crops to
224 × 224 and horizontal flips with probability 0.5. Validation and testing use
one temporal clip and a center crop after resizing the shorter side to 256 pixels.

Training uses AdamW, batch size 4, two epochs of linear warm-up, and then cosine
learning-rate decay. Classifier dropout is 0.5 and stochastic depth is 0.3. For
each preprocessing configuration, three Optuna trials sample the classifier
learning rate log-uniformly from [8 × 10⁻⁵, 1.5 × 10⁻⁴] and weight decay
log-uniformly from [10⁻⁴, 5 × 10⁻⁴]. The backbone learning rate is 0.2 times the
classifier learning rate. Trials run for 15 epochs under Curated and 35 epochs
under Realistic. The trial and checkpoint with the highest validation
scratching-class F1 are used for full-test evaluation.

The loss is unweighted cross-entropy. Inputs use ImageNet normalization; training
uses mixed precision on one A100 GPU per run.

## InternVideo-Next

Core implementation and pinned backbone:
[internvideo_next/README.md](../internvideo_next/README.md).

The Base encoder remains frozen, with LoRA on `qkv` and `proj` in its final four
transformer blocks. LoRA rank, alpha and dropout are 8, 16 and 0.05. The adapters,
attentive probe and binary classifier are optimized jointly. Inputs contain
16 uniformly sampled frames at 224 × 224.

Training uses AdamW, a constant learning rate of 10⁻⁴, weight decay 10⁻³ and
batch size 1. Runs last 6 epochs under Curated and 15 epochs under Realistic.
Training and validation use a center crop. Testing averages logits across three
spatial crops; these crops coincide for square inputs resized to 224 × 224.
The checkpoint with the highest validation scratching-class F1 is selected.

The loss is unweighted cross-entropy. Inputs use the pretrained image processor
and ImageNet normalization. Training uses mixed precision on one A100 GPU per run.

## Gemini 3.1 Flash-Lite

Exact prompts and request/retrieval settings:
[gemini/README.md](../gemini/README.md).

The model is `gemini-3.1-flash-lite`, accessed through the API with frozen weights.
Video metadata specifies 24 fps; temperature is 0, and responses use structured
JSON. Class-balanced demonstrations are retrieved from the training partition
using cosine nearest-neighbor search over video embeddings. Two-shot ICL uses
one example per class; four-shot ICL uses two per class. The prompt files and
settings preserve the order of demonstrations and target video.

## Qwen3-VL

Exact prompts and generation settings:
[qwen3_vl/README.md](../qwen3_vl/README.md).

The reported model is frozen Qwen3-VL-32B-Instruct, run locally on A100 GPUs.
Video inputs request 29 uniformly sampled frames per clip. Labeled demonstrations
are fixed within each prompting run, with one example per class for two-shot ICL
and two for four-shot ICL. Greedy decoding generates at most 256 new tokens.

Generated labels and confidence values are converted to scratching scores for
the auxiliary AUROC computation. The requested confidence is a model self-report
and is not established as a calibrated probability. The full behavioral
definitions and exact prompts are included in the model folder. Table S1 reports
accuracy and scratching-class F1 for the manuscript's four-shot condition.
