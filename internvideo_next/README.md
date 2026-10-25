# InternVideo-Next with LoRA

The retained source modules contain the binary classifier, attentive probe,
LoRA linear layers, adapter injection and checkpoint loading. The backbone is
provided by the public pretrained model repository. This folder's intended
configuration is **Base + attentive probe + LoRA**, with the base weights frozen.

| Setting | Value |
| --- | --- |
| Backbone | `revliter/internvideo_next_base_p14_res224_f16` |
| Revision | `ff2659b9be360a6b1e94b1eb381778a960da6019` |
| LoRA targets | `qkv`, `proj` in the last 4 of 12 blocks |
| LoRA rank / alpha / dropout | 8 / 16 / 0.05 |
| Classifier dropout | 0.2 |
| Input | `[batch, 16, 3, 224, 224]`, processed with the pretrained image processor |
| Output | Two logits: nonscratching, scratching |

Install this folder's `requirements.txt` in a dedicated environment. The source
environment pins PyTorch and torchvision but does not lock all remaining
packages. From the repository root:

```python
from internvideo_next import InternVideoNextBinaryClassifier, configure_lora_adapters

model = InternVideoNextBinaryClassifier.from_pretrained(
    model_id="revliter/internvideo_next_base_p14_res224_f16",
    revision="ff2659b9be360a6b1e94b1eb381778a960da6019",
    pooling_mode="attentive_probe", train_mode="lora",
    freeze_backbone=True, dropout=0.2,
)
configure_lora_adapters(
    model, rank=8, alpha=16, dropout=0.05,
    last_n_blocks=4, target_modules=("qkv", "proj"),
)
```

This downloads the pinned public backbone and executes its model code. The
adapters, probe and classifier require task-trained weights before evaluation.
Other loading modes are retained in the source module for checkpoint
compatibility. The snippet explicitly selects the paper's LoRA configuration.

Public backbone: [InternVideo-Next Base](https://huggingface.co/revliter/internvideo_next_base_p14_res224_f16).
