# VideoMAE classifier core

[model.py](model.py) contains the tubelet embedding, transformer encoder,
pooling and classification head, plus the 224-pixel ViT-B/16 and ViT-L/16
factories. Model computations and parameter names are preserved from the
source. Other factory variants and the training script are omitted. The
factories are called directly, without registering or replacing timm models.

Inputs have shape `[batch, 3, frames, height, width]`; outputs are unnormalized
class logits. Set `num_classes=2` for `0 = nonscratching`, `1 = scratching`.
The ViT-L/16 factory has 24 blocks, width 1024 and 16 attention heads; the
ViT-B/16 factory has 12 blocks, width 768 and 12 heads.

Install [requirements.txt](requirements.txt) in a dedicated environment. From
the repository root, this constructs the ViT-L/16 architecture for 29-frame
clips and mean pooling described in the paper:

```python
from videomae import vit_large_patch16_224

model = vit_large_patch16_224(
    num_classes=2,
    all_frames=29,
    tubelet_size=2,
    use_mean_pooling=True,
    drop_path_rate=0.1,
    init_scale=0.001,
)
```

This initializes a model without task weights. The inherited `pretrained`
argument does not load weights; load a matching checkpoint separately. The
source uses a stride-2, unpadded temporal convolution: 29 frames produce 14
tubelets, so the last unpaired frame is unused. Set `all_frames` to match the
input token count. Preprocessing, fine-tuning, checkpoint conversion and
evaluation pipelines are outside this release.

Offline verification used synthetic CPU inputs with PyTorch 2.6.0,
torchvision 0.21.0, timm 0.4.12 and NumPy 1.26.4. It checked the extracted
encoder against the source, without downloading weights or evaluating data.

## Upstream attribution

Derived from the official [VideoMAE implementation](https://github.com/MCG-NJU/VideoMAE)
by Zhan Tong, Yibing Song, Jue Wang and Limin Wang, for
[VideoMAE: Masked Autoencoders Are Data-Efficient Learners for Self-Supervised Video Pre-Training](https://arxiv.org/abs/2203.12602).
The upstream CC BY-NC 4.0 [LICENSE](LICENSE) and third-party [NOTICE.md](NOTICE.md)
are retained. Changes here are limited to extracting these definitions,
removing timm registration and normalizing trailing whitespace.
