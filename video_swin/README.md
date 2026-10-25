# Video Swin Transformer

`model_config.py` is the resolved, standalone **Swin-B** binary model
configuration. Its implementation comes from MMAction2 v1.2.0. The configuration
preserves the source backbone, two-class I3D head, 0.3 stochastic-depth rate,
0.5 classifier dropout and ImageNet normalization. No local base-config files,
datasets, custom metrics or training runtime are required to read it.

The public checkpoint is a **2D Swin-B initialization inflated to 3D**
(`pretrained2d=True`). It is not a task-trained scratching checkpoint. The model
uses full fine-tuning in the benchmark.

Install this folder's `requirements.txt` in a dedicated environment compatible
with MMAction2 v1.2.0. From the repository root:

```python
from mmaction.registry import MODELS
from mmaction.utils import register_all_modules
from video_swin.model_config import model as model_config

register_all_modules(init_default_scope=True)
model = MODELS.build(model_config)
model.init_weights()  # initializes from the public backbone checkpoint
```

The backbone consumes normalized `[batch, 3, frames, height, width]` tensors.
The benchmark uses 16 frames at 224 by 224 pixels; MMAction2's recognizer also
has a view/clip axis. Use its input-preprocessor interface when invoking the
full recognizer. Output class order is nonscratching, scratching.

Implementation and license: [MMAction2 v1.2.0](https://github.com/open-mmlab/mmaction2/tree/v1.2.0)
(Apache-2.0). The upstream license is included in this folder.
