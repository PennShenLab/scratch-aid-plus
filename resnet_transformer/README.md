# CNNTransformer

`CNNTransformer` is the supervised baseline called CNN-transformer in the
paper's implementation details and ResNet-transformer in its model overview
and results. This folder contains only the confirmed center-frame implementation.

An ImageNet-pretrained ResNet18 encodes each grayscale frame, with a new
one-channel input convolution and a linear projection to `d_model` features.
Sinusoidal positional encoding and a batch-first transformer encoder model the
full temporal sequence. The output feature at zero-based index `T // 2` is
passed to the final linear classifier. The legacy mean-pooling implementation
is not included.

## Input and output

Input: `[batch, T, 1, height, width]`, a floating-point tensor of grayscale
frames. The supplied encoder requires one channel; converting or adapting
three-channel preprocessed clips is the caller's responsibility. Spatial
resizing and normalization are also performed outside this model.

The model accepts variable `T`, up to the positional encoding capacity of
1,000 frames, and selects the center-frame feature of the supplied sequence.
It does not sample, pad or resample frames. With `num_classes=2`, output is
`[batch, 2]` raw logits ordered `[nonscratching, scratching]`, without softmax.

## Construction

Install this folder's `requirements.txt` in a dedicated environment. From the
repository root:

```python
from resnet_transformer import CNNTransformer

model = CNNTransformer(
    num_classes=2,
    d_model=256,
    nhead=8,
    num_layers=4,
    dim_feedforward=1024,
    dropout=0.1,
)
```

This example explicitly uses the defaults of the supplied model class; it does
not specify a training configuration or reproduce paper scores. Construction
requests torchvision's ResNet18 pretrained weights. The replacement input
convolution, feature projection, transformer and classifier still require
task-specific training.

For inference, instantiate the architecture with the settings that match the
task-trained checkpoint, load its `state_dict` and call `model.eval()`.
The name `CNNTransformer` replaces `CenterFrameResNetTransformerModel` without
changing module attributes, `state_dict` keys or forward computations. Data
loading, preprocessing, training and evaluation pipelines are outside this
folder's scope.
