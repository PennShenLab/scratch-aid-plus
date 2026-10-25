# Scratch-AID / ResNet-RNN

`model.py` contains the original ResNet18 frame encoder and GRU/LSTM decoder.
The existing model computations and parameter names are preserved. The encoder
replaces the first convolution with a one-channel convolution and projects
frame features before recurrent temporal modeling. The decoder classifies the
last recurrent output.

Input: `[batch, frames, 1, height, width]`. Output: two logits in the order
`[nonscratching, scratching]`. Supply already prepared grayscale tensors.

Install this folder's `requirements.txt` in a dedicated environment. From the
repository root, an example construction using the source factory defaults is:

```python
from torch import nn
from scratchaid_crnn import CNNEncoder, RNNDecoder

model = nn.Sequential(
    CNNEncoder(cnn_out_dim=256, drop_prob=0.3, bn_momentum=0.01),
    RNNDecoder(use_gru=True, cnn_out_dim=256, rnn_hidden_layers=2,
               rnn_hidden_nodes=512, num_classes=2, drop_prob=0.3,
               bidirectional=True),
)
```

Construction requests the torchvision ResNet18 pretrained weights. These are
backbone initialization weights, not a trained scratching detector. Match the
constructor values to a task checkpoint before loading it. The retained source
decoder uses a two-direction-width classifier; use `bidirectional=True`.
This folder contains the original CRNN variant. A later adapted CRNN variant
also exists in the development code; the final paper-to-checkpoint mapping
still needs confirmation.
