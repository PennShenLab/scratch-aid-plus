"""CNNTransformer with a ResNet18 encoder and center-frame temporal readout."""

import math
import torch
from torch import nn
from torchvision import models

class ResNetEncoder(nn.Module):
    def __init__(self, output_features=256):
        super().__init__()
        # pre-trained resnet
        weights = models.ResNet18_Weights.DEFAULT
        resnet = models.resnet18(weights=weights)

        # modify first layer for grayscale
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # remove final fc
        self.features = nn.Sequential(*list(resnet.children())[:-1])

        # add new fc
        self.fc = nn.Linear(resnet.fc.in_features, output_features)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class TemporalPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super(TemporalPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, num_frames, d_model)

        # Get the actual number of frames
        _, num_frames, _ = x.shape

        # Slice the first num_frames time codes
        # Shape becomes (num_frames, 1, d_model)
        pe = self.pe[:num_frames, :]

        # Flip dimensions to (1, num_frames, d_model) so it fits the batch
        pe = pe.permute(1, 0, 2)

        # Add positional encoding to each frame
        x = x + pe
        return self.dropout(x)


class CNNTransformer(nn.Module):
    def __init__(self, num_classes, d_model=256, nhead=8, num_layers=4, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.resnet_encoder = ResNetEncoder(output_features=d_model)
        self.pos_encoder = TemporalPositionalEncoding(d_model, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x shape: (batch_size, num_frames, channels, height, width)
        batch_size, num_frames, c, h, w = x.shape

        # process frames
        x = x.view(batch_size * num_frames, c, h, w)
        x = self.resnet_encoder(x)
        x = x.view(batch_size, num_frames, -1)

        # add pos encoding
        x = self.pos_encoder(x)

        # transformer encoding
        x = self.transformer_encoder(x)

        # Select the feature vector of the middle frame
        middle_frame_index = num_frames // 2
        x = x[:, middle_frame_index, :]

        # classify scratch or non scratch
        x = self.fc(x)
        return x
