"""ResNet18 frame encoder and recurrent scratching classifier."""

import torch
from torch import nn
from torchvision import models

class CNNEncoder(nn.Module):
    def __init__(self, cnn_out_dim=256, drop_prob=0.3, bn_momentum=0.01):
        
        super(CNNEncoder, self).__init__()

        self.cnn_out_dim = cnn_out_dim
        self.drop_prob = drop_prob
        self.bn_momentum = bn_momentum

        weights = models.ResNet18_Weights.DEFAULT
        pretrained_cnn = models.resnet18(weights=weights)

        # Modify the first convolutional layer to accept 1 channel input
        pretrained_cnn.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        cnn_layers = list(pretrained_cnn.children())[:-1]

        self.cnn = nn.Sequential(*cnn_layers)

        self.fc = nn.Sequential(
            *[
                self._build_fc(pretrained_cnn.fc.in_features, 256, True),
                nn.ReLU(),
                nn.Dropout(p=self.drop_prob),
                self._build_fc(256, self.cnn_out_dim, False)
            ]
        )

    def _build_fc(self, in_features, out_features, with_bn=True):
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features, momentum=self.bn_momentum)
        ) if with_bn else nn.Linear(in_features, out_features)

    def forward(self, x_3d):
        cnn_embedding_out = []
        for t in range(x_3d.size(1)):

            #with torch.no_grad():
            x = self.cnn(x_3d[:, t, :, :, :])
            x = torch.flatten(x, start_dim=1)

            x = self.fc(x)

            cnn_embedding_out.append(x)

        cnn_embedding_out = torch.stack(cnn_embedding_out, dim=0).transpose(0, 1)

        return cnn_embedding_out


class RNNDecoder(nn.Module):
    def __init__(self, use_gru=True, cnn_out_dim=256, rnn_hidden_layers=3, rnn_hidden_nodes=256,
            num_classes=2, drop_prob=0.3, bidirectional = True):
        super(RNNDecoder, self).__init__()

        self.rnn_input_features = cnn_out_dim
        self.rnn_hidden_layers = rnn_hidden_layers
        self.rnn_hidden_nodes = rnn_hidden_nodes
        self.bidirectional  = bidirectional
        self.drop_prob = drop_prob
        self.num_classes = num_classes 

        rnn_params = {
            'input_size': self.rnn_input_features,
            'hidden_size': self.rnn_hidden_nodes,
            'num_layers': self.rnn_hidden_layers,
            'batch_first': True,
            'bidirectional':self.bidirectional
        }

        self.rnn = (nn.GRU if use_gru else nn.LSTM)(**rnn_params)

        if self.bidirectional:
            self.fc = nn.Sequential(
                nn.Linear(self.rnn_hidden_nodes*2, 256),
                nn.ReLU(),
                nn.Dropout(self.drop_prob),
                nn.Linear(256, self.num_classes)
            )

        else:
            self.fc = nn.Sequential(
                nn.Linear(self.rnn_hidden_nodes*2, 256),
                nn.ReLU(),
                nn.Dropout(self.drop_prob),
                nn.Linear(256, self.num_classes)
            )

    def forward(self, x_rnn):
        self.rnn.flatten_parameters()
        rnn_out, _ = self.rnn(x_rnn, None)

        x = self.fc(rnn_out[:, -1, :]) 

        return x
