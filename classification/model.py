import torch.nn as nn
from typing import Optional, Union, List
import torch
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base.modules import Activation
from collections import OrderedDict
from segmentation_models_pytorch.base import initialization as init
from segmentation_models_pytorch.base import ClassificationHead


class ProjectHead(nn.Sequential):
    def __init__(self, in_channels, out_dims, pooling='avg', dropout=None, activation=None):
        if pooling not in ('max', 'avg'):
            raise ValueError("pooling should be one of ('max', 'avg'), got {}.".format(pooling))

        pool = nn.AdaptiveAvgPool2d(1) if pooling == 'avg' else nn.AdaptiveMaxPool2d(1)
        flatten = nn.Flatten()
        dropout = nn.Dropout(p=dropout, inplace=True) if dropout else nn.Identity()
        linear = nn.Linear(in_channels, out_dims, bias=True)
        activation = Activation(activation)
        super().__init__(pool, flatten, linear, activation)


class Classifier(nn.Module):
    def __init__(self,
                 encoder_name: str = 'resnet50',
                 in_channels: int = 3,
                 hidden_channels: int = 32,
                 output_channels: int = 128,
                 n_classes: int = 1,
                 aux_params: Optional[List[dict]] = None):
        super(Classifier, self).__init__()
        self.encoder_name = encoder_name
        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=5,
            weights=None,
        )
        self.classifier = nn.Sequential(
            nn.Linear(output_channels, output_channels // 2),
            nn.ReLU(),
            nn.Linear(output_channels // 2, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, n_classes)
        )
        self.encoder_project_heads = nn.ModuleList()
        self.init_weights()

        if aux_params is not None:
            # 从最深层到最浅层
            for k, aux_param in enumerate(aux_params):
                self.encoder_project_heads.append(
                    ProjectHead(in_channels=self.encoder.out_channels[-k - 1], **aux_param))
        else:
            raise NotImplementedError('encoder project heads not implemented')

    def load_encoder(self, checkpoint):
        encoder_dict = OrderedDict({k[15:]: v for k, v in checkpoint.items() if k.startswith('module.encoder.')})
        encoder_project_heads_dict = OrderedDict(
                {k[29:]: v for k, v in checkpoint.items() if k.startswith('module.encoder_project_heads.')})
        self.encoder.load_state_dict(encoder_dict)
        self.encoder_project_heads.load_state_dict(encoder_project_heads_dict)

    def load_encoder_and_classifier(self, checkpoint):
        self.load_encoder(checkpoint)
        classifier_dict = OrderedDict({k[11:]: v for k, v in checkpoint.items() if k.startswith('classifier.')})
        self.classifier.load_state_dict(classifier_dict)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def frozen_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.encoder_project_heads.parameters():
            param.requires_grad = False

    def forward(self, data):
        features = self.encoder(data)
        encoder_project_outputs = []
        for k, project_head in enumerate(self.encoder_project_heads):
            k_embed = project_head(features[-k - 1])
            encoder_project_outputs.append(k_embed)
        output = self.classifier(encoder_project_outputs[0])
        return output.squeeze(1)
