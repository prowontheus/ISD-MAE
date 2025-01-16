from typing import Optional, Union, List
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import (
    SegmentationHead,
    SegmentationModel
)
from segmentation_models_pytorch.base.modules import Activation
from segmentation_models_pytorch.base import initialization as init
import torch
import torch.nn as nn
import numpy as np
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from models.ResNet3DMedNet import generate_resnet3d
from models.UNet3D import UNet3D, UnetDecoder3D
from models.MixTransformer3D import generate_mit_encoder


class SegmentationModel3D(nn.Module):
    def initialize(self):
        initialize_decoder(self.decoder)
        initialize_head(self.segmentation_head)
        if self.classification_head is not None:
            initialize_head(self.classification_head)

    def check_input_shape(self, x):
        h, w = x.shape[-2:]
        output_stride = 32 # elf.encoder.output_stride
        if h % output_stride != 0 or w % output_stride != 0:
            new_h = (h // output_stride + 1) * output_stride if h % output_stride != 0 else h
            new_w = (w // output_stride + 1) * output_stride if w % output_stride != 0 else w
            raise RuntimeError(
                f"Wrong input shape height={h}, width={w}. Expected image height and width "
                f"divisible by {output_stride}. Consider pad your images to shape ({new_h}, {new_w})."
            )

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        self.check_input_shape(x)

        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return masks

    @torch.no_grad()
    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`
        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)
        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)
        """
        if self.training:
            self.eval()
        x = self.forward(x)
        return x

class SegmentationHead3D(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1):
        conv2d = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        activation = Activation(activation)
        super().__init__(conv2d, upsampling, activation)


class ProjectHead3D(nn.Sequential):
    def __init__(self, in_channels, out_dims, pooling='avg', dropout=None, activation=None):
        if pooling not in ('max', 'avg'):
            raise ValueError("pooling should be one of ('max', 'avg'), got {}.".format(pooling))

        pool = nn.AdaptiveAvgPool3d(1) if pooling == 'avg' else nn.AdaptiveMaxPool3d(1)
        flatten = nn.Flatten()
        dropout = nn.Dropout(p=dropout, inplace=True) if dropout else nn.Identity()
        linear = nn.Linear(in_channels, out_dims, bias=True)
        activation = Activation(activation)
        super().__init__(pool, flatten, linear, activation)


class ProjectHead(nn.Sequential):
    def __init__(self, in_channels, out_dims, pooling='avg', dropout=None, activation=None):
        if pooling not in ('max', 'avg'):
            raise ValueError("pooling should be one of ('max', 'avg'), got {}.".format(pooling))
        
        pool = nn.AdaptiveAvgPool2d(1) if pooling=='avg' else nn.AdaptiveMaxPool2d(1)
        flatten = nn.Flatten()
        dropout=nn.Dropout(p=dropout, inplace=True) if dropout else nn.Identity()
        linear = nn.Linear(in_channels, out_dims, bias=True)
        activation = Activation(activation)
        super().__init__(pool, flatten, linear, activation)


def initialize_decoder(module):
    for m in module.modules():
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def initialize_head(module):
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv3d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class UnetAE3D(SegmentationModel3D):
    def __init__(
            self,
            encoder_name: str = "resnet34",
            decoder_channels: List[int] = (512, 256, 128, 128, 128),
            in_channels: int = 3,
            classes: int = 1,
            depth: int = 5,
            activation: Optional[Union[str, callable]] = None,
            aux_params: Optional[List[dict]] = None):

        super().__init__()
        self.depth = depth
        if encoder_name.startswith("resnet"):
            model_depth = int(encoder_name[6:])
            self.encoder = generate_resnet3d(3, classes, model_depth=model_depth, depth=self.depth, segm=False)
        elif encoder_name.startswith("mit_"):
            self.encoder = generate_mit_encoder(
                                                encoder_name,
                                                in_channels=in_channels,
                                                depth=self.depth,
                                                weights=None,)
        self.decoder = UnetDecoder3D(
            encoder_channels=self.encoder.out_channels[:self.depth+1],
            decoder_channels=decoder_channels[-self.depth:],
            n_blocks=self.depth,
            use_batchnorm=True,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=None,
        )
        self.reconstruction_head = SegmentationHead3D(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3
        )
        self.encoder_project_heads = nn.ModuleList()

        if aux_params is not None:
            # 从最深层到最浅层
            for k, aux_param in enumerate(aux_params):
                self.encoder_project_heads.append(
                    ProjectHead3D(in_channels=self.encoder.out_channels[-k - 1], **aux_param))

        self.name = "u-{}".format(encoder_name)

        self.initialize()

    def initialize(self):
        initialize_decoder(self.decoder)
        initialize_head(self.reconstruction_head)

    def forward(self, x):
        # self.check_input_shape(x)
        features = self.encoder(x)
        decoder_output = self.decoder(*features)
        recons = self.reconstruction_head(decoder_output)

        if len(self.encoder_project_heads) > 0:
            encoder_project_outputs = []
            for k, project_head in enumerate(self.encoder_project_heads):
                k_embed = project_head(features[-k - 1])
                encoder_project_outputs.append(k_embed)
            return recons, encoder_project_outputs

        return recons


class UnetAE(SegmentationModel):
    def __init__(
        self,
        encoder_name: str = "resnet34",  
        decoder_channels: List[int] = (512, 256, 128, 128, 128),
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,  
        aux_params: Optional[List[dict]] = None     
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=5,
            weights=None,
        )

        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=5,
            use_batchnorm=True,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=None,
        )

        self.reconstruction_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        self.encoder_project_heads = nn.ModuleList()

        if aux_params is not None:
            #从最深层到最浅层
            for k, aux_param in enumerate(aux_params):
                self.encoder_project_heads.append(ProjectHead(in_channels=self.encoder.out_channels[-k-1], **aux_param))

            
        self.name = "u-{}".format(encoder_name)
        
        self.initialize()

    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.reconstruction_head)

    def forward(self, x):
        self.check_input_shape(x)
        features = self.encoder(x)
        decoder_output = self.decoder(*features)
        recons = self.reconstruction_head(decoder_output)

        if len(self.encoder_project_heads)>0:
            encoder_project_outputs = []
            for k, project_head in enumerate(self.encoder_project_heads):
                k_embed = project_head(features[-k-1])
                encoder_project_outputs.append(k_embed)
            return recons, encoder_project_outputs
        
        return recons

