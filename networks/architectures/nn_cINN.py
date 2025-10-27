import torch
import torch.nn as nn
import torch.nn.functional as F
from POLARIScore.config import LOGGER
from torch.nn import init
from POLARIScore.networks.architectures.nn_UNet import DoubleConvBlock, ResConvBlock
from typing import Union, Literal
from POLARIScore.networks.architectures.nn_BaseModule import BaseModule
from POLARIScore.networks.addons.HaarDS import HaarDownsampling
import FrEIA.framework as Ff
import FrEIA.modules as Fm

"""Architecture from https://arxiv.org/abs/2105.02104"""

"""
TODO _SubNetwork using a CNN in order to not flatten the features and inputs.
"""

#maybe add multihead self attention
#use residuals blocks?
class Encoder(nn.Module):
    """Encoder which returns 'num_layers' features in a list"""
    def __init__(self, num_layers:int=5, base_filters:int=64):
        super(Encoder, self).__init__()

        self.num_layers = num_layers
        self.base_filters = base_filters

        filter_sizes = [int(base_filters*2**i) for i in range(num_layers)]
        
        self.pool = nn.MaxPool2d(2,2)
        self.encoders = nn.ModuleList()
        
        in_channels = 1
        for i in range(num_layers):
            out_channels = filter_sizes[i]
            self.encoders.append(ResConvBlock(in_channels, out_channels, init_method=init.kaiming_uniform_))
            in_channels = out_channels
        
    def forward(self, x):
        """
        Args:
            x: tensor shape: (B,C,H,W)
        Returns:
            enc_features: list of tensor with shape: (B,C_i,H,W)
        """
        enc_features = []
        for i in range(self.num_layers):
            x = self.encoders[i](x)
            enc_features.append(x)
            x = self.pool(x)

        return enc_features
def _subnet_conv(in_ch, out_ch):
    hidden_features = 32
    return nn.Sequential(
        nn.Conv2d(in_ch, hidden_features, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(hidden_features, hidden_features, 3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(hidden_features),
        nn.Conv2d(hidden_features, out_ch, 3, padding=1),
    )

class cINN(BaseModule):
    """Conditional Invertible Neural Network (cINN)"""
    def __init__(self, img_dim=32, num_layers=4, coupling_block_per_layer=3, base_filters=32):
        super().__init__()
        self.img_dim = img_dim
        self.num_layers = num_layers
        self.coupling_block_per_layer = coupling_block_per_layer
        self.base_filters = base_filters
        self.data_dim = self.img_dim*self.img_dim
        assert self.data_dim % 2 == 0, LOGGER.error("cINN: data_dim must be divisible by 2 for coupling splits.")

        self.encoder = Encoder(num_layers=self.num_layers, base_filters=self.base_filters)
        self.coupling_blocks = nn.ModuleList() #list of ModuleList

        self.downsample = HaarDownsampling()

        self.input_adapter = nn.Conv2d(1, 4, kernel_size=1)
        self.output_adapter = nn.Conv2d(4, 1, kernel_size=1)

        for i in range(self.num_layers):
            cond_channels = base_filters*2**i
            dim = 4*4**i

            ccbs = nn.ModuleList([
                Fm.AllInOneBlock(
                    dims_in=[(dim, img_dim//(2**i), img_dim//(2**i))],
                    subnet_constructor=_subnet_conv,
                    dims_c=[(cond_channels, img_dim//(2**i), img_dim//(2**i))],
                    affine_clamping=2.0,
                    permute_soft=False
                ) for _ in range(coupling_block_per_layer)
            ])
            self.coupling_blocks.append(ccbs)

        with torch.no_grad():
            dummy_y = torch.zeros(1, 1, self.img_dim, self.img_dim)
            dummy_c = torch.zeros(1, 1, self.img_dim, self.img_dim)
            z, _ = self.forward(dummy_y, dummy_c)
            self.z_shape = z.shape[1:]
            print(self.z_shape)
        
    def forward(self, y, c):
        """
        Args:
            y: true data (B, C=1, H, W)
            c: condition data (B, C=1, H, W)
        Returns:
            z: latent (B, C=1, H, W)
            log_det: tensor (B,) aggregated log determinant
        """
        B, C, H, W = y.shape
        assert C == 1, LOGGER.error("cINN: currently expects single-channel inputs")
        assert H == self.img_dim and W == self.img_dim, LOGGER.error("cINN: image size mismatch")
        
        enc_feats = self.encoder(c)
        total_log_det = torch.zeros(B, device=y.device)

        x = self.input_adapter(y)
        for i in range(self.num_layers):
            cond = enc_feats[i]
            for block in self.coupling_blocks[i]:
                x, ld = block([x], [cond], rev=False)
                x, ld = x[0], ld
                total_log_det += ld
            if i < self.num_layers - 1:
                x = self.downsample(x)

        return x, total_log_det

    def inverse(self, z, c):
        """
        Args:
            z: latent (B,C,H,W)
            c: condition (B,1,H,W)
        Returns:
            reconstructed y
        """

        enc_feats = self.encoder(c)
        x = z
        for i in reversed(range(self.num_layers)):
            if i < self.num_layers - 1:
                x = self.downsample(x, reverse=True) 
            cond = enc_feats[i]
            for block in reversed(self.coupling_blocks[i]):
                x, _ = block([x], [cond], rev=True)
                x = x[0]
        x = self.output_adapter(x)
        return x