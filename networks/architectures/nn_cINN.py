import torch
import torch.nn as nn
import torch.nn.functional as F
from POLARIScore.config import LOGGER
from torch.nn import init
from POLARIScore.networks.architectures.nn_UNet import DoubleConvBlock, ResConvBlock
from typing import Union, Literal
from POLARIScore.networks.architectures.nn_BaseModule import BaseModule
from POLARIScore.networks.addons.HaarDS import HaarDownsampling
import numpy as np

"""Architecture from https://arxiv.org/abs/2105.02104"""

class Invertible1x1Conv(nn.Module):
    """
    Learnable invertible 1x1 convolution (Glow-style) that mixes channels.
    Keeps determinant easy to compute for log-likelihood if needed.
    """
    def __init__(self, num_channels):
        super().__init__()
        w_init = torch.linalg.qr(torch.randn(num_channels, num_channels))[0]
        self.weight = nn.Parameter(w_init)

    def _get_weight(self, reverse=False):
        W = self.weight
        if reverse:
            W = torch.inverse(W.double()).float()
        return W.view(W.shape[0], W.shape[1], 1, 1)

    def forward(self, x, reverse=False):
        W = self._get_weight(reverse)
        x = F.conv2d(x, W)
        logdet = torch.slogdet(self.weight)[1] * x.shape[2] * x.shape[3]
        if reverse:
            logdet = -logdet
        return x, logdet

class RandomPermutation(nn.Module):
    """Random permutation of features for invertible networks."""

    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        perm = torch.randperm(num_features)
        self.register_buffer("perm", perm)
        self.register_buffer("inv_perm", torch.argsort(perm))

    def forward(self, x, reverse=False):
        if reverse:
            return x[:, self.inv_perm]
        else:
            return x[:, self.perm]


# maybe add multihead self attention
# use residuals blocks?
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
            self.encoders.append(DoubleConvBlock(in_channels, out_channels, init_method=init.kaiming_uniform_))
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


class _SubNetwork(nn.Module):
    def __init__(self, split_dim, cond_dim):
        super().__init__()
        hidden_features = max(32,(split_dim+cond_dim)*2)
        self.nn = nn.Sequential(
            nn.Conv2d(split_dim+cond_dim, hidden_features, 3, padding=1),
            #nn.GroupNorm(hidden_features),
            nn.ReLU(),
            nn.Conv2d(hidden_features, hidden_features, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_features),
            nn.Conv2d(hidden_features, 2*split_dim, 3, padding=1),
        )

        nn.init.xavier_uniform_(self.nn[-0].weight)
        nn.init.constant_(self.nn[0].bias, 0.)
        nn.init.xavier_uniform_(self.nn[-4].weight)
        nn.init.constant_(self.nn[-4].bias, 0.)
        # zero-init final conv to produce near-zero s,t at start
        nn.init.zeros_(self.nn[-1].weight)
        nn.init.zeros_(self.nn[-1].bias)

        self.register_parameter("alpha_s", nn.Parameter(torch.tensor(0.01)))

    def forward(self, x):
        """
        Args:
            x: tensor shape: (B, split_dim+cond_dim,H,W)
        Returns:
            s, t each of shape (B,2.*split_dim,H,W)
        """
        x = self.nn(x)
        s, t = x.chunk(2, dim=1)
        s = 2.*self.alpha_s/torch.pi * torch.arctan(s/self.alpha_s)
        return s, t

class ConditionalCouplingLayer(nn.Module):
    def __init__(self, dim, cond_dim):
        super().__init__()
        assert dim % 2 == 0, LOGGER.error("cINN: Input dimension in CCB need to be divisible by 2.")
        self.dim = dim
        self.split_dim = dim // 2
        self.sub1 = _SubNetwork(self.split_dim, cond_dim)
        self.sub2 = _SubNetwork(self.split_dim, cond_dim)
        self.perm = RandomPermutation(dim)


    def forward(self, x, c, reverse=False):
        """
        Args:
            x: tensor shape: (B, dim, H, W)
            c: tensor shape: (B, cond_dim, H, W)
            reverse: bool, whether to run inverse mapping
        Returns:
            y: (B, dim, H, W)
            log_det_jac: (B,) log-determinant contribution of this layer
        """
        if not reverse:
            x1, x2 = x[:, :self.split_dim], x[:, self.split_dim:]
            s1, t1 = self.sub1(torch.cat([x2, c], dim=1))
            y1 = x1 * torch.exp(s1) + t1
            s2, t2 = self.sub2(torch.cat([y1, c], dim=1))
            y2 = x2 * torch.exp(s2) + t2
            y = torch.cat([y1, y2], dim=1)
            y = self.perm(y, reverse=False)
            log_det_jac = torch.sum(s1, dim=[1,2,3]) + torch.sum(s2, dim=[1,2,3])
        else:
            x = self.perm(x, reverse=True)
            x1, x2 = x[:, :self.split_dim], x[:, self.split_dim:]
            s2, t2 = self.sub2(torch.cat([x1, c], dim=1))
            y2 = (x2 - t2) * torch.exp(-s2)
            s1, t1 = self.sub1(torch.cat([y2, c], dim=1))
            y1 = (x1 - t1) * torch.exp(-s1)
            log_det_jac = -(torch.sum(s1, dim=[1, 2, 3]) + torch.sum(s2, dim=[1, 2, 3]))
            y = torch.cat([y1, y2], dim=1)
            log_det_jac = -(torch.sum(s1, dim=[1,2,3]) + torch.sum(s2, dim=[1,2,3]))

        return y, log_det_jac


class cINN(BaseModule):
    """Conditional Invertible Neural Network (cINN)"""

    def __init__(self, img_dim=128, num_layers=2, coupling_block_per_layer=3, base_filters=64):
        super().__init__()
        self.img_dim = img_dim
        self.num_layers = num_layers
        self.coupling_block_per_layer = coupling_block_per_layer
        self.base_filters = base_filters
        
        self.encoder = Encoder(num_layers=self.num_layers+1, base_filters=self.base_filters)
        self.coupling_blocks = nn.ModuleList()  # list of ModuleList

        self.downsample = HaarDownsampling()
        for i in range(self.num_layers):
            cond_channels = base_filters * 2**(i+1)
            ccbs = nn.ModuleList()
            for j in range(self.coupling_block_per_layer):
                ccbs.append(ConditionalCouplingLayer(dim=4*4**i, cond_dim=cond_channels))
            self.coupling_blocks.append(ccbs)

        with torch.no_grad():
            dummy_y = torch.zeros(1, 1, self.img_dim, self.img_dim)
            dummy_c = torch.zeros(1, 1, self.img_dim, self.img_dim)
            z, _ = self.forward(dummy_y, dummy_c)
            self.z_shape = z.shape[1:]

    def forward(self, y, c):
        """
        Args:
            y: true data (B, C=1, H, W)
            c: condition data (B, C=1, H, W)
        Returns:
            z: latent (B, C=1, H, W)
            log_det: tensor (B,) aggregated log determinant
            tent_y: reconstructed y
        """
        B, C, H, W = y.shape
        assert C == 1, LOGGER.error("cINN: currently expects single-channel inputs")
        assert H == self.img_dim and W == self.img_dim, LOGGER.error("cINN: image size mismatch")

        enc_feats = self.encoder(c)
        total_log_det = torch.zeros(B, device=y.device)
        x = y

        x = self.downsample(x)

        for i in range(self.num_layers):
            cond = enc_feats[i+1]
            for block in self.coupling_blocks[i]:
                x, ld = block(x, cond, reverse=False)
                total_log_det += ld
            if i < self.num_layers - 1:
                x = self.downsample(x)

        #C, H, W = self.z_shape
        #reconstructed_y = self.inverse(torch.randn((B, C, H, W), device=self.device), c)

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
            cond = enc_feats[i+1]
            for block in reversed(self.coupling_blocks[i]):
                x, _ = block(x, cond, reverse=True)

        x = self.downsample(x, reverse=True)

        x = torch.clamp(x, -1., 1.)

        return x
