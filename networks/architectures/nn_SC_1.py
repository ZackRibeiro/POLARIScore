import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from POLARIScore.config import LOGGER

import torch
import torch.nn as nn
import torch.nn.functional as F
from POLARIScore.networks.architectures.nn_UNet import ConvBlock, DoubleConvBlock, GatedAttentionBlock
from POLARIScore.networks.architectures.nn_BaseModule import BaseModule
import numpy as np
from typing import Optional, List
import matplotlib.pyplot as plt
from kan import KAN, MultKAN
from POLARIScore.networks.architectures.nn_MultiNet import MultiNet
from typing import List, Tuple, Dict, Optional, Literal

class SC_1(BaseModule):
    def __init__(self, num_layers=3, base_filters=32, gaussian_features:int=10):
        super(SC_1, self).__init__()
        self.num_layers = num_layers
        self.base_filters = base_filters

        channel_dimensions = [2].extend([2] for _ in range(gaussian_features))
        self.multinet = MultiNet(channel_dimensions=channel_dimensions, num_layers=num_layers, base_filters=base_filters)

        self.gaussian_encoder = GaussianEncoder(features_number=gaussian_features)
    def forward(self, *x):
        assert len(x) == 2
        column_density = x[0]
        gaussian_cube = x[1]

        gaussian_features = self.gaussian_encoder(gaussian_cube) #shape B, C_gaussian, H, W
        
        x = torch.cat([column_density, gaussian_features], dim=1)
        x = self.multinet(x)

        return x

class GaussianEncoder(BaseModule):
    """For now, this compute moments like quantities using the gaussian components."""
    def __init__(self, features_number=5, kan_outputs=2, merge_mode:Literal['sum','prod']="sum"):
        self.kans_layer1 = nn.ModuleList()
        self.kans_layer2 = nn.ModuleList()
        self.merge_mod = merge_mode
        self.features_number = features_number
        self.kan_outputs = kan_outputs
        for _ in range(features_number):
            self.kans_layer1.append(
                MultKAN(width=[1, [5,3], [5,3], kan_outputs], grid=5, k=3, device='cuda' if torch.cuda.is_available() else 'cpu', auto_save=False)
            )
            self.kans_layer2.append(
                MultKAN(width=[kan_outputs, [2,1], [2,1], 1], grid=5, k=3, device='cuda' if torch.cuda.is_available() else 'cpu', auto_save=False)
            )
    def forward(self, x):

        B, C, H, W, G = x.shape
        assert C == 1
        assert G % 3 == 0, LOGGER.error("Gaussian cube need to be divisible by 3.")
        gaussians_number = G//3


        features = []
        for f_i in range(self.features_number):
            if self.merge_mode == 'sum':
                y_merged = torch.zeros((B*H*W,self.kan_outputs))
            elif self.merge_mode == 'prod':
                y_merged = torch.ones((B*H*W,self.kan_outputs))

            for g_i in range(gaussians_number):
                y = self.kans_layer1[f_i](x.reshape(B*H*W, G)[:,g_i*3:(g_i+1)*3])
                if self.merge_mode == 'sum':
                    y_merged = y_merged + y
                elif self.merge_mode == 'prod':
                    y_merged = y_merged * y
            
            y = self.kans_layer2[f_i](y_merged)
            features.append(y.reshape(B,1,H,W))
        
        return torch.cat(features, dim=1)

