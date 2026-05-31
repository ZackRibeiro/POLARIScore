import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from POLARIScore.config import LOGGER
import torch
import torch.nn as nn
import torch.nn.functional as F
from POLARIScore.networks.architectures.nn_UNet import DoubleConvBlock, GatedAttentionBlock
from POLARIScore.networks.architectures.nn_BaseModule import BaseModule
from POLARIScore.networks.utils.nn_utils import init_network
from torch.nn import init
from typing import List, Literal, Tuple
from POLARIScore.networks.addons.FiLM import FiLMGenerator

#tensors shape B,C,H,W ; no third axis
class SizeAwareUNet(BaseModule):
    """
    Basic Attention U-Net and size aware (in reality it can be any other 1D parameter instead of physical size)
    """
    def __init__(self,num_layers:int=4, base_filters:int=64, init_method=init.kaiming_uniform_):
        super(SizeAwareUNet, self).__init__()

        self.num_layers = num_layers
        self.in_channels = 1
        self.out_channels = 1
        self.init_method = init_method

        filter_sizes = [int(base_filters * 2**i) for i in range(num_layers+1)]
        self.pool = nn.MaxPool2d(2,2)

        #Encoder
        self.encoders = nn.ModuleList()
        in_channels =  self.in_channels
        for i in range(num_layers):
            out_channels = filter_sizes[i]
            self.encoders.append(DoubleConvBlock(in_channels, out_channels, init_method=self.init_method))
            in_channels = out_channels
        out_channels = filter_sizes[-1]

        # Bottleneck
        self.bottleneck = DoubleConvBlock(in_channels, out_channels, init_method=self.init_method)
        in_channels = out_channels

        #FiLM learned using physical size
        self.film_physize = FiLMGenerator(1, filter_sizes[:-1])

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.attentions = nn.ModuleList()
        reversed_filters = filter_sizes[::-1]
        for j in range(self.out_channels):
            self.upconvs.append(nn.ModuleList())
            self.decoders.append(nn.ModuleList())
            self.attentions.append(nn.ModuleList())

            in_channels = filter_sizes[-1]
            for i in range(num_layers):
                out_channels = reversed_filters[1+i]
                self.upconvs[j].append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2))
                self.attentions[j].append(GatedAttentionBlock(F_g=out_channels, F_l=out_channels, F_int=out_channels//2, init_method=self.init_method))
                concat_ch = out_channels * 2
                block = DoubleConvBlock(concat_ch, out_channels, init_method=self.init_method)
                self.decoders[j].append(block)
                in_channels = out_channels

        # Output layer
        self.final_conv = nn.ModuleList()
        for o in range(self.out_channels):
            self.final_conv.append(nn.Conv2d(base_filters, 1, kernel_size=1))

        self.initialize()
    
    def initialize(self):
        for f in self.final_conv:
            init_network(self,self.init_method)
            init.zeros_(f.bias)

    def forward(self, *x:List[torch.tensor]):
        """
        Args
            x: tensor shape [(B,1,H,W),(B,1)] ; i.e [region, physical_size]
        """
        size_x = torch.ones((1, 1, 1), device=self.device)*3. #x[1]
        #size_x = x[1]
        x = x[0]

        # Encoders forward pass
        enc_features = []
        for i in range(self.num_layers):
            x = self.encoders[i](x)
            enc_features.append(x)
            x = self.pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)

        film_physize_params = self.film_physize(size_x)
        
        # Decoder forward pass with film modulation
        decoded_x = []
        for j in range(self.out_channels):
            xj = x
            for i in range(self.num_layers):
                xj = self.upconvs[j][i](xj)
                enc_feat = enc_features[-(i+1)]
                enc_feat = self.attentions[j][i](xj, enc_feat)
                skip_feats = [xj, enc_feat]
                xj = torch.cat(skip_feats, dim=1)
                gamma, beta = film_physize_params[::-1][i]
                xj = gamma*self.decoders[j][i](xj)+beta
                
            decoded_x.append(xj)
        
        # Output
        rslt = self.final_conv[0](decoded_x[0])
        return rslt