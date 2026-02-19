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
#from kan import KAN
from POLARIScore.networks.utils.fastkanconv import FastKANConvLayer
from typing import Optional, List

import matplotlib.pyplot as plt

class MultiNet(BaseModule):
    def __init__(self, convBlock=DoubleConvBlock, channel_dimensions=[2], channel_modes=[None] , num_layers=3, base_filters=32, attention = True, is3D=None):
        super(MultiNet, self).__init__()

        self.channel_dimensions = channel_dimensions if type(channel_dimensions) is list else [channel_dimensions]
        channel_is3D = []
        self.channel_modes = channel_modes.copy()
        self.channel_inchannels = [1 for _ in self.channel_dimensions]
        num_channels = 0
        for i,dim in enumerate(self.channel_dimensions):
            channel_is3D.append(True if dim == 3 else False)
            num_channels += 1
            if len(self.channel_modes) >= i+1:
                c_mode = self.channel_modes[i]
                if c_mode is None:
                    continue
                if not(type(c_mode) is list or type(c_mode) is tuple):
                    continue
                if type(c_mode) is tuple:
                    self.channel_modes[i] = list(c_mode)
                    c_mode = self.channel_modes[i]
                
                if "proj" in c_mode[0]:
                    self.channel_modes[i][1] = nn.Conv2d(in_channels=self.channel_modes[i][1], out_channels=1, kernel_size=1, device=self.device)
                elif "moments" in c_mode[0]:
                    self.channel_inchannels[i] = c_mode[1]+1
        self.num_channels = num_channels
        self.is3D = (True if True in channel_is3D else False) if is3D is None else is3D

        self.num_layers = num_layers
        self.attention = attention

        filter_sizes = [int(base_filters * 2**i) for i in range(num_layers+1)]

        self.pool2D = nn.MaxPool2d(2, 2)
        self.pool3D = nn.MaxPool3d(2, 2)

        #Channels Encoder
        self.channels_encoder = nn.ModuleList()
        self.channels_merger = nn.ModuleList()
        #self.channels_merger.append(JustKAN(in_channels=self.num_channels, out_channels=1))
        self.channels_merger.append(nn.Conv2d(in_channels=self.num_channels, out_channels=1, kernel_size=1))
        for j,is3D in enumerate(channel_is3D): #Loop over channels
            encoders = nn.ModuleList()
            in_channels = self.channel_inchannels[j]
            for i in range(num_layers): #Loop over layers
                out_channels = filter_sizes[i]
                encoders.append(convBlock(in_channels, out_channels, is3D=is3D))
                
                if j == 0: #Need just one list of mergers
                    #k = JustKAN(in_channels=(self.num_channels+1)*out_channels, out_channels=out_channels)
                    k = nn.Conv2d(in_channels=(self.num_channels+1)*out_channels, out_channels=out_channels, kernel_size=1)
                    self.channels_merger.append(k)
                
                in_channels = out_channels

            self.channels_encoder.append(encoders)


        #Main Encoder
        self.encoders = nn.ModuleList()
        in_channels = 1
        for i in range(num_layers):
            out_channels = filter_sizes[i]
            self.encoders.append(convBlock(in_channels, out_channels, is3D=self.is3D))
            in_channels = out_channels

        out_channels = filter_sizes[-1]

        # Bottleneck
        self.bottleneck = convBlock(in_channels, out_channels, is3D=self.is3D)
        in_channels = out_channels

        # Decoder
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.attentions = nn.ModuleList()
        reversed_filters = filter_sizes[::-1]

        for i in range(num_layers):
            out_channels = reversed_filters[1+i]
            if self.is3D:
                self.upconvs.append(nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2))
            else:
                self.upconvs.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2))

            if self.attention:
                self.attentions.append(GatedAttentionBlock(F_g=out_channels, F_l=out_channels, F_int=out_channels//2, is3D=self.is3D))
            self.decoders.append(convBlock(2*out_channels, out_channels, is3D=self.is3D))
            in_channels = out_channels

        # Output layer
        if self.is3D:
            self.final_conv = nn.Conv3d(base_filters, 1, kernel_size=1)
        else:
            self.final_conv = nn.Conv2d(base_filters, 1, kernel_size=1)

    def _compute_moments(self, v, v_axis=torch.linspace(-12.8, 12.8, 256, device="cuda" if torch.cuda.is_available() else "cpu").view(1, 1, 1, 1, 256), moments=2):
        if moments < 0:
            moments = 0
        moments_list = []
        for i in range(moments+1):
            moments_list.append((v_axis ** i * v).sum(dim=-1) / v.shape[-1])
        return moments_list
        
    def forward(self, *x):

        C = len(x)
        channels = x
        assert C == self.num_channels, LOGGER.error(f"Model trained with {self.num_channels} inputs but received {C} inputs")

        def _is3D(t):
            return len(t.shape) > 4
        def _convertToModelDimension(t,channel_index=None,return_only_one_channel=False):
            if not self.is3D:
                if _is3D(t):
                    cB, cC, cH, cW, cV = t.shape
                    if channel_index is None or len(self.channel_modes) < channel_index+1 or self.channel_modes[channel_index] is None or not(type(self.channel_modes[channel_index]) is list or type(self.channel_modes[channel_index]) is tuple):
                        return torch.sum(t, dim=-1)
                    channel_mode = self.channel_modes[channel_index]
                    if "split" in channel_mode[0]:
                        return torch.split(t, dim=-1)
                    if "proj" in channel_mode[0]:
                        assert cV == channel_mode[1].in_channels, LOGGER.error(f"Model can't work because you defined a channel projection on {channel_mode[1].in_channels} but the input tensor has a {cV} depth")
                        return channel_mode[1](t.view(cB, cC*cV, cH, cW))
                    if "moments" in channel_mode[0]:
                        if not(return_only_one_channel):
                            channel_momments = self._compute_moments(t,v_axis=torch.linspace(-12.8, 12.8, cV, device=self.device).view(1, 1, 1, 1, cV),moments=channel_mode[1])
                            return torch.cat(channel_momments, dim=1)
                    return torch.sum(t, dim=-1)
                return t
            if not _is3D(t):
                t = t.unsqueeze(-1)
                t = t.expand(-1,-1,-1, -1, t.shape[-2])
            elif t.shape[-1] != t.shape[-3]:
                t = torch.nn.functional.pad(t, (0, t.shape[-2] - t.shape[-1]))
            return t

        # list of enc_features for each channel
        channels_features=[]
        for i in range(C):
            xc = channels[i]
            channels_features.append([])
            if _is3D(xc) and self.channel_dimensions[i] == 2:
                xc = _convertToModelDimension(xc, channel_index=i)
            for j in range(self.num_layers):
                xc = self.channels_encoder[i][j](xc)
                channels_features[i].append(xc)
                xc = self.pool3D(xc) if _is3D(xc) else self.pool2D(xc)
        channels_features = list(map(list, zip(*channels_features)))

        #Multi Encoder
        enc_features = []

        x = torch.cat([_convertToModelDimension(c,channel_index=ci,return_only_one_channel=True) for ci,c in enumerate(channels)], dim=1)
        x = self.channels_merger[0](x)

        for i in range(self.num_layers):
            x = self.encoders[i](x)
            x = _convertToModelDimension(x)
            l = [x]
            l.extend([_convertToModelDimension(c) for c in channels_features[i]])
            x = torch.cat(l, dim=1)
            x = self.channels_merger[i+1](x)
            enc_features.append(x)
            x = self.pool3D(x) if self.is3D else self.pool2D(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Main Decoder
        for i in range(self.num_layers):
            x = self.upconvs[i](x)
            if self.attention:
                enc_features[-(i+1)] = self.attentions[i](x, enc_features[-(i+1)])
            x = torch.cat([x, enc_features[-(i+1)]], dim=1)
            x = self.decoders[i](x)
        
        # Output
        f_x = self.final_conv(x)
        return f_x if self.is3D else f_x
    
    def plot_channel_weights(self, ax=None, channel_names:Optional[List[str]]=None, cmap:Optional[str]=None):
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        num_layers = len(self.channels_merger)
        num_channels = self.num_channels

        all_importances = []

        for layer_idx, merger in enumerate(self.channels_merger):

            conv = None
            for m in merger.modules():
                if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                    conv = m
                    break

            if conv is None:
                raise RuntimeError("No convolution found in merger block")

            weight = conv.weight.data  # shape: [out_channels, in_channels, 1, 1]
            importance = weight.abs().mean(dim=0)
            importance = importance.view(-1)

            if layer_idx == 0:
                importance = importance[:num_channels]
            else:
                chunk_size = importance.shape[0] // (num_channels + 1)
                importance = importance[chunk_size:].view(num_channels, chunk_size).mean(dim=1)

            all_importances.append(importance.cpu().numpy())

        all_importances = np.array(all_importances)

        x = np.arange(num_layers)
        width = 0.8 / num_channels

        if cmap is not None:
            colormap = plt.get_cmap(cmap)
            colors = [colormap(i / max(num_channels - 1, 1)) for i in range(num_channels)]
        else:
            colors = [None] * num_channels

        for c in range(num_channels):
            ax.bar(
                x + c * width,
                all_importances[:, c],
                width,
                label=f"Channel {c}" if channel_names is None else channel_names[c],
                color=colors[c]
            )

        ax.set_xlabel("Layer / Depth")
        ax.set_ylabel("Mean |Weight|")
        ax.set_xticks(x + width * (num_channels - 1) / 2)
        ax.set_xticklabels([f"L{i}" for i in range(num_layers)])
        ax.legend()

        return ax
    
if __name__ == "__main__":
    model = MultiNet(channel_dimensions=[2,2], channel_modes=[None,("moments",2)])
    model.cuda()
    x = torch.randn(1, 1, 128, 128).cuda()
    y = torch.randn(1, 1, 128, 128, 256).cuda()
    print(model(x,y).shape)