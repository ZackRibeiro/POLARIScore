import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from POLARIScore.config import LOGGER
from POLARIScore.networks.architectures.nn_BaseModule import BaseModule

"""
Heavily inspired by 'https://apxml.com/courses/advanced-diffusion-architectures'

Redefine some blocks like ResConvBlock with time embedding and Attention Block but using multi head attention (different from gated ones).
And also DDPM_UNet, a UNet made for diffusion, i.e predicting the noise added at a step t.
"""

class ResConvBlock(nn.Module):
    """Residuals convolution block with time embedding"""
    def __init__(self, in_channels, out_channels, group_over=32, activation_function=nn.SiLU, time_emb_dim=None):
        super().__init__()
        self.norm1 = nn.GroupNorm(group_over, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.time_mlp = None
        if time_emb_dim is not None:
            self.time_mlp = nn.Sequential(
                activation_function(),
                nn.Linear(time_emb_dim, out_channels * 2)
            )

        self.norm2 = nn.GroupNorm(group_over, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.activation = activation_function()

        if in_channels != out_channels:
            self.match_dim = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.match_dim = nn.Identity()

    def forward(self, x, t_emb=None):
        residual = x

        h = self.norm1(x)
        h = self.activation(h)
        h = self.conv1(h)

        if self.time_mlp is not None and t_emb is not None:
            time_encoding = self.time_mlp(t_emb)
            time_encoding = time_encoding.view(h.shape[0], h.shape[1] * 2, 1, 1)
            scale, shift = torch.chunk(time_encoding, 2, dim=1)
            h = self.norm2(h) * (1 + scale) + shift # Modulate features
        else:
             h = self.norm2(h)

        h = self.activation(h)
        h = self.conv2(h)

        return h + self.match_dim(residual)
    
class MHSAttentionBlock(nn.Module):
    """
    Multi Head Self Attention
    """
    def __init__(self, channels, num_heads=8):
        super().__init__()
        assert channels % num_heads == 0, LOGGER.error(f"Channels ({channels}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = 1 / math.sqrt(self.head_dim)

        self.norm = nn.GroupNorm(32, channels)
        self.to_qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.to_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape
        residual = x

        qkv = self.to_qkv(self.norm(x))
        # Reshape for attention: (b, c, h, w) -> (b, c, h*w) -> (b, num_heads, head_dim, h*w) -> (b*num_heads, head_dim, h*w)
        q, k, v = qkv.chunk(3, dim=1)

        q = q.reshape(b * self.num_heads, self.head_dim, h * w)
        k = k.reshape(b * self.num_heads, self.head_dim, h * w)
        v = v.reshape(b * self.num_heads, self.head_dim, h * w)

        # Scaled dot-product attention
        # (b*num_heads, head_dim, h*w) @ (b*num_heads, h*w, head_dim) -> (b*num_heads, h*w, h*w)
        attention_scores = torch.bmm(q.transpose(1, 2), k) * self.scale
        attention_probs = F.softmax(attention_scores, dim=-1)

        # (b*num_heads, h*w, h*w) @ (b*num_heads, head_dim, h*w)' -> (b*num_heads, h*w, head_dim)
        out = torch.bmm(attention_probs, v.transpose(1, 2))

        # Reshape back: (b*num_heads, h*w, head_dim) -> (b, num_heads, h*w, head_dim) -> (b, c, h, w)
        out = out.transpose(1, 2).reshape(b, c, h, w)

        out = self.to_out(out)

        return out + residual
    
class DDPMUnet(BaseModule):
    """"""
    def __init__(self, num_layers:int=4, base_filters:int=64, attention_layer:int=2,init_method=nn.init.kaiming_uniform_):
        super(DDPMUnet, self).__init__()

        self.num_layers = num_layers
        self.in_channels = 1
        self.out_channels = 1
        self.init_method = init_method
        self.attention_layer = attention_layer

        filter_sizes = [int(base_filters * 2**i) for i in range(num_layers+1)]
        self.pool = nn.AvgPool2d(2,2)

        #Encoder
        self.encoders = nn.ModuleList()
        in_channels = self.in_channels
        for i in range(num_layers):
            out_channels = filter_sizes[i]
            self.encoders.append(ResConvBlock())
    

