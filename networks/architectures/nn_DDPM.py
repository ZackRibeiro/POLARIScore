from typing import Optional, List
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
        self.norm1 = nn.GroupNorm(min(group_over, in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.dropout = nn.Dropout2d(p=0.05)

        self.time_mlp = None
        if time_emb_dim is not None:
            self.time_mlp = nn.Sequential(
                activation_function(),
                nn.Linear(time_emb_dim, out_channels * 2)
            )

        self.norm2 = nn.GroupNorm(min(group_over, out_channels), out_channels)
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

        h = self.dropout(h)

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

        self.norm = nn.GroupNorm(min(32,channels), channels)
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
    
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor):
        """
        timesteps: tensor shape (B)
        returns: (batch, dim)
        """
        assert timesteps.dim() == 1
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / float(half)
        )  # (half,)
        args = timesteps[:, None].float() * freqs[None, :]  # (B, half)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:  # pad if odd
            emb = F.pad(emb, (0, 1))
        return emb

class DDPMUnet(BaseModule):
    """"""
    def __init__(self, num_layers:int=4, base_filters:int=64, attention_layers:Optional[List[int]]=[3,4], attention_heads:Optional[List[int]]=None,time_emb_dim: int=256, filter_function="constant", init_method=nn.init.kaiming_uniform_):
        super(DDPMUnet, self).__init__()

        self.num_layers = num_layers
        self.in_channels = 2
        self.out_channels = 1
        self.init_method = init_method
        self.attention_layers = attention_layers
        self.attention_heads = attention_heads
        if attention_layers is not None and attention_heads is not None:
            assert len(self.attention_layers) == len(self.attention_heads), LOGGER.error("Attention heads given need to be the same length of attention_layers.")
        self.time_emb_dim = time_emb_dim
        self.filter_function = filter_function

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_emb_dim // 2),
            nn.Linear(time_emb_dim // 2, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        self.init_conv = nn.Conv2d(self.in_channels, base_filters, kernel_size=3, padding=1)
        if filter_function == "constant":
            filters = [base_filters * (2 ** i) for i in range(num_layers + 1)]
        elif filter_function == "linear":
            filters = [base_filters * (i+1) for i in range(num_layers + 1)]
        else:
            raise ValueError()

        self.enc_blocks = nn.ModuleList()
        self.enc_attn = nn.ModuleList()

        for i in range(num_layers):
            in_ch = filters[i]
            out_ch = filters[i+1]
            block = nn.Sequential(
                ResConvBlock(in_ch, out_ch, time_emb_dim=time_emb_dim),
                ResConvBlock(out_ch, out_ch, time_emb_dim=time_emb_dim),
            )
            self.enc_blocks.append(block)
            self.enc_attn.append(MHSAttentionBlock(out_ch) if (attention_layers and i in attention_layers) else nn.Identity())
        
        bottleneck_ch = filters[-1]
        self.bottleneck = nn.Sequential(
            ResConvBlock(bottleneck_ch, bottleneck_ch, time_emb_dim=time_emb_dim),
            MHSAttentionBlock(bottleneck_ch),
            ResConvBlock(bottleneck_ch, bottleneck_ch, time_emb_dim=time_emb_dim),
        )

        self.up_blocks = nn.ModuleList()
        self.up_attn = nn.ModuleList()
        for i in reversed(range(num_layers)):
            in_ch = filters[i+1]*2
            out_ch = filters[i]
            block = nn.Sequential(
                ResConvBlock(in_ch, out_ch, time_emb_dim=time_emb_dim),
                ResConvBlock(out_ch, out_ch, time_emb_dim=time_emb_dim),
            )
            self.up_blocks.append(block)
            self.up_attn.append(MHSAttentionBlock(out_ch, num_heads=self.attention_heads[self.attention_layers.index(i)-1] if self.attention_heads else 8) if (attention_layers and i in attention_layers) else nn.Identity())
    
        self.final_norm = nn.GroupNorm(8, filters[0]) if filters[0] % 8 == 0 else nn.GroupNorm(1, filters[0])
        self.final_act = nn.SiLU()
        self.final_conv = nn.Conv2d(filters[0], self.out_channels, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x, t):

        t_emb = self.time_embed(t) #(B, time_emb_dim)
        h = self.init_conv(x)
        skips = []

        for enc_block, attn in zip(self.enc_blocks, self.enc_attn):
            h = enc_block[0](h, t_emb)

            h = enc_block[1](h, t_emb)
            h = attn(h)
            skips.append(h)
            h = self.pool(h)

        h = self.bottleneck[0](h, t_emb)
        h = self.bottleneck[1](h)
        h = self.bottleneck[2](h, t_emb)

        for up_block, attn in zip(self.up_blocks, self.up_attn):
            h = self.upsample(h)
            skip = skips.pop()
            if skip.shape[-2:] != h.shape[-2:]:
                sh, sw = skip.shape[-2:]
                th, tw = h.shape[-2:]
                dh = (sh - th) // 2
                dw = (sw - tw) // 2
                skip = skip[:, :, dh:dh + th, dw:dw + tw]
            h = torch.cat([h, skip], dim=1)
            h = up_block[0](h, t_emb)
            h = up_block[1](h, t_emb)
            h = attn(h)

        h = self.final_norm(h)
        h = self.final_act(h)
        h = self.final_conv(h)

        return h
