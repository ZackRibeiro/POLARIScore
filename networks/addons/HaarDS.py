import torch
import torch.nn as nn
import torch.nn.functional as F
from POLARIScore.config import LOGGER

class HaarDownsampling(nn.Module):
    """Invertible Haar wavelet downsampling (splits into 4 channels per input channel)"""
    def __init__(self):
        super().__init__()
        haar_weights = torch.tensor([
            [[1, 1], [1, 1]],     # LL
            [[1, -1], [1, -1]],   # LH
            [[1, 1], [-1, -1]],   # HL
            [[1, -1], [-1, 1]]    # HH
        ], dtype=torch.float32) / 2.0
        self.register_buffer("haar_weights", haar_weights[None, :, :, :])  # (1, 4, 2, 2)

    def forward(self, x, reverse=False):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            If not reverse: (B, 4C, H/2, W/2)
            If reverse: (B, C, H, W)
        """
        if not reverse:
            B, C, H, W = x.shape
            filters = self.haar_weights.repeat(C, 1, 1, 1)  # (C, 4, 2, 2)
            # group convolution: each input channel produces 4 outputs
            y = F.conv2d(x, filters, stride=2, padding=0, groups=C)
            y = y.view(B, 4 * C, H // 2, W // 2)
            return y
        else:
            # Inverse Haar = transposed convolution with the same filters
            B, C4, H, W = x.shape
            assert C4 % 4 == 0, LOGGER.error("Haar: channels must be multiple of 4 for inverse Haar")
            C = C4 // 4
            filters = self.haar_weights.repeat(C, 1, 1, 1)  # (C, 4, 2, 2)
            y = x.view(B, C, 4, H, W)
            y = y.view(B, C * 4, H, W)
            out = F.conv_transpose2d(y, filters, stride=2, groups=C)
            return out