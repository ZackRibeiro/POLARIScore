from POLARIScore.config import LOGGER
import torch
import torch.nn as nn
from typing import List, Literal, Tuple
try:
    from kan import KAN
    kan_available = False
except ImportError:
    kan_available = True


class FiLMGenerator(nn.Module):
    def __init__(self, in_channels:int, film_dim_list:List[int], used_network:Literal["mlp","kan"]="mlp"):
        super().__init__()
        if(used_network=="mlp"):
            self.mlp = nn.Sequential(
                nn.Linear(in_channels, 256),
                nn.ReLU(),
                nn.Linear(256, sum(2*d for d in film_dim_list))  # gamma and beta for each layer
            )
        elif(used_network=="kan") and kan_available:
            self.mlp = KAN(width=[in_channels,16,16,sum(2*d for d in film_dim_list)], grid=5, k=3, seed=1, device='cuda' if torch.cuda.is_available() else 'cpu', auto_save=False)
        self.layer_dims = film_dim_list
    def forward(self, x)->List[Tuple]:
        """
        Args:
            x: tensor shape: (B,C)
        """
        params = self.mlp(x)
        out = []
        idx = 0
        for d in self.layer_dims:
            g = params[:, idx:idx+d]; idx += d
            b = params[:, idx:idx+d]; idx += d
            out.append((g.unsqueeze(-1).unsqueeze(-1), b.unsqueeze(-1).unsqueeze(-1)))
        return out