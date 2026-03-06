from POLARIScore.config import LOGGER

import torch
import torch.nn as nn
import torch.nn.functional as F
from POLARIScore.networks.architectures.nn_BaseModule import BaseModule
from typing import List, Tuple, Dict, Optional, Literal
from POLARIScore.networks.addons.FiLM import FiLMGenerator

class SpectraNetwork(BaseModule):
    def __init__(self, num_layers=3, out_features=10*3, base_filters=32, environment_dim=3, spectra_dim=128 ):
        super(SpectraNetwork, self).__init__()
        self.num_layers = num_layers
        self.out_features = out_features
        self.base_filters = base_filters
        self.environment_dim = environment_dim
        self.spectra_dim = spectra_dim

        self.conv_first = nn.Sequential(
            nn.Conv3d(1, base_filters, kernel_size=(environment_dim,environment_dim,3), padding=(0,0,1)),
            nn.BatchNorm3d(base_filters),
            nn.ReLU(),
            nn.Conv3d(base_filters, base_filters, kernel_size=(1,1,3), padding=(0,0,1)),
            nn.BatchNorm3d(base_filters),
            nn.ReLU(),
            nn.Dropout3d(p=0.01)
        )

        filter_sizes = [int(base_filters * 2**(i+1)) for i in range(num_layers)]
        self.encoder = nn.ModuleList()
        in_channels = base_filters
        for i in range(num_layers):
            out_channels = filter_sizes[i]
            self.encoder.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                    nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                    nn.Dropout1d(p=0.01)
                )
            )
            in_channels = out_channels
        out_channels = filter_sizes[-1]

        self.film_snr = FiLMGenerator(1, filter_sizes)
        self.pool = nn.MaxPool1d(2)

        self.conv_final = nn.Sequential(
            nn.Conv1d(filter_sizes[-1]*(spectra_dim//(2**(num_layers-1))), out_features, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
            nn.Conv1d(out_features, out_features,kernel_size=3, padding=1),
            nn.Threshold(threshold=0.1, value=0.1)
        )

    def forward(self, *y:List[torch.tensor]):
        """
        Input shape: Batch,1,Environment_dim,Environment_dim,Spectra_dim ; Batch, 1
        Output shape: Batch, 1, Num_components*3
        """

        snr = y[1]
        x = y[0]
        B,C,H,W,D = x.shape
        assert H==W
        assert H==self.environment_dim, LOGGER.error(f"Environment dim(Width and Height) specified in network is {self.environment_dim} but the given tensor has a dim of {H}")
        assert D==self.spectra_dim, LOGGER.error(f"Spectra dim(Depth) specified in network is {self.spectra_dim} but the given tensor has a dim of {D}")
        
        x = self.conv_first(x)
        x=x.squeeze(2).squeeze(2)

        film_snr_params = self.film_snr(snr)

        for i in range(self.num_layers):
            x = self.encoder[i](x)
            gamma, beta = film_snr_params[i]
            x = gamma.squeeze(-1)*x+beta.squeeze(-1)
            if i<self.num_layers-1:
                x = self.pool(x)
        x = x.reshape(B, x.shape[1]*x.shape[2], 1)

        x = self.conv_final(x)
        x = x.reshape(B, 1, x.shape[1])
        if len(y) >= 3:
            return x, y[2]
        else:
            return x