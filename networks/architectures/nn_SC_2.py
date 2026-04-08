from POLARIScore.config import LOGGER

import torch 
import torch.nn as nn
import torch.nn.functional as F
from POLARIScore.networks.architectures.nn_UNet import UNet, DoubleConvBlock
from POLARIScore.networks.architectures.nn_BaseModule import BaseModule

class SC_2(BaseModule):
    def __init__(self, encoder_layers=3, encoder_filters=32, latent_features=16, hidden_features=64, spectra_dim=128):

        self.latent_features = latent_features
        self.encoder_cdens = UNet(convBlock=DoubleConvBlock, num_layers=encoder_layers, base_filters=encoder_filters,
                                  in_channels=1, out_channels=latent_features)
        
        self.encoder_spectra = UNet(convBlock=DoubleConvBlock,num_layers=encoder_layers,base_filters=encoder_filters,
                                    in_channels=1, out_channels=1, is3D=True)
        self.out_spectra = nn.Conv2d(spectra_dim, latent_features, kernel_size=1, padding=0)

        self.decoder = nn.Sequential(
            nn.Conv2d(latent_features*2, hidden_features, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_features, hidden_features, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_features),
            nn.Conv2d(hidden_features, 1, kernel_size=1, padding=0),
        )

    def forward(self, *x):
        assert len(x) == 2
        cdens = x[0]
        spectra = x[1]

        z_cdens = self.encoder_cdens(cdens)
        z_spectra = self.encoder_spectra(spectra)
        z_spectra = self.out_spectra(torch.moveaxis(z_spectra, 4, 1).squeeze(-1))

        z = torch.cat([z_cdens, z_spectra], dim=1)
        x = self.decoder(z)
        return x