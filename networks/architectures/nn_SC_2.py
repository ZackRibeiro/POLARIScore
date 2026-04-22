from POLARIScore.config import LOGGER

import torch 
import torch.nn as nn
import torch.nn.functional as F
from POLARIScore.networks.architectures.nn_UNet import UNet, DoubleConvBlock, ResConvBlock, ConvBlock
from POLARIScore.networks.architectures.nn_BaseModule import BaseModule

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

class SC_2(BaseModule):
    def __init__(self, encoder_layers=3, encoder_filters=32, latent_features=16, hidden_features=64, spectra_dim=128):

        super(SC_2, self).__init__()

        self.latent_features = latent_features
        self.encoder_cdens = UNet(convBlock=DoubleConvBlock, num_layers=encoder_layers, base_filters=encoder_filters,
                                  in_channels=1, out_channels=latent_features)
        self.encoder_spectra = SC_2_spectra_encoder_1D2DUNet(encoder_layers=encoder_layers, encoder_filters=encoder_filters, latent_features=latent_features, spectra_dim=spectra_dim)
        #Test to replace it by flatten and KAN
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_features*2, hidden_features, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_features, hidden_features, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_features),
            nn.Conv2d(hidden_features, 1, kernel_size=1, padding=0),
        )
        self.save_tensors = False

    def forward(self, *x):
        assert len(x) == 2
        cdens = x[0]
        spectra = x[1]

        z_cdens = torch.cat(self.encoder_cdens(cdens), dim=1)
        z_spectra = self.encoder_spectra(spectra)

        z = torch.cat([z_cdens, z_spectra], dim=1)
        if self.save_tensors:
            self.z = z
            self.x = cdens
        x = self.decoder(z)
        return x

    def plot_latent_space(self):
        latent_space = self.z.detach().cpu().numpy()  # (B, C, H, W)
        B, C, H, W = latent_space.shape
        _, ax2 = plt.subplots()
        ax2.imshow((self.x.detach().cpu().numpy())[0,0,:,:])

        b_idx = 0
        c_idx = 0

        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.25)

        img = ax.imshow(latent_space[b_idx, c_idx], cmap='viridis')
        ax.set_title(f"B={b_idx}, C={c_idx}")

        ax_b = plt.axes([0.2, 0.1, 0.65, 0.03])
        ax_c = plt.axes([0.2, 0.05, 0.65, 0.03])

        slider_b = Slider(ax_b, 'Batch (B)', 0, B - 1, valinit=b_idx, valstep=1)
        slider_c = Slider(ax_c, 'Channel (C)', 0, C - 1, valinit=c_idx, valstep=1)

        def update(val):
            b = int(slider_b.val)
            c = int(slider_c.val)

            img.set_data(latent_space[b, c])
            ax.set_title(f"B={b}, C={c}")
            fig.canvas.draw_idle()

        slider_b.on_changed(update)
        slider_c.on_changed(update)

class SC_2_spectra_encoder_3DUNet(BaseModule):
    def __init__(self, encoder_layers=3, encoder_filters=32, latent_features=16, spectra_dim=128):

        super(SC_2_spectra_encoder_3DUNet, self).__init__()

        self.encoder_spectra = UNet(convBlock=ConvBlock,num_layers=encoder_layers,base_filters=encoder_filters,
                                    in_channels=1, out_channels=1, dim=3, attention=False)
        self.out_spectra = nn.Conv2d(spectra_dim, latent_features, kernel_size=1, padding=0)
    def forward(self, spectra):
        z_spectra = self.encoder_spectra(spectra)
        z_spectra = self.out_spectra(torch.moveaxis(z_spectra, 4, 1).squeeze(2))
        return z_spectra

class SC_2_spectra_encoder_1D2DUNet(BaseModule):
    def __init__(self, encoder_layers=3, encoder_filters=32, latent_features=16, spectra_dim=128):

        super(SC_2_spectra_encoder_1D2DUNet, self).__init__()

        self.hidden_features = 8
        self.latent_features = latent_features
        self.velocity_encoder = UNet(in_channels=1, out_channels=1, num_layers=encoder_layers, base_filters=encoder_filters, dim=1, attention=False)
        self.position_encoder = UNet(in_channels=self.hidden_features, out_channels=latent_features, base_filters=encoder_filters, dim=2)

        self.out = DoubleConvBlock(spectra_dim, self.hidden_features)

    @torch.autocast(device_type="cuda")
    def forward(self, spectra):
        B, _, H, W, D = spectra.shape
        spectra_flatten = torch.reshape(spectra, (B*H*W, 1, D))
        chunk_size = 128
        outputs = []
        for i in range(0, B*H*W, chunk_size):
            chunk = spectra_flatten[i:i+chunk_size]
            output = self.velocity_encoder(chunk)
            if len(output[0].shape) < 3:
                output = [o.unsqueeze(0) for o in output]
            outputs.append(torch.cat(output, dim=1))
        s = torch.cat(outputs, dim=0)

        s = torch.reshape(s, (B, D, H, W))
        s = self.out(s)
        s = torch.cat(self.position_encoder(s), dim=1)
        #s = torch.reshape(s,(B, self.latent_features, H, W)) #wtf
        return s

