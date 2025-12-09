import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from POLARIScore.config import LOGGER
from .Trainer import Trainer, load_trainer
from typing import Literal, Union, Optional, List, Tuple, Dict
from POLARIScore.networks.addons.NoiseScheduler import *
from POLARIScore.networks.architectures.nn_DDPM import DDPMUnet

class DDPTrainer(Trainer):
    """
    Extension of Trainer class to train Denoising Diffusion Probabilistic Models.
    """
    def __init__(self, *args,
                 timesteps:int=1000, beta_schedule:Literal["linear","cosine","quadratic"]="linear", beta_start:float=1e-4, beta_end:float=0.02,
                  **kwargs):
        """
        To know args and variables, please refers to '~.Trainer.__init__'
        """
        super(DDPTrainer, self).__init__(*args, **kwargs)

        self.timesteps = timesteps

        if beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps, beta_start=beta_start, beta_end=beta_end)
        elif beta_schedule == "quadratic":
            betas = quadratic_beta_schedule(timesteps, beta_start=beta_start, beta_end=beta_end)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps, beta_start=beta_start, beta_end=beta_end)
        else:
            LOGGER.error(beta_schedule+" is not an option.")
            raise ValueError("Beta schedule incorrect")
        
        self.betas = betas.float() #(T,)
        self.alphas = 1.-self.betas #(T,)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0) #(T,)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1,0), value=1.) # (T,)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

        self.loss_method = self.mse_loss
        self.validation_loss_method = nn.MSELoss()

    @staticmethod
    def mse_loss(output, target):
        return F.mse_loss(output, target)

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None):
        """
        Diffuse the data (sample from q(x_t | x_0))
        Args:
            x_start: (B, C, H, W)
            t: (B,) timesteps indices in [0, T-1]
            noise: optional noise, same shape as x_start
        Returns
            x_t
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        device = x_start.device
        sqrt_ac = self.sqrt_alphas_cumprod.to(device)[t].view(-1, 1, 1, 1)
        sqrt_omac = self.sqrt_one_minus_alphas_cumprod.to(device)[t].view(-1, 1, 1, 1)
        return sqrt_ac * x_start + sqrt_omac * noise

    def _train_model(self, model, input, target):
        if(type(input) is list):
            input = input[0]
        if(type(target) is list):
            target = target[0]

        x0 = target.to(self.device)
        B = x0.shape[0]

        t = torch.randint(0, self.timesteps, (B,), device=self.device).long()
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise=noise)

        return model(torch.cat([xt, input], dim=1), t), noise

    def _infer_model(self, model, input):
        # input: (B, C, H, W)
        if isinstance(input, list):
            input = input[0]

        B, C, H, W = input.shape
        device = self.device

        x_t = torch.randn((B, C, H, W), device=device)

        for time_step in reversed(range(self.timesteps)):
            t = torch.full((B,), time_step, device=device, dtype=torch.long)

            eps_pred = model(torch.cat([x_t, input], dim=1), t)

            sqrt_ac = self.sqrt_alphas_cumprod[t].view(B, 1, 1, 1)
            sqrt_omac = self.sqrt_one_minus_alphas_cumprod[t].view(B, 1, 1, 1)
            x0_pred = (x_t - sqrt_omac * eps_pred) / (sqrt_ac + 1e-8)

            x0_pred = x0_pred.clamp(-1.0, 1.0)

            if time_step == 0:
                x_t = x0_pred
                break

            beta_t      = self.betas[t].view(B, 1, 1, 1)
            alpha_t     = self.alphas[t].view(B, 1, 1, 1)
            alpha_bar_t = self.alphas_cumprod[t].view(B, 1, 1, 1)
            alpha_bar_prev = self.alphas_cumprod_prev[t].view(B, 1, 1, 1)
            posterior_var = self.posterior_variance[t].view(B, 1, 1, 1)

            coef1 = (beta_t * torch.sqrt(alpha_bar_prev)) / (1.0 - alpha_bar_t)
            coef2 = (torch.sqrt(alpha_t) * (1.0 - alpha_bar_prev)) / (1.0 - alpha_bar_t)
            posterior_mean = coef1 * x0_pred + coef2 * x_t

            noise = torch.randn_like(x_t)
            x_t = posterior_mean + torch.sqrt(posterior_var.clamp(min=1e-20)) * noise

            x_t = x_t.clamp(-1.5, 1.5)

        return x_t.clamp(-1.0, 1.0)
    
    @staticmethod
    def load(model_name, load_model=True):
        return load_trainer(model_name, load_model, trainer_class=DDPTrainer)
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from POLARIScore.objects.Dataset import getDataset
    from POLARIScore.config import DATA_NORMALIZATION_CDENS, DATA_NORMALIZATION_VDENS, DATA_NORMALIZATION_CDENS_TORCH, DATA_NORMALIZATION_VDENS_TORCH
    ds1 = getDataset("batch_training")
    ds2 = getDataset("batch_validation")

    def classic_log_mse(output, target):
        output_phys = DATA_NORMALIZATION_VDENS_TORCH[1](output)
        target_phys = DATA_NORMALIZATION_VDENS_TORCH[1](target)
        output_log = torch.log(output_phys)
        target_log = torch.log(target_phys)
        mse = torch.mean((output_log - target_log) ** 2)
        return mse


    trainer = DDPTrainer(DDPMUnet, ds1, ds2, model_name="DDPM", timesteps=1000, beta_schedule='linear')
    #trainer = load_trainer("DDPM", trainer_class=DDPTrainer)
    trainer.norms = {
        "cdens": DATA_NORMALIZATION_CDENS,
        "vdens": DATA_NORMALIZATION_VDENS,
    }
    #ds = getDataset("batch_highres_2")
    #ds1, ds2 = ds.split(0.8)
    #trainer.get_validation_error()

    #trainer.ema = True
    trainer.validation_loss_method = classic_log_mse
    #trainer.ema_warmup = 500
    trainer.learning_rate = 1e-3
    trainer.training_set = ds1
    trainer.validation_set = ds2
    trainer.network_settings["base_filters"] = 32
    trainer.network_settings["num_layers"] = 4
    trainer.training_random_transform = True
    trainer.optimizer_name = "Adam"
    trainer.target_names = ["vdens"]
    trainer.input_names = ["cdens"]
    trainer.auto_save = 250
    trainer.scheduler = ReduceLROnPlateau(trainer.optimizer, 'min', patience=10, factor=0.1, threshold=0.0001)
    trainer.init()
    trainer.train(500,batch_number=8,compute_validation=100,early_stopping=False)
    trainer.save()

    trainer.plot(save=True)
    trainer.plot_validation(save=True, number=8, number_per_row=4)

    plt.show()