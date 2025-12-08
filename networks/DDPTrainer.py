from click import Option

import torch
import torch.nn as nn
import torch.nn.functional as F

from POLARIScore.config import LOGGER
from .Trainer import Trainer
from typing import Literal, Union, Optional, List, Tuple, Dict
from POLARIScore.networks.addons.NoiseScheduler import *

class DDPTrainer(Trainer):
    """
    Extension of Trainer class to train Denoising Diffusion Probabilistic Models.
    """
    def __init__(self, *args,
                 timesteps:int=1000, beta_schedule:Literal["linear","cosine","quadratic"], beta_start:float=1e-4, beta_end:float=0.02,
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
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0) #(T,)
        self.alphas_cumprod_prev = F.pad(self.alpha_cumprod[:-1], (1,0), value=1.) # (T,)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

        self.loss_method = self.mse_loss
        self.validation_loss_method = nn.MSELoss()

    @staticmethod
    def mse_loss(output, target):
        return F.mse_loss(output, target[:,])

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

        return model(torch.cat([xt, input], dim=1), t)

    def _infer_model(self, model, input):
        if(type(input) is list):
            input = input[0]
        
        B, C, H, W = input.shape
        x_t = torch.randn((B,C,H,W), device=self.device)

        for time_step in reversed(range(self.timesteps)):
            t = torch.full((B,), time_step, device=self.device, dtype=torch.long)

            output_model = model(torch.cat([input, x_t], dim=1), t)
            
            pred_epsilon = output_model
            sqrt_ac = self.sqrt_alphas_cumprod.to(self.device)[t].view(-1, 1, 1, 1)
            sqrt_omac = self.sqrt_one_minus_alphas_cumprod.to(self.device)[t].view(-1, 1, 1, 1)
            pred_x0 = (x_t - sqrt_omac * pred_epsilon) / (sqrt_ac + 1e-8)

        if time_step == 0:
                x_t = pred_x0
        else:
            beta_t = self.betas.to(self.device)[t].view(-1, 1, 1, 1)
            sqrt_one_minus_ac_t = sqrt_omac
            alpha_t = self.alphas.to(self.device)[t].view(-1, 1, 1, 1)
            alpha_cumprod_t = self.alphas_cumprod.to(self.device)[t].view(-1, 1, 1, 1)
            alpha_cumprod_prev_t = self.alphas_cumprod_prev.to(self.device)[t].view(-1, 1, 1, 1)
            posterior_variance_t = self.posterior_variance.to(self.device)[t].view(-1, 1, 1, 1)

            # formula for posterior mean mu_t = (sqrt(alpha_cumprod_prev) * beta_t / (1-alpha_cumprod)) * x0
            #                        + (sqrt(alpha_t) * (1 - alpha_cumprod_prev) / (1-alpha_cumprod)) * x_t
            # but simpler and numerically stable version from DDPM code:
            coef1 = (beta_t * torch.sqrt(alpha_cumprod_prev_t)) / (1.0 - alpha_cumprod_t)
            coef2 = (torch.sqrt(alpha_t) * (1.0 - alpha_cumprod_prev_t)) / (1.0 - alpha_cumprod_t)
            posterior_mean = coef1 * pred_x0 + coef2 * x_t

            # sample from N(posterior_mean, posterior_variance)
            noise = torch.randn_like(x_t)
            x_t = posterior_mean + torch.sqrt(posterior_variance_t.clamp(min=1e-20)) * noise

        return x_t.clamp(-1.0, 1.0)


    



