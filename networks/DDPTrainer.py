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
                 timesteps:int=1000, beta_schedule:Literal["linear","cosine","quadratic"]="linear", pred_type:Literal["epsilon","v","x0"]="v", beta_start:float=1e-4, beta_end:float=0.02,
                  **kwargs):
        """
        To know args and variables, please refers to '~.Trainer.__init__'
        """
        super(DDPTrainer, self).__init__(*args, **kwargs)

        self.timesteps = timesteps
        self.pred_type = pred_type

        self.set_scheduler(timesteps=timesteps, beta_schedule=beta_schedule,
                           beta_start=beta_start, beta_end=beta_end)        
        
        self.loss_method = self.mse_loss
        self.validation_loss_method = nn.MSELoss()

    def _modify_saved_settings(self, settings):
        settings = super()._modify_saved_settings(settings)
        settings["ddpm_timesteps"] = self.timesteps
        settings["ddpm_pred_type"] = self.pred_type
        settings["ddpm_beta_schedule"] = self.beta_schedule
        settings["ddpm_beta_start"] = self.beta_start
        settings["ddpm_beta_end"] = self.beta_end
        return settings
    
    def _modify_loaded_settings(self, settings):
        super()._modify_loaded_settings(settings)

        self.timesteps = settings["ddpm_timesteps"] if "ddpm_timesteps" in settings else 1000
        self.pred_type = settings["ddpm_pred_type"] if "ddpm_pred_type" in settings else "v"
        self.beta_schedule = settings["ddpm_beta_schedule"] if "ddpm_beta_schedule" in settings else "linear"
        self.beta_start = settings["ddpm_beta_start"] if "ddpm_beta_start" in settings else 1e-4
        self.beta_end = settings["ddpm_beta_end"] if "ddpm_beta_end" in settings else 0.02

        self.set_scheduler(self.timesteps, self.beta_schedule, self.beta_start, self.beta_end)

    def set_scheduler(self, timesteps:int=1000, beta_schedule:Literal["linear","cosine","quadratic"]="linear"
                      , beta_start:float=1e-4, beta_end:float=0.02):
        self.timesteps = timesteps
        self.beta_schedule = beta_schedule
        self.beta_start = beta_start
        self.beta_end = beta_end
        if beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps, beta_start=beta_start, beta_end=beta_end)
        elif beta_schedule == "quadratic":
            betas = quadratic_beta_schedule(timesteps, beta_start=beta_start, beta_end=beta_end)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            LOGGER.error(beta_schedule+" is not an option.")
            raise ValueError("Beta schedule incorrect")
        
        self.betas = betas.float().to(self.device)
        self.alphas = 1.-self.betas #(T,)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0) #(T,)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1,0), value=1.) # (T,)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

        return self.betas


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

        half = (B + 1) // 2
        t_half = torch.randint(0, self.timesteps, (half,), device=self.device)
        t = torch.cat([t_half, self.timesteps - t_half - 1],dim=0)[:B]
        perm = torch.randperm(B, device=self.device)
        t = t[perm]

        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise=noise)

        if(self.pred_type == "epsilon"):
            return model(torch.cat([xt, input], dim=1), t), noise
        elif(self.pred_type == "v"):
            sqrt_alpha_bar = self.sqrt_alphas_cumprod.to(self.device)[t].view(B, 1, 1, 1)
            sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod.to(self.device)[t].view(B, 1, 1, 1)

            v_target = sqrt_alpha_bar * noise - sqrt_one_minus_alpha_bar * x0
            v_pred = model(torch.cat([xt, input], dim=1), t)

            return v_pred, v_target
        elif(self.pred_type == "x0"):
            return model(torch.cat([xt, input], dim=1), t), x0

        return model(torch.cat([xt, input], dim=1), t), noise

    def _infer_model(self, model, input):
        # input: (B, C, H, W)
        if isinstance(input, list):
            input = input[0]
                
        B, C, H, W = input.shape
        eta = 1.
        seq = range(0, self.timesteps, 1)

        if seq is None:
            seq = list(range(self.timesteps))

        seq_next = [-1] + list(seq[:-1])

        with torch.no_grad():
            x_t = torch.randn((B, C, H, W), device=self.device)

            for i, j in zip(reversed(seq), reversed(seq_next)):
                t = torch.full((B,), i, device=self.device, dtype=torch.long)
                next_t = torch.full((B,), j, device=self.device, dtype=torch.long)

                at = self.alphas_cumprod[t].view(B, 1, 1, 1)

                if j == -1:
                    at_next = torch.ones_like(at)
                else:
                    at_next = self.alphas_cumprod[next_t].view(B, 1, 1, 1)

                sqrt_at = torch.sqrt(at)
                sqrt_one_minus_at = torch.sqrt(1.0 - at)

                model_in = torch.cat([x_t, input], dim=1)

                if self.pred_type == "epsilon":
                    eps = model(model_in, t)
                    x0 = (x_t - sqrt_one_minus_at * eps) / (sqrt_at + 1e-5)

                elif self.pred_type == "v":
                    v = model(model_in, t)
                    x0 = sqrt_at * x_t - sqrt_one_minus_at * v
                    eps = (x_t - sqrt_at * x0) / (sqrt_one_minus_at + 1e-5)

                elif self.pred_type == "x0":
                    x0 = model(model_in, t)
                    eps = (x_t - sqrt_at * x0) / (sqrt_one_minus_at + 1e-5)

                #x0 = x0.clamp(-1.5, 1.5)

                if j == -1:
                    x_t = x0
                    break

                c1 = (eta* torch.sqrt((1 - at / at_next)* (1 - at_next)/ (1 - at)))
                c2 = torch.sqrt((1 - at_next) - c1 ** 2)

                noise = torch.randn_like(x_t)

                x_t = (torch.sqrt(at_next) * x0 + c1 * noise + c2 * eps)

            return x_t.clamp(-1.0, 1.0)
    
    @staticmethod
    def load(model_name, load_model=True):
        return load_trainer(model_name, load_model, trainer_class=DDPTrainer)
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from POLARIScore.objects.Dataset import getDataset
    from POLARIScore.config import DATA_NORMALIZATION_CDENS, DATA_NORMALIZATION_VDENS, DATA_NORMALIZATION_CDENS_TORCH, DATA_NORMALIZATION_VDENS_TORCH
    ds1 = getDataset("batch_highres_2_b1")
    ds2 = getDataset("batch_highres_2_b2")

    #ds = getDataset("batch_highres_2")
    #ds2, _ = ds2.split(0.5)

    def classic_log_mse(output, target):
        output_phys = DATA_NORMALIZATION_VDENS_TORCH[1](output)
        target_phys = DATA_NORMALIZATION_VDENS_TORCH[1](target)
        output_log = torch.log(output_phys)
        target_log = torch.log(target_phys)
        mse = torch.mean((output_log - target_log) ** 2)
        return mse


    #trainer = DDPTrainer(DDPMUnet, ds1, ds2, model_name="BigDDPM", timesteps=1000, beta_schedule='linear')
    trainer = load_trainer("BigDDPM", trainer_class=DDPTrainer)
    #trainer.pred_type = "epsilon"
    trainer.norms = { 
        "cdens": DATA_NORMALIZATION_CDENS,
        "vdens": DATA_NORMALIZATION_VDENS,
    }

    #trainer.get_validation_error()

    trainer.ema = True
    trainer.validation_loss_method = classic_log_mse
    trainer.ema_warmup = 0
    trainer.learning_rate = 1e-4
    trainer.training_set = ds1
    trainer.validation_set = ds2
    trainer.network_settings["base_filters"] = 64
    trainer.network_settings["num_layers"] = 4
    #trainer.network_settings["filter_function"] = "linear"
    trainer.training_random_transform = True
    trainer.optimizer_name = "Adam"
    trainer.target_names = ["vdens"]
    trainer.input_names = ["cdens"]
    trainer.auto_save = 500
    trainer.scheduler = None
    #trainer.init()
    trainer.get_validation_error()
    #trainer.train(1000,batch_number=8,compute_validation=100,early_stopping=False)
    #trainer.save()

    trainer.plot(save=True)
    trainer.plot_validation(save=True, number=8, number_per_row=4)

    plt.show()