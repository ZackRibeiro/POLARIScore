import torch
import torch.nn.functional as F
import math

def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    """Generates a linear schedule for beta_t."""
    return torch.linspace(beta_start, beta_end, timesteps)

def cosine_beta_schedule(timesteps, s=0.008):
    """Proposed in: https://arxiv.org/abs/2102.09672"""
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((t / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def quadratic_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    betas_quad = torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2
    return betas_quad