import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from POLARIScore.config import LOGGER
from .Trainer import Trainer, load_trainer
from typing import Literal, Union, Optional, List, Tuple, Dict
from POLARIScore.networks.addons.NoiseScheduler import *
from POLARIScore.networks.architectures.nn_DDPM import DDPMUnet
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import matplotlib.axes
from POLARIScore.utils.utils import compute_pdf, printProgressBar
from scipy.stats import wasserstein_distance
from scipy.ndimage import gaussian_filter
from POLARIScore.utils.utils import plot_map, plot_rect_bg


import matplotlib.colors as mcolors
def _symmetrical_cmap(cmap_settings, n=128):
    cmap = plt.cm.get_cmap(*cmap_settings)
    new_name = "sym_"+cmap_settings[0]
    colors_r = cmap(np.linspace(0, 1, n))
    colors_l = colors_r[::-1]
    colors = np.vstack((colors_l, colors_r))
    new_map = mcolors.LinearSegmentedColormap.from_list(new_name, colors)
    return new_map

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
        self.inference_timestep = 20
        self.pred_type = pred_type
        self.dimension = 2

        self.set_scheduler(timesteps=timesteps, beta_schedule=beta_schedule,
                           beta_start=beta_start, beta_end=beta_end)        
        
        self.loss_method = self.mse_loss
        self.validation_loss_method = nn.MSELoss()

        self._has_target_in_train_output = True

    def _modify_saved_settings(self, settings):
        settings = super()._modify_saved_settings(settings)
        settings["ddpm_timesteps"] = self.timesteps
        settings["ddpm_pred_type"] = self.pred_type
        settings["ddpm_beta_schedule"] = self.beta_schedule
        settings["ddpm_beta_start"] = self.beta_start
        settings["ddpm_beta_end"] = self.beta_end
        settings["inference_timestep"] = self.inference_timestep
        return settings
    
    def _modify_loaded_settings(self, settings):
        super()._modify_loaded_settings(settings)

        self.timesteps = settings["ddpm_timesteps"] if "ddpm_timesteps" in settings else 1000
        self.pred_type = settings["ddpm_pred_type"] if "ddpm_pred_type" in settings else "v"
        self.beta_schedule = settings["ddpm_beta_schedule"] if "ddpm_beta_schedule" in settings else "linear"
        self.beta_start = settings["ddpm_beta_start"] if "ddpm_beta_start" in settings else 1e-4
        self.beta_end = settings["ddpm_beta_end"] if "ddpm_beta_end" in settings else 0.02
        self.inference_timestep = settings["inference_timestep"] if "inference_timestep" in settings else 20

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

        self.save_intermediaries_steps:bool = False
        self.intermediaries_steps:Optional[List[np.ndarray]] = None
        """Save the noisy steps"""

        return self.betas


    @staticmethod
    def mse_loss(output, target):
        return F.mse_loss(output, target)

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None):
        """
        Dimse_lossffuse the data (sample from q(x_t | x_0))
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
        if self.dimension == 2:
            sqrt_ac = self.sqrt_alphas_cumprod.to(device)[t].view(-1, 1, 1, 1)
            sqrt_omac = self.sqrt_one_minus_alphas_cumprod.to(device)[t].view(-1, 1, 1, 1)
        elif self.dimension == 1:
            sqrt_ac = self.sqrt_alphas_cumprod.to(device)[t].view(-1, 1, 1)
            sqrt_omac = self.sqrt_one_minus_alphas_cumprod.to(device)[t].view(-1, 1, 1)

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
            if self.dimension == 2:
                sqrt_alpha_bar = self.sqrt_alphas_cumprod.to(self.device)[t].view(B, 1, 1, 1)
                sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod.to(self.device)[t].view(B, 1, 1, 1)
            elif self.dimension == 1:
                sqrt_alpha_bar = self.sqrt_alphas_cumprod.to(self.device)[t].view(B, 1, 1)
                sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod.to(self.device)[t].view(B, 1, 1)
           

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

        B, *C = input.shape
            
        eta = .5
        seq = range(0, self.timesteps, self.inference_timestep)

        if seq is None:
            seq = list(range(self.timesteps))

        seq_next = [-1] + list(seq[:-1])

        if self.save_intermediaries_steps:
            self.intermediaries_steps = []

        with torch.no_grad():
            x_t = torch.randn((B, *C), device=self.device)

            for i, j in zip(reversed(seq), reversed(seq_next)):
                t = torch.full((B,), i, device=self.device, dtype=torch.long)
                next_t = torch.full((B,), j, device=self.device, dtype=torch.long)

                if self.dimension == 2:
                    at = self.alphas_cumprod[t].view(B, 1, 1, 1)
                elif self.dimension == 1:
                    at = self.alphas_cumprod[t].view(B, 1, 1)


                if j == -1:
                    at_next = torch.ones_like(at)
                else:
                    if self.dimension == 2:
                        at_next = self.alphas_cumprod[next_t].view(B, 1, 1, 1)
                    elif self.dimension == 1:
                        at_next = self.alphas_cumprod[next_t].view(B, 1, 1)


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

                if self.save_intermediaries_steps:
                    self.intermediaries_steps.append(x_t.clone().squeeze(1).cpu().detach().numpy())

            x_t = x_t.clamp(-1., 1.)

            return x_t
        
    def plot_intermediaries_steps(self, cmap:Optional[str]=None):

        if self.intermediaries_steps is None:
            self.save_intermediaries_steps = True
            self.get_prediction_batch(force_compute=True)
            if self.intermediaries_steps is None or len(self.intermediaries_steps) <= 0:
                LOGGER.error("Can't plot intermediaries steps because nothing was saved in inference time.")
                return
            
        steps = self.intermediaries_steps
        map_per_row = 8
        nrows, ncols = int(np.ceil(len(steps)/map_per_row)), map_per_row
        fig, axes = plt.subplots(nrows, ncols)

        seq = range(0, self.timesteps, 5)
        for r in range(nrows):
            for c in range(ncols):
                ax:matplotlib.axes.Axes = None
                try:
                    ax = axes[r][c]
                except:
                    break
                index=r*ncols+c
                ax.imshow(steps[index][0], label=str(seq[index]), cmap="jet" if cmap is None else cmap)
                ax.legend()
        
        return fig, axes 
    
    def plot_degeneracy(self, sample_number:int, plot_boxes:bool=True, cmap="viridis", bins:Optional[int]=None):
        
        if self.intermediaries_steps is None:
            self.save_intermediaries_steps = True
            printProgressBar(0, sample_number, prefix="Predicting")
            self.get_prediction_batch(force_compute=True)
            if self.intermediaries_steps is None or len(self.intermediaries_steps) <= 0:
                LOGGER.error("Can't plot intermediaries steps because nothing was saved in inference time.")
                return
        pred_batch = self.get_prediction_batch(force_compute=False)
    
        sample_number = int(sample_number)
        sample_number = max(sample_number, 1)

        fig, axes = plt.subplot_mosaic(
            [
                ["Value1","Value2","Value3","PDF_high","PDF_high"],
                ["Std1","Std2","Std3","PDF_low","PDF_low"]
            ],
            figsize=(10,4)
        )

        steps = np.array(self.intermediaries_steps)
        squared_steps = steps**2

        final_map = pred_batch[-1][0]
        high_threshold = np.nanpercentile(final_map, 99.9)
        medium_threshold = np.nanpercentile(final_map, 50)
        low_threshold = np.nanpercentile(final_map, 40)
        high_candidates = np.argwhere(final_map >= high_threshold)
        low_candidates = np.argwhere((final_map <= medium_threshold) & (final_map >= low_threshold))
        high_yx = high_candidates[len(high_candidates) // 2]
        low_yx = low_candidates[len(low_candidates) // 2]
        high_y, high_x = high_yx
        low_y, low_x = low_yx

        #keep values of a high density pixel and low density pixel to plots distributions
        values_high_px = []
        values_low_px = []
        values_high_px.append(steps[:, 0, high_y, high_x].copy())
        values_low_px.append(steps[:, 0, low_y, low_x].copy())
        for i in range(sample_number-1):
            printProgressBar(i+1, sample_number, prefix="Predicting")
            self.get_prediction_batch(force_compute=True)
            i_steps = np.array(self.intermediaries_steps)
            steps += i_steps
            squared_steps += i_steps**2
            values_high_px.append(i_steps[:, 0, high_y, high_x].copy())
            values_low_px.append(i_steps[:, 0, low_y, low_x].copy())
        values_high_px = np.array(values_high_px)
        values_low_px = np.array(values_low_px)

        squared_steps /= sample_number
        steps /= sample_number
        std_steps = squared_steps-steps**2
        del squared_steps

        ax_index_to_time_index = np.logspace(0, np.log10(len(steps)-1), 3)
        ax_index_to_time_index = (ax_index_to_time_index.max() - ax_index_to_time_index)
        ax_index_to_time_index = ax_index_to_time_index.astype(int)[::-1]
        ax_index_to_time_index[-1] = len(steps)-1
        ax_index_to_time_index[0] = 1

        axes_mean:List[matplotlib.axes.Axes] = []
        axes_std:List[matplotlib.axes.Axes] = []
        for i in range(3):
            time_index = int(np.floor(ax_index_to_time_index[i]))
            
            ax_mean = axes["Value"+str(i+1)]
            axes_mean.append(ax_mean)
            ax_mean.imshow(steps[time_index][0], label=time_index, norm=Normalize(vmin=-0.5, vmax=0.5), cmap=cmap)

            ax_mean.scatter(high_x,high_y,s=50,marker="o",edgecolor="white",facecolor="red",linewidth=1.5,label="High density")
            ax_mean.scatter(low_x,low_y,s=50,marker="s",edgecolor="white",facecolor="cyan",linewidth=1.5,label="Low density")
            ax_mean.set_xlabel(f"$t={time_index/(len(steps)-1):.2f}$")
            ax_mean.legend()

            ax_std = axes["Std"+str(i+1)]
            axes_std.append(ax_std)
            ax_std.imshow(std_steps[time_index][0], label=time_index, cmap=cmap)
            ax_std.set_xlabel(f"$t={time_index/(len(steps)-1):.2f}$")
        axes_mean[0].set_ylabel(r"$<n_H>_m$")
        axes_std[0].set_ylabel(r"$\sigma_{n}$")

        ax_pdf_high = axes["PDF_high"]
        ax_pdf_low = axes["PDF_low"]
        selected_times = [int(i) for i in range(int(len(steps)/1.25),len(steps))]#[int(ax_index_to_time_index[0]),int(ax_index_to_time_index[1]),int(ax_index_to_time_index[2])]
        colors = plt.cm.viridis(np.linspace(0, 1, len(selected_times)))

        for i, (c, t) in enumerate(zip(colors, selected_times)):
            if i == len(selected_times)-1:
                c = "black"
            n, bins, _ = ax_pdf_high.hist(self.norms['vdens'][1](values_high_px[:, t]),bins="auto" if bins is None else bins,density=True,histtype="step",linewidth=2,color=c,label=f"t={t/(len(steps)-1):.2f}",)
            bins = 0.5 * (bins[:-1] + bins[1:])
            ax_pdf_high.axvline(np.sum(n*bins)/np.sum(n), color="red")
            n, bins, _ = ax_pdf_low.hist(self.norms['vdens'][1](values_low_px[:, t]),bins="auto" if bins is None else bins,density=True,histtype="step",linewidth=2,color=c,label=f"t={t/(len(steps)-1):.2f}",)
            bins = 0.5 * (bins[:-1] + bins[1:])
            ax_pdf_high.axvline(np.sum(n*bins)/np.sum(n), color="cyan")

        ax_pdf_low.set_xscale('log')
        ax_pdf_high.set_xscale('log')
        ax_pdf_low.grid(alpha=0.3)
        ax_pdf_high.grid(alpha=0.3)
        ax_pdf_high.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        ax_pdf_low.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

        ax_pdf_high.set_xlabel(r"$<n_H>_m$")
        ax_pdf_low.set_xlabel(r"$<n_H>_m$")

        #ax_pdf_high.legend()
        #ax_pdf_low.legend()

        if plot_boxes:
            plot_rect_bg(fig=fig, axes=[ax_pdf_high, ax_pdf_low], color="tab:green", pad=0.001, text="Pixel distributions", text_pos="top", show_bbox=True)
            plot_rect_bg(fig=fig, axes=axes_mean, color="tab:orange", pad=0.001, text="Samples mean", text_pos="top", show_bbox=True)
            plot_rect_bg(fig=fig, axes=axes_std, color="tab:red", pad=0.001, text="Samples deviations", text_pos="bottom", show_bbox=True)


    def plot_pdf_trajectory(self, traj_number:int=20, same_pdf_lims:bool=True, plot_boxes:bool=True, cmap="viridis"):

        if self.intermediaries_steps is None:
            self.save_intermediaries_steps = True
            self.get_prediction_batch(force_compute=True)
            if self.intermediaries_steps is None or len(self.intermediaries_steps) <= 0:
                LOGGER.error("Can't plot intermediaries steps because nothing was saved in inference time.")
                return
            
        steps = self.intermediaries_steps
        #steps = self.norms['vdens'][1](np.array(self.intermediaries_steps))

        fig, axes = plt.subplot_mosaic(
        [
            ["PDF_Beg", "Traj", "Traj", "PDF_End", "PDF_True"],
            ["PDF_Beg", "Traj", "Traj", "PDF_End", "PDF_True"],
            ["Time0", "Time1", "Time2", "Time3", "True"],
        ],
        figsize=(10, 6)
        )
        axes["Time0"].set_ylabel(r"$<n_H>_m$ maps")

        #Timesteps map
        ax_index_to_time_index = np.logspace(0, np.log10(len(steps)-1), 4)
        ax_index_to_time_index = (ax_index_to_time_index.max() - ax_index_to_time_index)
        ax_index_to_time_index = ax_index_to_time_index.astype(int)[::-1]
        ax_index_to_time_index[-1] = len(steps)-1
        ax_index_to_time_index[0] = 1
        axes_time_keys = []
        for key in axes.keys():
            if not("Time" in key):
                continue
            index = int(str(key).replace("Time",""))
            time_index = int(np.floor(ax_index_to_time_index[index]))
            map = steps[time_index]
            axes_time_keys.append(key)
            axes[key].imshow(map[0], label=time_index, norm=Normalize(vmin=-0.5, vmax=0.5), cmap=cmap)
            axes[key].set_xlabel(f"$t={time_index/(len(steps)-1):.2f}$")
        #Truth map
        target_tensor, _ = self.norms["vdens"][0](self.get_prediction_batch()[-1])
        axes["True"].imshow(target_tensor, label="Truth", norm=Normalize(vmin=-0.5, vmax=0.5), cmap=cmap)
        axes["True"].set_xlabel("Truth / Simulation")
        #plot_map(target_tensor, ax=axes["True"], cmap=cmap, norm=Normalize(vmin=-0.5, vmax=0.5), toplabel="Truth")

        #Trajectory
        def _plot_density_evolution(ax, steps, bins=100):
            all_values = np.concatenate([s[0].ravel() for s in steps])
            vmin = all_values.min()
            vmax = all_values.max()
            density_map = []
            for s in steps:
                values = s[0].ravel()
                hist, edges = np.histogram(values,bins=bins,range=(vmin, vmax),density=True)
                density_map.append(hist)
            density_map = np.array(density_map).T
            extent = [0, len(steps)-1, vmin, vmax]
            im = ax.imshow(density_map,aspect='auto',origin='lower',extent=extent,cmap='viridis')
            return im
        def _plot_std_evolution(ax, steps, bins=100, min_count=50):
            all_values = np.concatenate([s[0].ravel() for s in steps])
            vmin, vmax = all_values.min(), all_values.max()

            bin_edges = np.linspace(vmin, vmax, bins + 1)

            sums = np.zeros((len(steps), bins))
            squared_sums = np.zeros((len(steps), bins))
            counts = np.zeros((len(steps), bins))

            for t, s in enumerate(steps):
                current = s[0].ravel()
                final = steps[-1][0].ravel()

                bin_idx = np.digitize(current, bin_edges) - 1

                for x, f, b in zip(current, final, bin_idx):
                    if 0 <= b < bins:
                        sums[t, b] += f
                        squared_sums[t, b] += f**2
                        counts[t, b] += 1
            
            valid = counts >= min_count
            sums[~valid] = 0
            squared_sums[~valid] = 0
            counts[~valid] = 0

            mean_map = np.divide(sums, counts,out=np.zeros_like(sums),where=counts > 0)
            squared_mean_map = np.divide(squared_sums, counts,out=np.zeros_like(sums),where=counts > 0)

            map = squared_mean_map-np.power(mean_map, 2)


            extent = [0, len(steps) - 1, vmin, vmax]
            return ax.imshow(map.T,aspect='auto',origin='lower',extent=extent,cmap='viridis', label="Binned std")
        _plot_std_evolution(axes["Traj"], steps)
        #_plot_density_evolution(axes["Traj"], steps)

        #cmap = plt.get_cmap("viridis")
        #norm = plt.Normalize(vmin=target_tensor.min(), vmax=target_tensor.max())
        random_pixels = np.random.randint(low=0,high=len(steps[0][0])-1,size=(traj_number, 2))
        for pixel_id, (px_x, px_y) in enumerate(random_pixels):
            value = target_tensor[px_y, px_x]
            #color = cmap(norm(value))
            axes["Traj"].plot(np.array([t[0,px_y, px_x] for t in steps]), color="white", label="Pixel trajectories" if pixel_id == 0 else None)        
        #axes["Traj"].set_title("PDFs trajectory")
        #axes["Traj"].tick_params(labeltop=True, labelbottom=False)
        #axes["Traj"].set_xlabel("Timesteps")
        axes["Traj"].legend()

        #PDFs
        def _plot_pdf(ax:matplotlib.axes.Axes, values, label=None, lims=None):

            pdf, edges =compute_pdf(values, func=lambda x:x)
            centers = 0.5 * (edges[:-1] + edges[1:])
            ax.plot(pdf, centers, lw=2, label=label)
            ax.fill_between(pdf, centers, alpha=0.3)
            ax.grid(alpha=0.3)
            if lims is not None:
                ax.set_ylim([lim_min, lim_max])
            if label is not None:
                ax.legend()
            return edges.min(), edges.max()
        
        lim_min, lim_max = _plot_pdf(axes["PDF_Beg"], steps[0][0], label=r"Model PDF at $t_{begin}$")
        _plot_pdf(axes["PDF_End"], steps[-1][0], lims=(lim_min, lim_max) if same_pdf_lims else None, label=r"Model PDF at $t_{end}$")
        _plot_pdf(axes["PDF_True"], target_tensor, lims=(lim_min, lim_max) if same_pdf_lims else None, label=r"True PDF")
        axes["PDF_Beg"].set_ylabel(r"Normalized $<n_H>_m$")
        #axes["PDF_Beg"].set_title(r"PDF at $t_{begin}$")
        #axes["PDF_End"].set_title(r"PDF at $t_{end}$")

        if plot_boxes:
            plot_rect_bg(fig=fig, axes=[axes["PDF_Beg"],axes["Traj"],axes["PDF_End"]], color="tab:orange", pad=0.001, text="Network pdf trajectories", show_bbox=True)
            plot_rect_bg(fig=fig, axes=[axes["PDF_True"], axes["True"]], color="tab:green", pad=0.001, text="Simulation / Truth", show_bbox=True)
            plot_rect_bg(fig=fig, axes=[axes[a] for a in axes_time_keys], color="tab:red", pad=0.001, text="Diffusion process", text_pos="bottom", show_bbox=True)

            
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


    #trainer = DDPTrainer(DDPMUnet, ds1, ds2, model_name="DDPM3", timesteps=500, beta_schedule='cosine')
    trainer = load_trainer("DDPM3", trainer_class=DDPTrainer)
    trainer.pred_type = "v"
    trainer.norms = { 
        "cdens": DATA_NORMALIZATION_CDENS,
        "vdens": DATA_NORMALIZATION_VDENS,
    }

    #trainer.get_validation_error()

    trainer.ema = True
    trainer.validation_loss_method = classic_log_mse
    trainer.ema_warmup = 30
    trainer.learning_rate = 1e-4
    trainer.training_set = ds1
    trainer.validation_set = ds2
    trainer.network_settings["base_filters"] = 32
    trainer.network_settings["num_layers"] = 3
    trainer.network_settings["attention_layers"] = [2]
    trainer.network_settings["attention_heads"] = [8]
    #trainer.network_settings["filter_function"] = "linear"
    trainer.training_random_transform = True
    trainer.optimizer_name = "Adam"
    trainer.target_names = ["vdens"]
    trainer.input_names = ["cdens"]
    trainer.auto_save = 5000
    trainer.scheduler = None
    #trainer.init()
    trainer.train(10000,batch_number=8,compute_validation=100,early_stopping=False)
    trainer.save()
    trainer.get_validation_error()

    trainer.plot(save=True)
    trainer.plot_validation(save=True, number=8, number_per_row=4)

    plt.show()