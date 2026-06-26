from POLARIScore.networks.Trainer import Trainer
from POLARIScore.utils.utils import printProgressBar, plot_rect_bg
import numpy as np
import matplotlib.pyplot as plt
from typing import *
from POLARIScore.config import *
import matplotlib.axes
from matplotlib.colors import Normalize, LogNorm
from scipy.stats import lognorm
import time, copy

class ProbabilisticTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(ProbabilisticTrainer, self).__init__(*args, **kwargs)

    def get_prediction_batch(self,force_compute=False,batch_number:int=1,sample_number=1, remove_residuals_baseline:bool=False):
        if not(self.prediction_batch is None or force_compute):
            return self.prediction_batch
        
        start_time = time.time()
        pred_batch = self.predict(self.validation_set,batch_number=batch_number)

        self.prediction_batch = pred_batch
        if sample_number > 1:
            samples = np.array([s[1] for s in pred_batch])
            samples = self.norms[self.target_names[0]][0](samples)
            for i in range(sample_number-1):
                i_batch = self.predict(self.validation_set,batch_number=batch_number)
                i_step = np.array([s[1] for s in i_batch])
                i_step = self.norms[self.target_names[0]][0](i_step)
                samples += i_step
            samples /= sample_number


            for i in range(len(pred_batch)):
                pred_batch[i] = (pred_batch[i][0], self.norms[self.target_names[0]][1](samples[i]))
            self.prediction_batch = pred_batch

        end_time = time.time()
        self.inference_time = (end_time - start_time)/len(self.prediction_batch[0])

        if remove_residuals_baseline:
            pred_batch = copy.deepcopy(self.get_prediction_batch())
            for j,c in enumerate(pred_batch):
                pred_batch[j] = (c[0], self.apply_baseline(c[1], log=False))
            return pred_batch

        return self.prediction_batch
    
    def plot_sampling(self, sample_number:int, plot_boxes:bool=True, cmap="viridis", bins:Optional[int]=None):
        printProgressBar(0, sample_number, prefix="Predicting")
        pred_batch = self.get_prediction_batch(force_compute=True, sample_number=1)
    
        sample_number = int(sample_number)
        sample_number = max(sample_number, 1)

        fig, axes = plt.subplot_mosaic(
            [
                ["Truth","Pred","PDF_high","PDF_high"],
                ["Input","Std_Pred","PDF_low","PDF_low"]
            ],
            figsize=(8,4)
        )

        steps = self.norms[self.target_names[0]][0](pred_batch[-1][1])
        squared_steps = steps**2

        final_map = self.norms[self.target_names[0]][0](pred_batch[-1][0])
        high_threshold = np.nanpercentile(final_map, 99.9)
        medium_threshold = np.nanpercentile(final_map, 50)
        low_threshold = np.nanpercentile(final_map, 40)
        high_candidates = np.argwhere(final_map >= high_threshold)
        low_candidates = np.argwhere((final_map <= medium_threshold) & (final_map >= low_threshold))
        high_yx = high_candidates[len(high_candidates) // 2]
        low_yx = low_candidates[len(low_candidates) // 2]
        high_y, high_x = high_yx
        low_y, low_x = low_yx

        values_high_px = []
        values_low_px = []
        values_high_px.append(steps[high_y, high_x].copy())
        values_low_px.append(steps[low_y, low_x].copy())
        for i in range(sample_number-1):
            printProgressBar(i+1, sample_number, prefix="Predicting")
            i_steps = self.get_prediction_batch(force_compute=True, sample_number=1)[-1][1]
            i_steps = self.norms[self.target_names[0]][0](i_steps)
            steps += i_steps
            squared_steps += i_steps**2
            values_high_px.append(i_steps[high_y, high_x].copy())
            values_low_px.append(i_steps[low_y, low_x].copy())
        values_high_px = np.array(values_high_px)
        values_low_px = np.array(values_low_px)

        axes["Truth"].imshow(final_map, cmap=cmap, norm=Normalize(vmin=-0.5, vmax=0.5))
        axes["Truth"].set_ylabel(r"normalized $<n_H>_m$")
        axes["Truth"].set_xlabel("True density")
        axes["Input"].imshow(self.norms[self.input_names[0]][0](self.validation_set.get(-1)[self.validation_set.get_element_index(self.input_names[0])]), cmap=cmap, norm=Normalize(vmin=-0.5, vmax=0.5))
        axes["Input"].set_ylabel(r"normalized $N_H$")
        axes["Input"].set_xlabel("Input / Column density")

        squared_steps /= sample_number
        steps /= sample_number
        std_steps = squared_steps-steps**2
        del squared_steps
            
        ax_pred = axes["Pred"]
        ax_pred.imshow(steps, cmap=cmap, norm=Normalize(vmin=-0.5, vmax=0.5))
        ax_pred.scatter(high_x,high_y,s=50,marker="o",edgecolor="white",facecolor="red",linewidth=1.5,label="High density")
        ax_pred.scatter(low_x,low_y,s=50,marker="s",edgecolor="white",facecolor="cyan",linewidth=1.5,label="Low density")
        ax_pred.set_xlabel(f"Density prediction by model")
        ax_pred.legend()

        ax_std = axes["Std_Pred"]
        ax_std.imshow(std_steps, cmap=cmap)
        ax_std.set_xlabel(f"Deviation of model samples")
        #ax_pred.set_ylabel(r"$<n_H>_m$")
        ax_std.set_ylabel(r"$\sigma_{n}$")

        ax_pdf_high = axes["PDF_high"]
        ax_pdf_low = axes["PDF_low"]

        def _plot_pdf(values, ax, label, pos):
            n, bin_edges = np.histogram(values, bins="auto" if bins is None else bins, density=False)
            n_err = np.sqrt(n)
            hist_bins = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            mean_high = np.sum(n*hist_bins)/np.sum(n)
            hist_bins = self.norms[self.target_names[0]][1](hist_bins)
            values_fit = self.norms[self.target_names[0]][1](values)
            values_fit = values_fit[values_fit > 0]
            
            shape, loc, scale = lognorm.fit(values_fit, floc=0)
            sigma = shape
            mu = np.log(scale)
            delta = np.sqrt(2 * sigma**2 * np.log(2))
            x_left_fit  = np.exp(mu - sigma**2 - delta)
            x_right_fit = np.exp(mu - sigma**2 + delta)
            fwhm_fit = np.log10(x_right_fit) - np.log10(x_left_fit)

            bin_edges_fit = self.norms[self.target_names[0]][1](bin_edges)
            mask = bin_edges_fit > 0
            bin_edges_fit = bin_edges_fit[mask]
            cdf_vals = lognorm.cdf(bin_edges_fit, shape, loc=loc, scale=scale)
            expected = np.diff(cdf_vals) * np.sum(n)

            #ax_pdf_high.plot(hist_bins,n,drawstyle="steps-mid",linewidth=2,color="black",label=r"High density pixel pdf")
            ax.errorbar(hist_bins,n,yerr=n_err,fmt='-',drawstyle="steps-mid",linewidth=2,color="black",capsize=2,label=label)
            ax.step(bin_edges_fit[:-1],expected,where="post",color="red",linewidth=2,label=r"Lognormal fit: $\mathrm{FWHM}_{\mathrm{log}}$="+rf"${fwhm_fit:.2f}$")

            #ax.axvline(self.norms[self.target_names[0]][1](final_map[pos[0], pos[1]]), color="blue", linewidth=1.5)

            ax.set_xscale('log')
            ax.grid(alpha=0.3)
            ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
            ax.set_xlabel(r"$<n_H>_m$")
            ax.set_xlim([self.norms[self.target_names[0]][1](steps.min()), self.norms[self.target_names[0]][1](steps.max())])
            ax.legend()

            return mean_high
        
        mean_high = _plot_pdf(values_high_px, ax_pdf_high, label=r"High density pixel pdf", pos=high_yx)
        mean_low = _plot_pdf(values_low_px, ax_pdf_low, label=r"Low density pixel pdf", pos=low_yx)
        
        ax_pdf_high.axvline(self.norms[self.target_names[0]][1](mean_high), color="red", linewidth=1.5)
        ax_pdf_high.axvline(self.norms[self.target_names[0]][1](mean_low), color="cyan", linewidth=1.5)
        ax_pdf_low.axvline(self.norms[self.target_names[0]][1](mean_high), color="red", linewidth=1.5)
        ax_pdf_low.axvline(self.norms[self.target_names[0]][1](mean_low), color="cyan", linewidth=1.5)

        if plot_boxes:
            plot_rect_bg(fig=fig, axes=[ax_pdf_high, ax_pdf_low], color="tab:green", pad=0.001, text="Pixel distributions", text_pos="top", show_bbox=True)
            plot_rect_bg(fig=fig, axes=[axes["Pred"], axes["Std_Pred"]], color="tab:orange", pad=0.001, text="From Model", text_pos="top", show_bbox=True)
            plot_rect_bg(fig=fig, axes=[axes["Input"], axes["Truth"]], color="tab:blue", pad=0.001, text="Truth / Simulation", text_pos="top", show_bbox=True)
