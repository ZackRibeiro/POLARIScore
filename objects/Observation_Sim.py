from POLARIScore.config import LOGGER
from POLARIScore.objects.Simulation_DC import Simulation_DC
from POLARIScore.objects.Observation import Observation
from POLARIScore.objects.DenseCore import DenseCore
from POLARIScore.utils.utils import *
from POLARIScore.utils.physics_utils import dcmf_func
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm

from typing import *
from scipy.optimize import curve_fit

import numpy as np

class Observation_Sim(Observation):
    def __init__(self,simulation:Simulation_DC, axis=0):
        super().__init__(simulation.name, None, 0., 'pc_map')
        LOGGER.log(f"Creating observation based on simulation {simulation.name}")
        LOGGER.warn("Certain methods of the original Observation class are not (yet) supported by this subclass.")

        self.axis = axis
        self.simulation = simulation
        self.folder = simulation.folder
        self.data = compute_column_density(simulation.data['RHO'], simulation.cell_size, axis=axis)
        self.get_cores()

    def skycoord_to_pixel(self,coords, _=None):
        x_pc, y_pc = coords
        x_idx = convert_pc_to_index(x_pc, len(self.data), self.simulation.size, start=self.simulation.bbox[0][0], clip=False)
        y_idx = convert_pc_to_index(y_pc, len(self.data), self.simulation.size, start=self.simulation.bbox[0][0], clip=False)
        return (x_idx, y_idx)

    def get_cores(self):
        try:
            self.cores, coords = self.simulation.get_cores(axis=self.axis, flip_y=False)
            cores = []
            for i,c in enumerate(self.cores):
                new_core = DenseCore(self,{
                    **c
                })
                new_core.data['pos_x'] = coords[0][i]
                new_core.data['pos_y'] = coords[1][i]
                cores.append(new_core)
            self.cores = cores

        except:
            LOGGER.error("Can't load cores in observation simulation")
        return self.cores
    
    def pc_to_pixels(self, pc):
        result = pc*self.simulation.nres/self.simulation.size
        return result
    
    def plot_error_histogram(self,ax=None,predicted_quantity: Callable = compute_mass_weighted_density,
                             bins: int = 100, min_truth:Union[float, List[float]]=1e2, density=True):
        assert self.prediction is not None, LOGGER.error("There is no prediction to plot")

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        truth = predicted_quantity(self.simulation.data['RHO'],axis=self.axis)
        prediction = self.prediction
        r = prediction/truth

        cmap = plt.cm.viridis
        max_y = 0

        def _plot_hist(bin_min:float, bin_max:float,fit:bool=True,use_cmap:bool=True,color:Optional[str]=None, max_y=0):

            mask = (truth >= bin_min) & (truth <= bin_max)
            masked_truth = truth[mask]
            masked_r = r[mask]

            probs, edges = compute_pdf(masked_r, bins=bins, density=density)
            max_y = max(np.max(probs),max_y)
            edges = 10**edges
            centers = (edges[:-1]+edges[1:])/2

            bin_ids = np.digitize(masked_r, edges) - 1
            truth_per_bin = np.full(len(centers), np.nan)
            for i in range(len(centers)):
                mask = bin_ids == i
                if np.any(mask):
                    truth_per_bin[i] = np.median(masked_truth[mask])

            norm = LogNorm(vmin=np.nanmin(truth_per_bin),vmax=np.nanmax(truth_per_bin))

            for i in range(len(centers)):
                ax.scatter(centers[i],probs[i],color=cmap(norm(truth_per_bin[i])) if use_cmap else color,s=1,marker="o", label=f"[{bin_min:.2e},{bin_max:.2e}]" if i == 0 else None)

            points = np.array([centers, probs]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            segment_values = 0.5 * (truth_per_bin[:-1] + truth_per_bin[1:])

            lc = LineCollection(segments,cmap=cmap if use_cmap else None,color=color,norm=norm,linewidth=2)
            lc.set_array(segment_values)

            ax.add_collection(lc)

            if fit:
                lognormal = lambda x,amp,mean,sigma: dcmf_func(x,amp,mean,sigma,1,np.inf, enable_cutoff=False)
                popt, _ = curve_fit(lognormal, centers, probs,
                    p0=[np.max(probs),centers[np.argmax(probs)],np.std(centers)])
                func = lambda X: lognormal(X, *popt)
                plot_function(func, ax=ax, scatter=False, logspace=True, lims= (np.min(centers), np.max(centers)), color="red" if color is None else color, linestyle="--")

                LOGGER.log(f"Fitted lognormal sigma: {popt[-1]}")

            if use_cmap:
                sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
                cbar = fig.colorbar(sm, ax=ax)
                cbar.set_label(r"Median simulation quantity in bin")

            return max_y

        if isinstance(min_truth, (float,int)):
            max_y = _plot_hist(min_truth, np.inf, fit=True, use_cmap=True)
        elif isinstance(min_truth, (list, np.ndarray, tuple)):
            if len(min_truth) == 1:
                 max_y = _plot_hist(min_truth[0], np.inf, fit=True, use_cmap=True)
            elif len(min_truth) == 2:
                max_y = _plot_hist(min_truth[0], min_truth[1], fit=True, use_cmap=True)
            else:
                for i in range(len(min_truth)-1):
                    max_y = _plot_hist(min_truth[i], min_truth[i+1], fit=False, use_cmap=False, color=cmap(i/(len(min_truth)-1)),max_y=max_y)

        ax.axvline(1., 0., 1., color='black', ls='-')

        ax.set_xscale("log")
        if not(density):
            ax.set_ylim([0, max_y])
        ax.set_xlabel("prediction/truth")
        ax.set_ylabel("pdf")
        ax.legend()

        return fig, ax
            
            
        