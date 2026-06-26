import os
from POLARIScore.config import *
from POLARIScore.utils.utils import dictsToString, plot_map, plot_rect_bg, bin_mean
from POLARIScore.networks.utils.nn_utils import open_samples_as_spectrummap
from typing import *
from matplotlib.colors import LogNorm, Normalize
import matplotlib.pyplot as plt
import argparse
import numpy as np
from POLARIScore.objects.Observation import Observation, _worker_degeneracy, _crop
from POLARIScore.objects.Observation_Sim import Observation_Sim
from POLARIScore.objects.Simulation_DC import Simulation_DC, openSimulation
from scipy.ndimage import zoom

OBS_NAME = "OrionB"
MODEL_NAME = "DDPM"
SAMPLES_NAME = "pdf_orionb_ddpm_16"
PREDICTION_NAME = "ddpm_16"
METRICS = ["variance", "entropy"]
from astropy.coordinates import Angle
REGIONS = [
    [Angle("5h50m").deg, Angle("5h45").deg, Angle("-0d19m").deg, Angle("0d53m").deg],[Angle("5h48m").deg, Angle("5h39m").deg, Angle("-3d10m").deg, Angle("-0d58m").deg]
    ]

parser = argparse.ArgumentParser()
parser.add_argument("--obs", required=False, default=OBS_NAME, help="Observation name, can be sim name if --sim is True")
parser.add_argument("--sim", required=False, default=False, help="If True then the obs is a sim casted as obs")
parser.add_argument("--model", required=False, default=MODEL_NAME, help="Model name, in order to get the validation error")
parser.add_argument("--samples", required=False, default=SAMPLES_NAME, help="Name of the file with the samples")
parser.add_argument("--prediction", required=False, default=PREDICTION_NAME, help="Name of the prediction file")
parser.add_argument("--metrics", required=False, nargs="+", default=METRICS, help="Metrics to plot (see what is accepted in _worker_degeneracy)")
parser.add_argument("--x", required=False, default="column_density", help="Set what is used as x-axis.")

args = parser.parse_args()

if (isinstance(args.sim, bool)):
    pass
elif args.sim.lower() in ['true', 'on', 'yes', 'y']:
    args.sim = True
else:
    args.sim = False

if args.sim:
    try:
        sim = Simulation_DC(args.obs)
    except:
        sim = openSimulation("orionMHD_lowB_multi_", global_size=66.0948+0.12,keys=['RHO'],cache_name="orion")
    obs = Observation_Sim(sim)
    REGIONS = [None]
else:
    obs = Observation(args.obs, "column_density_map")
obs.load_error(args.model)
obs.load("_"+str(args.prediction))
bin_centers, q1, q2 , means = obs.network_error


if args.x.lower() in ["cdens","column_dens","column_densities","column_density"]:
    x_label = "column_density"
    x_data = obs.data
elif args.x.lower() in ["vdens", "volume_dens","predicted_dens","predicted","prediction"]:
    x_label = "predicted density"
    x_data = obs.prediction
else:
    raise NotImplementedError()

smap = open_samples_as_spectrummap(args.samples, 32, use_kernel=False)
if args.x.lower() in ["cdens","column_dens","column_densities","column_density"]:
    xlabel = "column_density"
    x_data = obs.data
elif args.x.lower() in ["vdens", "volume_dens","predicted_dens","predicted","prediction"]:
    xlabel = "predicted density"
    x_data = obs.prediction
else:
    raise NotImplementedError()
zoom_factor_y = smap.map.shape[0] / x_data.shape[0]
zoom_factor_x = smap.map.shape[1] / x_data.shape[1]
x_data = zoom(x_data,(zoom_factor_y, zoom_factor_x),order=1)

ax_names = []
ax_region_names = []
ax_metrics_names = []
for j in range(len(args.metrics)):
    ax_names_j = ["REGION"+str(i)+"_"+str(j) for i in range(len(REGIONS))]
    ax_region_names.extend([*ax_names_j])
    ax_metrics_names.append("METRIC"+str(j))
    ax_names_j.append("METRIC"+str(j))
    ax_names.append(ax_names_j)
fig, axes = plt.subplot_mosaic(ax_names,figsize=(2*len(ax_names[0]),2*len(ax_names)))

def _plot_curve(x,y,ax=None, label=None, color=None, linestyle=None, log_average=50):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    x, y, x_std, y_std = bin_mean(x.flatten(), y.flatten(), dx=None,min_per_bin=2, nbins=log_average, return_deviation=True, method="median")
    line, = ax.plot(x,y, marker="+", label=label, color=color, linestyle=linestyle if linestyle is not None else "-")
    ax.fill_between(x,np.clip(y-y_std, 0., np.max(y)),y+y_std, color=line.get_color(), alpha=0.2)
    return fig, ax

cores = obs.get_cores()

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
axes_metrics = []
for j,m in enumerate(args.metrics):

    axes_metric = []

    degeneracy = np.sum(smap.compute(remove_lambda_functions=True, method=_worker_degeneracy, save=True, stride=1,
                              extra_args={"error_x":bin_centers, "error_q": (q1, q2),'method': m})
                              , axis=-1).T
    
    ax_metric = axes["METRIC"+str(j)]
    ax_metric.set_xlim(np.percentile(x_data[x_data > 0], 1),x_data.max())
    _plot_curve(x_data, degeneracy, ax=ax_metric, label="Map", color="black")
    obs.plot_cores_data(obs.data, degeneracy,ax=ax_metric, show_deviation=True, label="Cores", color="tab:orange")
    ax_metric.set_xscale("log")
    ax_metric.set_ylabel(m)
    ax_metric.set_xlabel(x_label)
    ax_metric.grid()
    axes_metric.append(ax_metric)
    

    for i,r in enumerate(REGIONS):

        ax_region = axes["REGION"+str(i)+"_"+str(j)]
        axes_metric.append(ax_region)

        if r is not None:
            x_min, x_max, y_min, y_max = _crop(obs.wcs, r)
            x_min, x_max = int(x_min*zoom_factor_x), int(x_max*zoom_factor_x)
            y_min, y_max = int(y_min*zoom_factor_y), int(y_max*zoom_factor_y)
            region_x_data = x_data[y_min:y_max, x_min:x_max]
            region_degeneracy = degeneracy[y_min:y_max, x_min:x_max]
        else:
            region_degeneracy = degeneracy
            region_x_data = x_data

        im = plot_map(region_degeneracy, ax=ax_region, 
                      contour=np.log10(np.clip(region_x_data, np.nanpercentile(region_x_data, 5), np.nanpercentile(region_x_data, 95))),
                        cmap="viridis", contour_sigma=3,
            contour_levels=3, norm=Normalize(vmin=np.nanpercentile(region_degeneracy, 2),vmax=np.nanpercentile(region_degeneracy, 98)))

    axes_metrics.append(axes_metric)

fig.tight_layout()
for j,m in enumerate(args.metrics):
    text_label = m.upper() + ": "
    print(m.lower())
    if "variance" in m.lower():
        print("test")
        text_label += r"$\sum\left((p_{\mathrm{\langle n\rangle,pred}}(\log_{10}\langle n\rangle-\sum p_{\mathrm{\langle n\rangle,pred}}\times\langle n\rangle))/\sigma_{\mathrm{nn,val}}\right)^2$"
    elif "entropy" in m.lower():
        text_label += r"$H_{x,y}$ = exp($-\sum p_{\mathrm{<n>,pred}}(x,y)\mathrm{log}(p_{\mathrm{<n>,pred}}(x,y))$)"

    plot_rect_bg(fig, axes_metrics[j], colors[j], text=text_label, show_bbox=True, text_offset=0.04)

plt.show()
