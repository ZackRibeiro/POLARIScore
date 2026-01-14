import os
from POLARIScore.config import EXPORT_FOLDER, LOGGER, DATA_NORMALIZATION_CDENS, DATA_NORMALIZATION_VDENS
from POLARIScore.utils.utils import dictsToString
import uuid
from typing import Tuple, List
from matplotlib.colors import LogNorm, CenteredNorm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import argparse
import numpy as np
import time

"""
Usage:
python -m POLARIScore.scripts.makeBenchmark --models UNet cINN DDPM --obs_name OrionB --obs_suffixes unet cinn ddpm --obs_repair no yes yes --name Benchmark_OrionB
"""

#If you want to add zoom in regions
from astropy.coordinates import Angle
REGIONS = [
[Angle("5h50m").deg, Angle("5h45").deg, Angle("-0d19m").deg, Angle("0d53m").deg],[Angle("5h48m").deg, Angle("5h39m").deg, Angle("-3d10m").deg, Angle("-0d58m").deg]
]

MONTE_CARLO = 20

start_time = time.process_time()
def _format_time(seconds:float)->str:
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

parser = argparse.ArgumentParser()
parser.add_argument("--models", required=True, nargs="+", help="List of model names")
parser.add_argument("--trainers", required=False, nargs="+", default=None, help="List of which trainers to use in the same order of the models, can be 'ddpm', 'inn', any other string leads to default trainer.")

parser.add_argument("--obs_name", required=False, default=None, help="Molecular cloud name / Folder name of the observation")
parser.add_argument("--obs_catalog", required=False, default="Könyves et al, 2020")
parser.add_argument("--obs_dist", required=False, default=400., help="Distance to the molecular cloud")
parser.add_argument("--obs_suffixes", required=False, nargs="+", default=None, help="Suffixes of the .npy predictions (one for each model given)")
parser.add_argument("--obs_repair", required=False, nargs="+", default=None)

parser.add_argument("--extra_suffixes", required=False, nargs="+", default=None, help="If make benchmark of other observation suffixes not linked to neural networks")

parser.add_argument("--format", required=False, default="jpg", help="Image format (default: jpg)")
parser.add_argument("--density_correction", required=False, default="yes", help="Apply two medius approx to go from the mass-weighted density to core volume density.")

parser.add_argument("--name", required=False, default=str(uuid.uuid1()), help="Name of the benchmark (default: uuid1)")
parser.add_argument("--output", required=False, default=EXPORT_FOLDER, help="Benchmark will be generate in this folder (default POLARIScore export folder).")
args = parser.parse_args()

if(str.lower(args.density_correction) in ["false","no","n"]):
    args.density_correction = False
else:
    args.density_correction = True

if args.trainers is None:
    auto_list = []
    for m in args.models:
        if "ddpm" in str.lower(m):
            auto_list.append("ddpm")
        elif "inn" in str.lower(m):
            auto_list.append("inn")
        else:
            auto_list.append("default")
    trainers = auto_list
trainers = [str.lower(t) for t in trainers]

from POLARIScore.objects.Observation import Observation
observation:'Observation' = None
if args.obs_name is not None:
    observation = Observation(args.obs_name,"column_density_map")
    observation.catalog_name = args.obs_catalog
    observation.distance = float(args.obs_dist)
    assert args.obs_suffixes is not None, LOGGER.error("If an observation is given then you must also give corresponding prediction suffixes.")
    assert len(args.obs_suffixes) == len(trainers), LOGGER.error("Given suffixes must have the same length than given trainers/models.")
    assert args.obs_repair is None or len(args.obs_repair) == len(trainers), LOGGER.errror("Given repair arguments must have the same length than given trainers/models.")

BENCHMARK_PATH = os.path.join(args.output, args.name)
assert not(os.path.exists(BENCHMARK_PATH)), LOGGER.error(f"Benchmark folder already exists, delete it. ({BENCHMARK_PATH})")
os.mkdir(BENCHMARK_PATH)

F_DCMF_PATH = os.path.join(BENCHMARK_PATH, "dcmfs")
os.mkdir(F_DCMF_PATH)
F_CORE_HISTS = os.path.join(BENCHMARK_PATH, "core_hists")
os.mkdir(F_CORE_HISTS)
F_CORE_RELATIONS = os.path.join(BENCHMARK_PATH, "core_relations")
os.mkdir(F_CORE_RELATIONS)
F_DENSITY_DISTS = os.path.join(BENCHMARK_PATH, "density_dists")
os.mkdir(F_DENSITY_DISTS)
F_CLOUDS = os.path.join(BENCHMARK_PATH, "cloud_voldens")
os.mkdir(F_CLOUDS)
F_CORRELATION_PATH = os.path.join(BENCHMARK_PATH, "correlations")
os.mkdir(F_CORRELATION_PATH)

global_axes = {
    "core_residuals": None,
    "core_hists": None,
    "core_relations": None,
    "density_dists": None,
    "core_diffs": None,
}

in_files = []

accuracy_fig = plt.figure()
accuracy_ax_total = plt.subplot2grid((3, len(trainers)), (0, 0), rowspan=2, colspan=3, fig=accuracy_fig)

if len(REGIONS) > 0:
    region_axes = [[None for _ in range(1+len(args.obs_suffixes if args.obs_suffixes is not None else [])+len(args.extra_suffixes if args.extra_suffixes is not None else []))] for i in range(2)]
    region_figs = [plt.figure(figsize=(len(region_axes[0])*2.75,6),dpi=300.) for _ in REGIONS]
    for fig in region_figs:
        fig.subplots_adjust(hspace=0.05)

linestyles = ["-","--","-.",":"]
colors = ["tab:blue","tab:orange","tab:green","tab:red","tab:purple"]

predictions = []
def make_obs_benchmark(suffix,model_name=None,i=0):
        predictions.append(observation.load(suffix="_"+suffix))
        m = suffix if model_name is None else model_name
        if model_name is not None:
            observation.load_error(model_name=m)
        if model_name is not None and args.obs_repair is not None:
            if args.obs_repair[i] in [True, "yes","y"]:
                observation.prediction = observation.rectify_error_baseline()
            elif args.obs_repair[i] in ["reverse","rev","revert"]:
                observation.prediction = observation.rectify_error_baseline(revert=True)

        #Figures with all models:
        _, ax = observation.plot_cores_error(ax=global_axes["core_residuals"], 
                                             mov_average=0, log_average=50, show_errors=False, show_model_errors=False,
                                            correction=args.density_correction, color="black", linestyle=linestyles[i], label=m)
        global_axes["core_residuals"] = ax

        _, ax = observation.plot_cores_hist(ax=global_axes["core_hists"], plot_catalog=global_axes["core_hists"] is None, label=m, correction=args.density_correction)
        global_axes["core_hists"] = ax

        _, ax = observation.plot_cores_baseline(ax=global_axes["core_relations"], derived_cores=False, density_correction=args.density_correction, invert_xy=True, x_coldens=True, mov_average=1, fit=True, cmap_color=False, forced_label=m)
        global_axes["core_relations"] = ax

        _, ax = observation.plot_density_distributions(ax=global_axes["density_dists"], monte_carlo=0, offset_method="wout_ncol", draw_style=None, color=None, label=m)
        global_axes["density_dists"] = ax

        _, ax = observation.plot_cores_mass(ax=global_axes["core_diffs"], bins_mean=20, label=m, show_errors=False, linestyle=linestyles[i])
        global_axes["core_diffs"] = ax

        #One figure per model:


        fig, _ = observation.plot_correlation()
        fig.savefig(os.path.join(F_CORRELATION_PATH,"correlation_"+m+"."+args.format))
        plt.close(fig)

        fig, _ = observation.plot_dcmf(method="constant", monte_carlo=MONTE_CARLO, fit=False, bins=15, correction=args.density_correction)
        fig.savefig(os.path.join(F_DCMF_PATH,"dcmf_"+m+"."+args.format))
        plt.close(fig)

        fig, _ = observation.plot_cores_hist(label=m, correction=args.density_correction)
        fig.savefig(os.path.join(F_CORE_HISTS,"core_hist_"+m+"."+args.format))
        plt.close(fig)

        fig, _ = observation.plot_cores_baseline(derived_cores=True, density_correction=args.density_correction, invert_xy=True, x_coldens=True, mov_average=1, fit=True, forced_label=m)
        fig.savefig(os.path.join(F_CORE_RELATIONS,"core_relations_"+m+"."+args.format))
        plt.close(fig)

        fig, _ = observation.plot_density_distributions(monte_carlo=0, offset_method="max", color="red", label=m, marker="+")
        fig.savefig(os.path.join(F_DENSITY_DISTS,"density_dists_"+m+"."+args.format))
        plt.close(fig)

        fig, _ = observation.plot(data=observation.prediction, norm=LogNorm(vmin=1e2,vmax=3e5), plot_cores=False, force_vol=True)
        fig.savefig(os.path.join(F_CLOUDS,"voldens_"+m+"."+args.format))
        plt.close(fig)
        if observation.get_skeleton() is not None:
            fig, _ = observation.plot(data=observation.prediction, norm=LogNorm(vmin=1e2,vmax=3e5), plot_cores=False, force_vol=True, plot_skeleton=True)
            fig.savefig(os.path.join(F_CLOUDS,"voldens_skeleton_"+m+"."+args.format))
            plt.close(fig)
        for j,r in enumerate(REGIONS):
            if observation.get_skeleton() is not None:
                fig, _ = observation.plot(data=observation.prediction, crop=r, norm=LogNorm(vmin=1e2,vmax=3e5), plot_cores=False, force_vol=True, plot_skeleton=True)
                fig.savefig(os.path.join(F_CLOUDS,f"voldens_skeleton_r{str(j)}_"+m+"."+args.format))
                plt.close(fig)
            region_ax = plt.subplot2grid((2, len(region_axes[0])),(0, i+1), fig=region_figs[j], projection=observation.wcs)
            _, region_ax = observation.plot(data=observation.prediction, crop=r, ax=region_ax, norm=LogNorm(vmin=50,vmax=3e5), plot_cores=False, force_vol=True,
                             cbar=False, sbar=2., sbar_transparent=True,
                             show_ax_labels=False, toplabel=m
                             )
            region_axes[j][i+1] = region_ax
            
from POLARIScore.networks import INNTrainer,DDPTrainer,Trainer
for i,t,m in zip(range(len(trainers)),trainers, args.models):
    used_trainer = Trainer.Trainer 
    if t == "ddpm" or t == "ddp":
        t = "ddpm"
        used_trainer = DDPTrainer.DDPTrainer
    elif t == "inn" or t == "cinn":
        used_trainer = INNTrainer.INNTrainer
    else:
        t = "default"
        
    trainer = Trainer.load_trainer(m, trainer_class=used_trainer)
    if t != "default":
        trainer.norms = {
            "cdens": DATA_NORMALIZATION_CDENS,
            "vdens": DATA_NORMALIZATION_VDENS,
        }
    trainer.get_prediction_batch()

    #Accuracy plot
    Trainer.plot_accuracy([trainer], ax=accuracy_ax_total, linestyle=linestyles[i], color=colors[i], xlabel="")
    acc_ax = plt.subplot2grid((3, len(trainers)), (2, i), rowspan=1, colspan=1, fig=accuracy_fig)
    Trainer.plot_accuracy([trainer], ax=acc_ax, bins=[0,2.3,4,7], use_linestyles=True, color=colors[i], legend=False, xlabel="Error allowed (in log10)" if i == int(len(trainers)/2) else "", ylabel="Accuracy" if i==0 else "")

    in_files.append({
        "model_name": m,
        "inference_time(s/img)": str(trainer.inference_time),
        "inference_speed(img/s)": str(1/trainer.inference_time),
        "parameters":  sum(p.numel() for p in trainer.model.parameters()),
        "MSE": trainer.validation_losses[-1],
    })

    if observation is not None :
        make_obs_benchmark(suffix=args.obs_suffixes[i], model_name=m, i=i)
    del trainer
    
if observation is not None:
    if args.extra_suffixes is not None:
        for i,s in enumerate(args.extra_suffixes):
            make_obs_benchmark(suffix=s, i=len(trainers)+i)

    if len(REGIONS) > 0:
        for i,r in enumerate(REGIONS):
            region_ax = plt.subplot2grid((2, len(region_axes[0])),(0, 0), fig=region_figs[i], projection=observation.wcs)
            _, region_ax = observation.plot(data=observation.data, crop=r, ax=region_ax, norm=LogNorm(vmin=1e21,vmax=None), plot_cores=False, force_col=True, cbar=False, sbar=2., sbar_transparent=True,
                             show_ax_labels=False, toplabel="$N_H$"
                             )
            region_axes[i][0] = region_ax

            if(len(args.models) == 3):
                pred1 = np.log10(np.nan_to_num(predictions[0], nan=1.))
                pred2 = np.log10(np.nan_to_num(predictions[1], nan=1.))
                pred3 = np.log10(np.nan_to_num(predictions[2], nan=1.))
                rgb = np.stack((pred1/ np.max(pred1), pred2/ np.max(pred2), pred3/ np.max(pred3)), axis=-1)
                rgb = rgb**1.3
                rgb = rgb/np.max(rgb)
                rgb_ax = plt.subplot2grid((2, len(region_axes[0])),(1, 0), fig=region_figs[i], projection=observation.wcs)
                observation.plot(data=rgb,
                                crop=r, ax=rgb_ax, plot_cores=False, force_col=True, cbar=False, sbar=2., sbar_transparent=True,
                                show_ax_labels=False, toplabel="$RGB$")

            _length = len(args.models)
            sub_axes = []
            for j in range(_length):
                idx1 = j % _length
                idx2 = (j+1) % _length
                ax = plt.subplot2grid((2, len(region_axes[0])),(1, 1+j), fig=region_figs[i], projection=observation.wcs)
                observation.plot(np.clip(np.log10(np.nan_to_num(predictions[idx2], nan=1.))-np.log10(np.nan_to_num(predictions[idx1], nan=1.)),-.5,.5), ax=ax, crop=r,
                                  cmap="coolwarm", norm=CenteredNorm(), plot_cores=False, show_ax_labels=False, cbar=False, sbar=2., sbar_transparent=True, cores_color="purple",
                                 toplabel=f"log({args.models[idx2]})-log({args.models[idx1]})")
                sub_axes.append(ax)

            region_figs[i].colorbar(region_axes[i][0].images[0],ax=[region_axes[i][0]],orientation="vertical",
                location="left",fraction=0.03, pad=0.02, label=r"$N_H(cm^{-2})$"
            )

            region_figs[i].colorbar(region_axes[i][1].images[0],ax=region_axes[i][1:][-1],orientation="vertical",
                location="right",fraction=0.03,pad=0.02, label=r"$<n_H>_m(cm^{-3})$"
            )

            region_figs[i].colorbar(sub_axes[0].images[0],ax=sub_axes[-1],orientation="vertical",
                location="right",fraction=0.03,pad=0.02, label=r"clipped difference"
            )

            region_figs[i].savefig(os.path.join(BENCHMARK_PATH,f"region_{str(i)}."+args.format))
    
    fig, _ = observation.plot(norm=LogNorm(vmin=1e21), plot_cores=False, force_col=True)
    fig.savefig(os.path.join(BENCHMARK_PATH,"column_density."+args.format))

    fig, _ = observation.plot_fractal_dim(suffixes=["_"+s for s in args.obs_suffixes], thresholds=[l for l in np.logspace(np.log10(30), np.log10(1e5), 30)], colors=colors)
    fig.savefig(os.path.join(BENCHMARK_PATH,"fractal_dim."+args.format))

accuracy_fig.savefig(os.path.join(BENCHMARK_PATH,"accuracy."+args.format))
for name, ax in zip(global_axes.keys(), global_axes.values()):
    fig = ax.get_figure()
    fig.savefig(os.path.join(BENCHMARK_PATH,name+"."+args.format))

string = dictsToString(in_files)
with open(os.path.join(BENCHMARK_PATH, "benchmark.txt"), "w") as file:
    file.write(f"Benchmark done in {_format_time(time.process_time()-start_time)}"+"\n")
    file.write(f"---------------------------------------------"+"\n")
    file.write(string)

LOGGER.log(f"Benchmark done in {_format_time(time.process_time()-start_time)}.")