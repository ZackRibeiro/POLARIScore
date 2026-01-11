import os
from POLARIScore.config import EXPORT_FOLDER, LOGGER
import uuid
from typing import Tuple, List
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import argparse
import numpy as np
import time

"""
Usage:
python -m POLARIScore.scripts.makeBenchmark --models UNet cINN DDPM --obs_name OrionB --obs_suffixes unet cinn ddpm --obs_repair no yes yes --name Benchmark_OrionB
"""

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

linestyles = ["-","--","-.",":"]
from POLARIScore.networks import INNTrainer,DDPTrainer,Trainer
for i,t,m in zip(range(len(trainers)),trainers, args.models):
    used_trainer = Trainer.Trainer 
    t = "default"
    if t == "ddpm" or t == "ddp":
        t = "ddpm"
        used_trainer = DDPTrainer.DDPTrainer
    elif t == "inn" or t == "cinn":
        used_trainer = INNTrainer.INNTrainer
        
    trainer = Trainer.load_trainer(m, trainer_class=used_trainer)

    if observation is not None :

        observation.load(suffix="_"+args.obs_suffixes[i])
        observation.load_error(model_name=m)
        if args.obs_repair is not None:
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

        fig, _ = observation.plot_dcmf(method="constant", monte_carlo=20, fit=False, bins=15, correction=args.density_correction)
        fig.savefig(os.path.join(F_DCMF_PATH,"dcmf_"+m+"."+args.format))
        plt.close(fig)

        fig, _ = observation.plot_cores_hist(label=m, correction=args.density_correction)
        fig.savefig(os.path.join(F_CORE_HISTS,"core_hist_"+m+"."+args.format))
        plt.close(fig)

        fig, _ = observation.plot_cores_baseline(derived_cores=True, density_correction=args.density_correction, invert_xy=True, x_coldens=True, mov_average=1, fit=True, forced_label=m)
        fig.savefig(os.path.join(F_CORE_RELATIONS,"core_relations_"+m+"."+args.format))
        plt.close(fig)

        fig, _ = observation.plot_density_distributions(monte_carlo=0, offset_method="max", color="red", draw_style=None, label=m, marker=None)
        fig.savefig(os.path.join(F_DENSITY_DISTS,"density_dists_"+m+"."+args.format))
        plt.close(fig)

        fig, _ = observation.plot(data=observation.prediction, norm=LogNorm(vmin=1e2,vmax=3e5), plot_cores=False, force_vol=True)
        fig.savefig(os.path.join(F_CLOUDS,"voldens_"+m+"."+args.format))
        if observation.get_skeleton() is not None:
            fig, _ = observation.plot(data=observation.prediction, norm=LogNorm(vmin=1e2,vmax=3e5), plot_cores=False, force_vol=True, plot_skeleton=True)
            fig.savefig(os.path.join(F_CLOUDS,"voldens_skeleton_"+m+"."+args.format))
        plt.close(fig)

    del trainer
    

if observation is not None:
    fig, _ = observation.plot(norm=LogNorm(vmin=1e21), plot_cores=False, force_col=True)
    fig.savefig(os.path.join(BENCHMARK_PATH,args.obs_name+"_col."+args.format))

    fig, _ = observation.plot_fractal_dim(suffixes=["_"+s for s in args.obs_suffixes], thresholds=[l for l in np.logspace(np.log10(30), np.log10(1e5), 30)])
    fig.savefig(os.path.join(BENCHMARK_PATH,"fractal_dim"+args.format))

for name, ax in zip(global_axes.keys(), global_axes.values()):
    fig = ax.get_figure()
    fig.savefig(os.path.join(BENCHMARK_PATH,name+"."+args.format))

end_time = time.process_time()

LOGGER.log(f"Benchmark done in {_format_time(end_time-start_time)}.")