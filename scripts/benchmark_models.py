import os
from POLARIScore.config import EXPORT_FOLDER, LOGGER, DATA_NORMALIZATION_CDENS, DATA_NORMALIZATION_VDENS
from POLARIScore.utils.utils import dictsToString, plot_map, plot_rect_bg
from POLARIScore.networks.utils.nn_utils import find_error_for_batch_accuracy
import uuid
from typing import Tuple, List
from matplotlib.colors import LogNorm, CenteredNorm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import argparse
import numpy as np
import time
from POLARIScore.objects.Dataset import Dataset, getDataset
from POLARIScore.objects.Simulation_DC import Simulation_DC
from POLARIScore.objects.SimulationArray import SimulationArray

"""
Generate full benchmark of trained models with plots.

Usage example:
#python -m POLARIScore.scripts.benchmark_models --models UNet cINN DDPM --obs_name OrionB --obs_suffixes unet cinn ddpm  --name Benchmark_OrionB --toplot distribution

#python -m POLARIScore.scripts.benchmark_models --models UNet DDPM --obs_name OrionB --obs_suffixes unet ddpm  --name Benchmark_test --toplot region --ds_imgs 0 1 2


POLARIScore.scripts.benchmark_models --obs_name OrionB --obs_suffixes unet_2 cinn_2 ddpm_2 --models UNet cINN DDPM  --name Benchmark_OrionB_likeliest
"""

#If you want to add zoom in regions
from astropy.coordinates import Angle
REGIONS = [
[Angle("5h50m").deg, Angle("5h45").deg, Angle("-0d19m").deg, Angle("0d53m").deg],[Angle("5h48m").deg, Angle("5h39m").deg, Angle("-3d10m").deg, Angle("-0d58m").deg]
]
FIGURE_CMAP = "jet"

MONTE_CARLO = 30

INPUT_LABEL = r"$N_H(cm^{-2})$"
TARGET_LABEL = r"$<n_H>_m(cm^{-3})$"

start_time = time.process_time()
def _format_time(seconds:float)->str:
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

parser = argparse.ArgumentParser()
parser.add_argument("--models", required=True, nargs="+", help="List of model names")
parser.add_argument("--trainers", required=False, nargs="+", default=None, help="List of which trainers to use in the same order of the models, can be 'ddpm', 'inn', any other string leads to default trainer.")
parser.add_argument("--toplot", required=False, nargs="+", default=["all"], help="what to plot")


parser.add_argument("--obs_name", required=False, default=None, help="Molecular cloud name / Folder name of the observation")
parser.add_argument("--obs_catalog", required=False, default="Könyves et al, 2020")
parser.add_argument("--obs_dist", required=False, default=400., help="Distance to the molecular cloud")
parser.add_argument("--obs_suffixes", required=False, nargs="+", default=None, help="Suffixes of the .npy predictions (one for each model given)")
parser.add_argument("--obs_repair", required=False, nargs="+", default=None)

parser.add_argument("--extra_suffixes", required=False, nargs="+", default=None, help="If make benchmark of other observation suffixes not linked to neural networks")

parser.add_argument("--ds", required=False, default=None, help="Force choose the validation set, can also be a simulation if there is a prefix as: 'sim:sim_name' ")
parser.add_argument("--ds_imgs", required=False, nargs="+", default=[], help="Indexes, Plot validation dataset imgs.")
parser.add_argument("--format", required=False, default="jpg", help="Image format (default: jpg)")
parser.add_argument("--density_correction", required=False, default="fixed", help="Apply two mediums approx to go from the mass-weighted density to core volume density.")

parser.add_argument("--figures", required=False, default="no", help="If you wants to save each figure in a extra folder.")
parser.add_argument("--figures_cbar", required=False, default="yes", help="If you wants to enable cbar for individual figures.")

parser.add_argument("--name", required=False, default=str(uuid.uuid1()), help="Name of the benchmark (default: uuid1)")
parser.add_argument("--output", required=False, default=EXPORT_FOLDER, help="Benchmark will be generate in this folder (default POLARIScore export folder).")
args = parser.parse_args()

if(str.lower(args.density_correction) in ["false","no","n"]):
    args.density_correction = None

if(str.lower(args.figures) in ["yes","y","true"]):
    args.figures = True
else:
    args.figures = False
if(str.lower(args.figures_cbar) in ["yes","y","true"]):
    args.figures_cbar = True
else:
    args.figures_cbar = False

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
    observation.get_cores(use_deconvolved_values=False)
    observation.distance = float(args.obs_dist)
    assert args.obs_suffixes is not None, LOGGER.error("If an observation is given then you must also give corresponding prediction suffixes.")
    assert len(args.obs_suffixes) == len(trainers), LOGGER.error("Given suffixes must have the same length than given trainers/models.")
    assert args.obs_repair is None or len(args.obs_repair) == len(trainers), LOGGER.errror("Given repair arguments must have the same length than given trainers/models.")

BENCHMARK_PATH = os.path.join(args.output, args.name)
if os.path.exists(BENCHMARK_PATH):
    LOGGER.warn(f"Benchmark folder already exists, if you have bugs delete it. ({BENCHMARK_PATH})")
else:
    os.mkdir(BENCHMARK_PATH)
EXTRA_FIGURES_PATH=None
if args.figures:
    EXTRA_FIGURES_PATH = os.path.join(BENCHMARK_PATH, "independant_figures")
    if not(os.path.exists(EXTRA_FIGURES_PATH)):
        os.mkdir(EXTRA_FIGURES_PATH)

F_DCMF_PATH = os.path.join(BENCHMARK_PATH, "dcmfs")
if not(os.path.exists(F_DCMF_PATH)):
    os.mkdir(F_DCMF_PATH)
F_CORE_HISTS = os.path.join(BENCHMARK_PATH, "core_hists")
if not(os.path.exists(F_CORE_HISTS)):
    os.mkdir(F_CORE_HISTS)
F_CORE_RELATIONS = os.path.join(BENCHMARK_PATH, "core_relations")
if not(os.path.exists(F_CORE_RELATIONS)):
    os.mkdir(F_CORE_RELATIONS)
F_DENSITY_DISTS = os.path.join(BENCHMARK_PATH, "density_dists")
if not(os.path.exists(F_DENSITY_DISTS)):
    os.mkdir(F_DENSITY_DISTS)
F_CLOUDS = os.path.join(BENCHMARK_PATH, "cloud_voldens")
if not(os.path.exists(F_CLOUDS)):
    os.mkdir(F_CLOUDS)
F_CORRELATION_PATH = os.path.join(BENCHMARK_PATH, "correlations")
if not(os.path.exists(F_CORRELATION_PATH)):
    os.mkdir(F_CORRELATION_PATH)

global_axes = {
    "core_residuals": None,
    "core_hists": None,
    "density_dists": None,
    "density_dists_wout_ncol": None,
    "core_diffs": None,
    "power_spectra":None,
    "power_spectra_normalized":None,
}

in_files = []

accuracy_fig, accuracy_axes = plt.subplot_mosaic(
    [["TOTAL" for _ in range(len(trainers))],
     ["TOTAL" for _ in range(len(trainers))],
        ["PLOT"+str(i) for i in range(len(trainers))]],
    figsize=(len(trainers)*3,6), gridspec_kw={"hspace": 0, "wspace": 0}
)
accuracy_axes["TOTAL"].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
for i in range(len(trainers)-1):
    accuracy_axes["PLOT"+str(i+1)].tick_params(left=False, labelleft=False)

if len(args.obs_suffixes) > 0:
    dcmf_fig, dcmf_axes = plt.subplot_mosaic(
        [["PLOT"+str(i) for i in range(len(args.obs_suffixes))]],
        figsize=(len(args.obs_suffixes)*6,4), gridspec_kw={"hspace": 0, "wspace": 0}
    )
    for i in range(len(trainers)-1):
        dcmf_axes["PLOT"+str(i+1)].tick_params(left=False, labelleft=False)


if len(REGIONS) > 0:
    region_axes = [[None for _ in range(1+len(args.obs_suffixes if args.obs_suffixes is not None else [])+len(args.extra_suffixes if args.extra_suffixes is not None else []))] for i in range(len(REGIONS))]
    region_figs = [plt.figure(figsize=(len(region_axes[0])*2.5,6),dpi=300.) for _ in REGIONS]
    for fig in region_figs:
        fig.subplots_adjust(hspace=0.05)

if len(args.ds_imgs) > 0:
    if "range" in args.ds_imgs[0]:
        args.ds_imgs = range(int(args.ds_imgs[0].split("range")[-1]))
    ds_figs = [plt.figure(figsize=((1+len(args.models))*2.5,6),dpi=300.) for _ in args.ds_imgs]
    for fig in ds_figs:
        fig.subplots_adjust(hspace=0.05)
    ds_lims = []
    ds_axes_error = [[None for _ in range(len(args.models))] for i in range(len(args.ds_imgs))]
    ds_axes = [[None for _ in range(len(args.models))] for i in range(len(args.ds_imgs))]
    ds_axes_sim =[[None, None] for i in range(len(args.ds_imgs))]

linestyles = ["-","--","-.",":"]
colors = ["tab:blue","tab:orange","tab:green","tab:red","tab:purple"]

predictions = []

validation_set:Dataset=None

remove_validation_set = False #If a dataset is generated using a simulation, the dataset is removed at the end of the benchmark

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
        if ("c_residual" in args.toplot or "all" in args.toplot) and not("-c_residual" in args.toplot):
            _, ax = observation.plot_cores_error(ax=global_axes["core_residuals"], 
                                                mov_average=0, log_average=50, show_errors=False, show_model_errors=True,
                                                correction=args.density_correction, color=colors[i], label=m)
            global_axes["core_residuals"] = ax

        if ("c_hist" in args.toplot or "all" in args.toplot) and not("-c_hist" in args.toplot):
            _, ax = observation.plot_cores_hist(ax=global_axes["core_hists"], plot_catalog=global_axes["core_hists"] is None, label=m, correction=args.density_correction)
            global_axes["core_hists"] = ax

        if ("distribution" in args.toplot or "all" in args.toplot) and not("-distribution" in args.toplot):
            if global_axes["density_dists"] is None:
                _, ax = observation.plot_pdf(ax=global_axes["density_dists"], monte_carlo=0, color="black", label="$N_H$", offset_method="max", what="data")
                global_axes["density_dists"] = ax
            _, ax = observation.plot_pdf(ax=global_axes["density_dists"], monte_carlo=MONTE_CARLO, color=colors[i], label=m+r" $<n_H>_m$", offset_method="max")
            _, ax2 = observation.plot_pdf(ax=global_axes["density_dists_wout_ncol"], monte_carlo=MONTE_CARLO, color=colors[i], label=m+r" $<n_H>_m$")
            global_axes["density_dists"] = ax
            global_axes["density_dists_wout_ncol"] = ax2
            global_axes["density_dists"].set_xlim([5e-1, 8])
            global_axes["density_dists"].set_ylim([1e-5, 4])
            global_axes["density_dists_wout_ncol"].set_xlim([1, 1e7])
            global_axes["density_dists_wout_ncol"].set_ylim([1e-5, 4])

        if ("c_diff" in args.toplot or "all" in args.toplot) and not("-c_diff" in args.toplot):
            _, ax = observation.plot_cores_mass(ax=global_axes["core_diffs"], bins_mean=20, label=m, show_errors=False, linestyle=linestyles[i])
            global_axes["core_diffs"] = ax

        if ("power_spectra" in args.toplot or "all" in args.toplot) and not("-power_spectra" in args.toplot):
            _, ax = observation.plot_power_spectrum(ax=global_axes["power_spectra"], bins=40, label=m, color=colors[i], normalize=False, plot_coldens=False)
            global_axes["power_spectra"] = ax
            _, ax = observation.plot_power_spectrum(ax=global_axes["power_spectra_normalized"], bins=40, label=m, color=colors[i], normalize=True, plot_coldens=global_axes["power_spectra_normalized"] is None)
            global_axes["power_spectra_normalized"] = ax

        #One figure per model:

        if ("correlation" in args.toplot or "all" in args.toplot) and not("-correlation" in args.toplot):
            fig, _ = observation.plot_correlation()
            fig.savefig(os.path.join(F_CORRELATION_PATH,"correlation_"+m+"."+args.format))
            plt.close(fig)

        if ("dcmf" in args.toplot or "all" in args.toplot) and not("-dcmf" in args.toplot):
            fig, _ = observation.plot_dcmf(method="constant", monte_carlo=MONTE_CARLO, fit=False, bins=15, correction=args.density_correction, color=colors[i])
            fig.savefig(os.path.join(F_DCMF_PATH,"dcmf_"+m+"."+args.format))
            plt.close(fig)
            _, t_ax = observation.plot_dcmf(ax=dcmf_axes["PLOT"+str(i)] ,method="constant", monte_carlo=MONTE_CARLO, fit=False, bins=15, correction=args.density_correction, color=colors[i])
            if i != 0:
                t_ax.set_ylabel("")

        if ("c_hist" in args.toplot or "all" in args.toplot) and not("-c_hist" in args.toplot):
            fig, _ = observation.plot_cores_hist(label=m, correction=args.density_correction)
            fig.savefig(os.path.join(F_CORE_HISTS,"core_hist_"+m+"."+args.format))
            plt.close(fig)

        if ("voldens" in args.toplot or "all" in args.toplot) and not("-voldens" in args.toplot):
            fig, _ = observation.plot(data=observation.prediction, norm=LogNorm(vmin=1e2,vmax=3e5), plot_cores=False, force_vol=True)
            fig.savefig(os.path.join(F_CLOUDS,"voldens_"+m+"."+args.format))
            plt.close(fig)
        if ("skeleton" in args.toplot or "all" in args.toplot) and not("-skeleton" in args.toplot):
            if observation.get_skeleton() is not None:
                fig, _ = observation.plot(data=observation.prediction, norm=LogNorm(vmin=1e2,vmax=3e5), plot_cores=False, force_vol=True, plot_skeleton=True)
                fig.savefig(os.path.join(F_CLOUDS,"voldens_skeleton_"+m+"."+args.format))
                plt.close(fig)
        if ("region" in args.toplot or "all" in args.toplot) and not("-region" in args.toplot):
            for j,r in enumerate(REGIONS):
                if observation.get_skeleton() is not None:
                    fig, _ = observation.plot(data=observation.prediction, crop=r, norm=LogNorm(vmin=1e2,vmax=3e5), plot_cores=False, force_vol=True, plot_skeleton=True, cmap=FIGURE_CMAP)
                    fig.savefig(os.path.join(F_CLOUDS,f"voldens_skeleton_r{str(j)}_"+m+"."+args.format))
                    plt.close(fig)
                region_ax = plt.subplot2grid((2, len(region_axes[0])),(0, i+1), fig=region_figs[j], projection=observation.wcs)
                _, region_ax = observation.plot(data=observation.prediction, crop=r, ax=region_ax, norm=LogNorm(vmin=50,vmax=3e5), plot_cores=False, force_vol=True,
                                cbar=False, sbar=2., sbar_transparent=False,
                                show_ax_labels=False, toplabel=m, cmap=FIGURE_CMAP
                                )
                if args.figures:
                    fig, _ = observation.plot(data=observation.prediction, crop=r, norm=LogNorm(vmin=50,vmax=3e5), plot_cores=False, force_vol=True,
                        cbar=args.figures_cbar, sbar=2., sbar_transparent=False,
                        show_ax_labels=True, toplabel=m
                        )
                    fig.tight_layout(pad=0)
                    fig.savefig(os.path.join(EXTRA_FIGURES_PATH,f"region{str(j)}_"+m+"."+args.format))
                    plt.close(fig)
                    
                region_axes[j][i+1] = region_ax
            
from POLARIScore.networks import INNTrainer,DDPTrainer,Trainer
full_trainers = [] #I gave up optimizing the memory, TODO, remove it and refactoring residuals plot.
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
    full_trainers.append(trainer)
    if t != "default":
        trainer.norms = {
            "cdens": DATA_NORMALIZATION_CDENS,
            "vdens": DATA_NORMALIZATION_VDENS,
        }

    has_no_val_set = False
    if validation_set is None:
        if args.ds is None:
            validation_set = trainer.validation_set
        else:
            if "sim:" in args.ds or "sims:" in args.ds:
                if "sim:" in args.ds:
                    sim = Simulation_DC(args.ds.split("sim:")[-1])
                    sim.init()
                else:
                    sim = SimulationArray(simulations=[] ,name=args.ds.split("sims:")[-1])
                uuid_name = str(uuid.uuid4())
                sim.generate_dataset(name=uuid_name, number=100)
                validation_set = getDataset("batch_"+uuid_name)
                remove_validation_set = True
            else:
                validation_set = getDataset(args.ds if "batch_" in args.ds else "batch_"+args.ds)
            trainer.validation_set = validation_set

        if validation_set is None:
            LOGGER.warn("No validation set loaded, can make errors if plots need it.")
            has_no_val_set = True
    if not(has_no_val_set):
        if args.ds is not None:
            trainer.validation_set = validation_set
        if validation_set.name != trainer.validation_set.name:
            LOGGER.warn("Two models don't use the same validation dataset, be careful if you plot predictions on validation set.")
        
        validation_batch = [p[1] for p in trainer.get_prediction_batch()]
        for j,img in enumerate(args.ds_imgs):
            img = int(img)
            validation_pair = validation_set.get(list(validation_set.batch.keys())[img])
            if i == 0:
                ds_ax_coldens = plt.subplot2grid((2, 1+len(args.models)),(1, 0), fig=ds_figs[j])
                ds_ax_voldens = plt.subplot2grid((2, 1+len(args.models)),(0, 0), fig=ds_figs[j])

                ds_axes_sim[j] = [ds_ax_coldens,ds_ax_voldens]

                vmin = np.min(validation_pair[validation_set.get_element_index('vdens')])
                vmax = np.max(validation_pair[validation_set.get_element_index('vdens')])
                
                ds_lims.append((vmin, vmax))

                im_coldens = plot_map(validation_pair[validation_set.get_element_index('cdens')],
                                       ax=ds_ax_coldens, norm=LogNorm(), cmap=FIGURE_CMAP, show_ax_labels=False, toplabel="Sim", clabel=INPUT_LABEL,
                                       save=os.path.join(EXTRA_FIGURES_PATH,f"ds{str(j)}_sim_cdens."+args.format) if args.figures else None)
                im_voldens = plot_map(validation_pair[validation_set.get_element_index('vdens')],
                                       ax=ds_ax_voldens, norm=LogNorm(vmin=vmin,vmax=vmax), cmap=FIGURE_CMAP, show_ax_labels=False, toplabel="Sim", clabel=TARGET_LABEL,
                                       save=os.path.join(EXTRA_FIGURES_PATH,f"ds{str(j)}_sim_vdens."+args.format) if args.figures else None)
                
                ds_figs[j].colorbar(im_coldens,ax=[ds_ax_coldens],orientation="vertical",
                    location="left",fraction=0.03, pad=0.02, label=INPUT_LABEL
                )

            vmin, vmax = ds_lims[j]

            ds_ax = plt.subplot2grid((2, 1+len(args.models)),(0, i+1), fig=ds_figs[j])
            im = plot_map(validation_batch[img], ax=ds_ax, cmap=FIGURE_CMAP, norm=LogNorm(vmin=vmin,vmax=vmax), show_ax_labels=False, toplabel=m,
                          clabel=TARGET_LABEL, save=os.path.join(EXTRA_FIGURES_PATH,f"ds{str(j)}_"+m+"_vdens."+args.format) if args.figures else None)


            ds_ax_error = plt.subplot2grid((2, 1+len(args.models)),(1, i+1), fig=ds_figs[j])
            im_error = plot_map(np.clip(np.log10(validation_batch[img])-np.log10(validation_pair[validation_set.get_element_index('vdens')]),-.5,.5)
                                ,ax=ds_ax_error, cmap="coolwarm", norm=CenteredNorm(), show_ax_labels=False, toplabel=m+"-sim",
                                clabel=r"clipped diff in log10", save=os.path.join(EXTRA_FIGURES_PATH,f"ds{str(j)}_"+m+"_error."+args.format) if args.figures else None)
            ds_axes_error[j][i] = ds_ax_error
            ds_axes[j][i] = ds_ax

            if i == len(args.models)-1:
                ds_figs[j].colorbar(im,ax=[ds_ax],orientation="vertical",
                    location="right",fraction=0.03, pad=0.02, label=TARGET_LABEL
                )
                ds_figs[j].colorbar(im_error,ax=[ds_ax_error],orientation="vertical",
                    location="right",fraction=0.03, pad=0.02, label=r"clipped diff in log10"
                )

            


        if ("accuracy" in args.toplot or "all" in args.toplot) and not("-accuracy" in args.toplot):
            Trainer.plot_accuracy([trainer], ax=accuracy_axes["TOTAL"], linestyle=linestyles[i], color=colors[i], xlabel="", marker="o")
            Trainer.plot_accuracy([trainer], ax=accuracy_axes["PLOT"+str(i)], bins=[0,2.3,4,7], use_linestyles=True, color=colors[i], legend=False, xlabel=r"Error allowed $\sigma$ (log10)" if i == int(len(trainers)/2) else "", ylabel="Accuracy" if i==0 else "")

        in_files.append({
            "model_name": m,
            "inference_time(s/img)": str(trainer.inference_time),
            "inference_speed(img/s)": str(1/trainer.inference_time),
            "parameters":  sum(p.numel() for p in trainer.model.parameters()),
            "MSE": trainer.validation_losses[-1],
            "Error(Acc=80%)": find_error_for_batch_accuracy(trainer.get_prediction_batch(), accuracy=0.8) ,
        })

    if observation is not None :
        make_obs_benchmark(suffix=args.obs_suffixes[i], model_name=m, i=i)
    del trainer

for i in range(len(args.ds_imgs)):
    plot_rect_bg(ds_figs[i], axes=ds_axes_error[i], color="tab:orange", text="Erros: "+r"log$_{10}$$\left( \frac{<n_{H,nn}>_m}{<n_{H,sim}>_m} \right)$")
    plot_rect_bg(ds_figs[i], axes=ds_axes_sim[i], color="tab:green", text="Simulation")
    plot_rect_bg(ds_figs[i], axes=ds_axes[i], color="tab:blue", text="Predictions by NNs")
    ds_figs[i].savefig(os.path.join(BENCHMARK_PATH,f"ds_{str(args.ds_imgs[i])}."+args.format))
    

    
if observation is not None:
    if args.extra_suffixes is not None:
        for i,s in enumerate(args.extra_suffixes):
            make_obs_benchmark(suffix=s, i=len(trainers)+i)

    if ("c_relation" in args.toplot or "all" in args.toplot) and not("-c_relation" in args.toplot):
        fig, _ = observation.plot_cores_density_relation(suffixes=["_"+s for s in args.obs_suffixes], fit=True, derived_cores=True, colors=colors)
        fig.savefig(os.path.join(BENCHMARK_PATH,"core_relations."+args.format))
        plt.close(fig)

    if ("region" in args.toplot or "all" in args.toplot) and not("-region" in args.toplot):
        if len(REGIONS) > 0:
            for i,r in enumerate(REGIONS):
                region_ax = plt.subplot2grid((2, len(region_axes[0])),(0, 0), fig=region_figs[i], projection=observation.wcs)
                _, region_ax = observation.plot(data=observation.data, crop=r, ax=region_ax, norm=LogNorm(vmin=1e21,vmax=None), plot_cores=False, force_col=True, cbar=False, sbar=2., sbar_transparent=False,
                                show_ax_labels=False, toplabel="$N_H$", cmap=FIGURE_CMAP)
                if args.figures:
                    f_add , _ = observation.plot(data=observation.data, crop=r, norm=LogNorm(vmin=1e21,vmax=None), plot_cores=False, force_col=True, cbar=args.figures_cbar, sbar=2., sbar_transparent=False,
                                show_ax_labels=True, toplabel="$N_H$", cmap=FIGURE_CMAP)
                    f_add.tight_layout(pad=0)
                    f_add.savefig(os.path.join(EXTRA_FIGURES_PATH,f"region{str(i)}_cdens."+args.format))
                    plt.close(f_add)
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
                                    crop=r, ax=rgb_ax, plot_cores=False, force_col=True, cbar=False, sbar=2., sbar_transparent=False,
                                    show_ax_labels=False, toplabel="$RGB$")
                    if args.figures:
                        f_add, _ = observation.plot(data=rgb,
                                    crop=r, plot_cores=False, force_col=True, cbar=args.figures_cbar, sbar=2., sbar_transparent=False,
                                    show_ax_labels=True, toplabel="$RGB$")
                        f_add.tight_layout(pad=0)
                        f_add.savefig(os.path.join(EXTRA_FIGURES_PATH,f"region{str(i)}_rgb."+args.format))
                        plt.close(f_add)

                _length = len(args.models)
                sub_axes = []
                for j in range(_length):
                    idx1 = j % _length
                    idx2 = (j+1) % _length
                    ax = plt.subplot2grid((2, len(region_axes[0])),(1, 1+j), fig=region_figs[i], projection=observation.wcs)
                    observation.plot(np.clip(np.log10(np.nan_to_num(predictions[idx2], nan=1.))-np.log10(np.nan_to_num(predictions[idx1], nan=1.)),-.5,.5), ax=ax, crop=r,
                                    cmap="coolwarm", norm=CenteredNorm(), plot_cores=False, show_ax_labels=False, cbar=False, sbar=2., sbar_transparent=False, cores_color="purple",
                                    toplabel=f"{args.models[idx2]}-{args.models[idx1]}")
                    if args.figures:
                        f_add, ax_add = observation.plot(np.clip(np.log10(np.nan_to_num(predictions[idx2], nan=1.))-np.log10(np.nan_to_num(predictions[idx1], nan=1.)),-.5,.5), ax=None, crop=r,
                            cmap="coolwarm", norm=CenteredNorm(), plot_cores=False, show_ax_labels=True, cbar=args.figures_cbar, sbar=2., sbar_transparent=False, cores_color="purple",
                            toplabel=f"{args.models[idx2]}-{args.models[idx1]}")
                        f_add.tight_layout(pad=0)
                        f_add.savefig(os.path.join(EXTRA_FIGURES_PATH,f"region{str(i)}_{str(j)}."+args.format))
                        plt.close(f_add)
                    sub_axes.append(ax)

                region_figs[i].colorbar(region_axes[i][0].images[0],ax=[region_axes[i][0]],orientation="vertical",location="left",fraction=0.03, pad=0.02, label=r"$N_H(cm^{-2})$")
                region_figs[i].colorbar(region_axes[i][1].images[0],ax=region_axes[i][1:][-1],orientation="vertical",location="right",fraction=0.03,pad=0.02, label=r"$<n_H>_m(cm^{-3})$")
                region_figs[i].colorbar(sub_axes[0].images[0],ax=sub_axes[-1],orientation="vertical",location="right",fraction=0.03,pad=0.02, label=r"clipped diff in log10")

                plot_rect_bg(region_figs[i], axes=sub_axes, color="tab:orange", text="Differences between NNs: "+ r"log$_{10}$$\left( \frac{<n_{H,nn1}>_m}{<n_{H,nn2}>_m} \right)$")
                plot_rect_bg(region_figs[i],
                             axes=region_axes[i][1:(len(region_axes[i])-len(args.extra_suffixes))] if args.extra_suffixes is not None else region_axes[i][1:(len(region_axes[i]))]
                             , color="tab:blue", text="Predictions: "+ r"$<n_H>_m$")

                region_figs[i].savefig(os.path.join(BENCHMARK_PATH,f"region_{str(i)}."+args.format))

    
    if ("coldens" in args.toplot or "all" in args.toplot) and not("-coldens" in args.toplot):
        fig, _ = observation.plot(norm=LogNorm(vmin=1e21), plot_cores=False, force_col=True)
        fig.savefig(os.path.join(BENCHMARK_PATH,"column_density."+args.format))
        plt.close(fig)

    if ("fractal" in args.toplot or "all" in args.toplot) and not("-fractal" in args.toplot):
        
        fig, _ = observation.plot_fractal_dim(suffixes=["_"+args.obs_suffixes[0]], thresholds=[2e3], colors=["black"])
        fig.savefig(os.path.join(BENCHMARK_PATH,"fractal_dim_fit."+args.format))
        plt.close(fig)
        fig, _ = observation.plot_fractal_dim(suffixes=["_"+s for s in args.obs_suffixes], thresholds=[l for l in np.logspace(np.log10(30), np.log10(1e5), 30)], colors=colors)
        fig.savefig(os.path.join(BENCHMARK_PATH,"fractal_dim."+args.format))
        plt.close(fig)


if ("accuracy" in args.toplot or "all" in args.toplot) and not("-accuracy" in args.toplot):
    accuracy_fig.savefig(os.path.join(BENCHMARK_PATH,"accuracy."+args.format))
    plt.close(accuracy_fig)

if ("dcmf" in args.toplot or "all" in args.toplot) and not("-dcmf" in args.toplot):
    dcmf_fig.savefig(os.path.join(BENCHMARK_PATH,"dcmfs."+args.format))
    plt.close(dcmf_fig)

if ("residual" in args.toplot or "all" in args.toplot) and not ("-residual" in args.toplot):
    fig, _ = Trainer.plot_models_residuals_extended(trainers=full_trainers, colors=colors)
    fig.savefig(os.path.join(BENCHMARK_PATH,"residuals."+args.format))
    plt.close(fig)

for name, ax in zip(global_axes.keys(), global_axes.values()):
    if ax is None:
        continue
    fig = ax.get_figure()
    fig.savefig(os.path.join(BENCHMARK_PATH,name+"."+args.format))
    plt.close(fig)

string = dictsToString(in_files)
with open(os.path.join(BENCHMARK_PATH, "benchmark.txt"), "w") as file:
    file.write(f"Benchmark done in {_format_time(time.process_time()-start_time)}"+"\n")
    file.write(f"What was drawn: "+str(args.toplot)+".\n")
    file.write(f"---------------------------------------------"+"\n")
    file.write(string)

if remove_validation_set:
    validation_set.delete()

LOGGER.log(f"Benchmark done in {_format_time(time.process_time()-start_time)}.")