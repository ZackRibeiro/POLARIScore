from POLARIScore.utils.vtk_io import readVTKCart
from POLARIScore.utils import compute_pdf
from POLARIScore.config import DATA_FOLDER
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib
from scipy.optimize import curve_fit
from typing import List

from POLARIScore.objects.Simulation_DC import Simulation_DC, openSimulation
from POLARIScore.objects.Simulation_AMR import *
from POLARIScore.utils.sim_utils import init_idefix,init_ramses
from POLARIScore.utils.utils import compute_mass_weighted_density
from POLARIScore.config import *

from POLARIScore.objects.SimulationArray import SimulationArray
from POLARIScore.objects.Dataset import Dataset, getDataset
from torch import nn
from POLARIScore.objects.Observation import Observation
from POLARIScore.networks.DDPTrainer import DDPTrainer, DDPMUnet
from POLARIScore.networks.Trainer import Trainer, load_trainer
from POLARIScore.networks.INNTrainer import INNTrainer
from POLARIScore.networks.architectures.nn_SpectraNetwork import SpectraNetwork
from POLARIScore.networks.architectures.nn_UNet import UNet
from POLARIScore.objects.Observation_Sim import Observation_Sim
from POLARIScore.config import DATA_NORMALIZATION_CDENS, DATA_NORMALIZATION_VDENS, DATA_NORMALIZATION_CDENS_TORCH, DATA_NORMALIZATION_VDENS_TORCH
from POLARIScore.objects.SpectrumMap import SpectrumMap, getSimulationSpectra
from POLARIScore.objects.Spectrum import Spectrum
from POLARIScore.networks.utils.nn_utils import open_samples_as_spectrummap

#sim = SimulationArray(name="sim_512_A_3")
#sim = openSimulation("orionMHD_lowB_multi_", global_size=66.0948+0.12,keys=['RHO'],cache_name="orion") #offset bcs without dense cores have an offset :/
#sim.plot(axis=-1)

sim = Simulation_AMR("orionMHD_lowB_AMR", global_size=66.0948+0.12, init_datacubes=False)
#bbox = np.array([34.6, 35.4, 34.5, 36.3])
bbox = np.array([34.1, 34.8, 35.2, 35.9])


tensor = sim.to_datacube("RHO", res=512, bbox=[[bbox[0], bbox[1]], [bbox[2], bbox[3]], None],
                          filling_method=lambda t: fill_zeros_slice(t, method=fill_zeros_nearest, axis=-1), smoothing=0., force=True)
plt.figure()
plt.imshow(np.sum(tensor, -1)*(sim.size*PC_TO_CM/512), cmap="jet", norm=LogNorm(), extent=bbox)
#sim.init_datacubes(res=512)
#

#sim.load_cores()
#smap = sim.format_key_to_spectrum_map()
#smap.gaussians(fit_method="iterative")
#smap.pca(plot=True, return_cube=False)

#sim.generate_dataset(name="highres_2",what_to_compute={"vdens":compute_mass_weighted_density},number=200, img_size=128, random_rotate=True)
#ds = getDataset("highres_2")
#ds1, ds2 = ds.split()
#ds1.save()
#ds2.save()

#sim.plot_pdf(what='rho')
#sim.load_cores()

AXIS = 0
#----------------------------------------
#REMOVE CORES if they are in the SAME L.O.S
#----------------------------------------
"""print(sim.get_cores_multiplicity(include_resolution=True, include_scale=False))
new_cores = []
for c in sim.cores:
    if 'confused' in c and c['confused'][AXIS]:
        continue
    new_cores.append(c)
sim.cores = new_cores"""

#obs = Observation_Sim(sim, axis=AXIS)
#path_samples = "pdf_orionb_cached"

#obs = Observation("OrionB", "column_density_map")

#obs.catalog_name = "Ntormousi & Hennebelle"
#trainer = load_trainer("DDPM")
#trainer = load_trainer("DDPM", trainer_class=DDPTrainer)
#trainer.norms = {
#    "cdens": DATA_NORMALIZATION_CDENS,
#    "vdens": DATA_NORMALIZATION_VDENS,
#}
#trainer.get_validation_error()
#trainer.plot_residuals()
#_, error = obs.predict(trainer, method="likeliest", repeat=1, overlap=0.75, downsample_factor=obs.find_scale(3.30474,128,400), nan_value=1e19, apply_baseline=True, kernel="gaussian", save_samples=path_samples, skip_using_saved_samples=True, only_error=False)

#obs.save("_ddpm_likeliest_gaussian")
#obs.load("_cinn")
#obs.plot_cores_error(show_errors=False, label="cinn",correction="blurred", log_average=30)

#obs.plot(error/obs.prediction, norm=None)
#obs.load_error("DDPM")
#obs.prediction = obs.rectify_error_baseline()
#obs.plot(obs.prediction, norm=LogNorm(vmin=1e2, vmax=1e6))

#----------------------------------------
#Blurred correction vs Fixed correction vs No correction
#----------------------------------------
#obs = Observation_Sim(sim, axis=AXIS)
#obs.prediction = compute_mass_weighted_density(sim.data['RHO'], axis=AXIS)
#_, ax = obs.plot_cores_error(show_errors=False, label="blurred",correction="blurred", log_average=30)
#_, ax = obs.plot_cores_error(ax=ax, show_errors=False, label="fixed",correction="fixed", log_average=30)
#_, ax = obs.plot_cores_error(ax=ax, show_errors=False, label="no correction",correction=None, log_average=30)
#obs.plot(obs.data)
#obs.plot(obs.convolved_data)

#obs.plot_dcmf()

#_, ax = obs.plot_dcmf()
#obs.load("_ddpm")
#obs.plot_dcmf(color="green")
#obs.plot_cores_error(ax=ax, show_errors=False, label="mean and uniform",correction=True, log_average=30)

#obs.predict(trainer, method="likeliest", repeat=1, overlap=0.75, nan_value=1e19, apply_baseline=True, kernel="uniform", save_samples=path_samples, skip_using_saved_samples=True, only_error=False)
#obs.plot_cores_error(ax=ax, show_errors=False, label="uniform",correction=True, log_average=30)



#obs.load("_likeliest_gaussian")
#_, ax = obs.plot_cores_error(show_errors=False, label="gaussian",correction=False, log_average=30)
#obs.plot_error_histogram(min_truth=[10,1e2,5e3,5e4,np.inf])
#obs.load("_likeliest_uniform")
#obs.plot_error_histogram(min_truth=[10,1e2,5e3,5e4,np.inf])
#obs.plot_cores_error(ax=ax, show_errors=False, label="uniform",correction=False, log_average=30)

#obs.prediction = compute_mass_weighted_density(sim.data['RHO'], axis=0,)
#obs.plot_cores_error(ax=ax, show_errors=False, label="simulation",correction=False, log_average=30)
#obs.plot(obs.prediction, norm=LogNorm(vmin=1e2, vmax=1e6))


#path_to_changes = ["pdf_orionb_cached_gaussian_weight.npy","pdf_cached_gaussian_weight.npy","pdf_cached.npy"]
#for p in path_to_changes:
#    path = os.path.join(CACHES_FOLDER,p)
#    arr = np.load(path)
#    new_arr = np.moveaxis(arr, 0, -1)
#    np.save(path, new_arr)

#smap = open_samples_as_spectrummap(path_samples, 16)
#smap.plot()

#obs.load()
#obs.plot_error_histogram()
#obs.save("_median")
#obs.plot_error_histogram()
#pred_like = obs.load("_likeliest")
#obs.plot_correlation()
#obs.load("_median")
#_, ax = obs.plot_cores_error(ax=ax, show_errors=False, label="Median",correction=False)
#pred_mean = obs.load()
#obs.plot_correlation()
#_, ax = obs.plot_cores_error(ax=ax,show_errors=False, label="mean",correction=False)
#obs.plot_correlation()
#obs.plot_cores_error(ax=ax,show_errors=False, label="sim",correction=False, log_average=30)

#obs.plot(error/obs.prediction, plot_cores=False, norm=None)
#obs.plot(obs.prediction, plot_cores=False, clabel="sim")
#obs.plot(pred_like, plot_cores=False, clabel="like")
#obs.plot(pred_mean, plot_cores=False, clabel="mean")

#histrogram rapport predic/simu 
#Blur to get diffuse col dens



#obs.plot_cores_baseline(mov_average=0, fit=True)
#obs.plot_dcmf(lims=None, monte_carlo=10)

#cores, pos = sim.get_cores(axis=1, box=[27.5, 29.5, 37.5, 38.75])
#sim.plot(axis=0)
#sim.plot_slice(axis=1)
#sim.generate_dataset(name="seg_orion_cores",what_to_compute={"vdens":compute_mass_weighted_density,"cores":True},number=200, img_size=128, random_rotate=True)

#ds = getDataset("seg_orion_cores")
#ds.plot_map(map_index=ds.get_element_index("vdens"), element_index=1)

#sim.get_core_distance_map(axis=1)
#sim.get_core_volumes(0, plot=True)
#print(sim.get_cores_multiplicity()*100)
#sim.plot()
#sim.plot_slice(axis=2)
#sim = SimulationArray(simulations=[Simulation_DC("sim_256_A_5"),Simulation_DC("sim_512_A_3")], indexes=[256,512])
#sim = SimulationArray(simulations=[Simulation_DC("adastra_512_old"),Simulation_DC("adastra_512")], indexes=[0,1])


#summed_densities = []
#for s in sim.simulations:
#    summed_densities.append(np.sum(s.data['RHO']))
#print(summed_densities)

#_, ax = sim.plot_pdf(what="vdens", vdens_method=compute_mass_weighted_density, offset_method="max")
#sim.plot_pdf(ax=ax, what="cdens", offset_method="max")


#sim.plot(plot_method=Simulation_DC.plot_pdf, what="cdens", colors="viridis", drawstyle=None)
#sim.plot(plot_method=Simulation_DC.plot, mode="slider")#, norm=LogNorm(vmin=1e3, vmax=1e6), method=compute_mass_weighted_density, label=r"$<n_H>_m$")

#sim.plot()
#sim.plot_pdf(what="cdens")

#sim.generate_dataset("idefix_512_A_training", number=100, axes=[0,2])
#sim.generate_dataset("idefix_512_A_validation", number=100, axes=[1])

#training_ds = getDataset("batch_idefix_512_A_training")
#validation_ds = getDataset("batch_idefix_512_A_validation")

"""
#trainer = DDPTrainer(network=DDPMUnet, training_set=training_ds, validation_set=validation_ds, model_name="idefix_512_ddpm")
trainer = Trainer(network=UNet, training_set=training_ds, validation_set=validation_ds, model_name="idefix_512_unet")

#trainer.pred_type = "v"
import torch
def classic_log_mse(output, target):
        output_phys = DATA_NORMALIZATION_VDENS_TORCH[1](output)
        target_phys = DATA_NORMALIZATION_VDENS_TORCH[1](target)
        output_log = torch.log(output_phys)
        target_log = torch.log(target_phys)
        mse = torch.mean((output_log - target_log) ** 2)
        return mse


trainer.norms = { 
    "cdens": DATA_NORMALIZATION_CDENS,
    "vdens": DATA_NORMALIZATION_VDENS,
}
trainer.network_settings['num_layers']=4
trainer.network_settings['base_filters']=64
trainer.optimizer_name = "Adam"
trainer.learning_rate = 1e-4
#trainer.ema = True
#trainer.ema_warmup = 50
#trainer.network_settings["attention_layers"] = [3]
#trainer.network_settings["attention_heads"] = [8]    
trainer.validation_loss_method = classic_log_mse
trainer.training_random_transform = True
trainer.input_names = ["cdens"]
trainer.target_names = ["vdens"]
trainer.init()
trainer.train(1000, batch_number=8, compute_validation=10, early_stopping=False)
trainer.save()
trainer.plot()
trainer.plot_validation()
"""

"""
from POLARIScore.objects.Observation import Observation
#trainer = load_trainer("idefix_512_unet", trainer_class=Trainer)
#trainer.get_validation_error()
#trainer.norms = {
#    "cdens": DATA_NORMALIZATION_CDENS,
#    "vdens": DATA_NORMALIZATION_VDENS,
#}


obs = Observation("OrionB","column_density_map")

#obs.load("_cinn")
#obs.plot_cores_error(correction=False)
#obs.plot_dcmf(monte_carlo=0, method="gaussian")

beam_width = 18.2/206265*400
fig, ax = plt.subplots()
cores = obs.get_cores()
convolved_radius = [c.data['radius_pc'] for c in cores]
print(all([c.data['radius_pc']>beam_width*np.sqrt(5) for c in cores]))
deconvolved_cores = obs.get_cores(force_compute=True, use_deconvolved_values=True)
deconvolved_radius = np.array([c.data['radius_pc'] for c in deconvolved_cores])

diffs = [np.sqrt(convolved_radius[i]**2-deconvolved_radius[i]**2) for i in range(len(convolved_radius))] 

print(np.min(diffs),np.max(diffs), np.mean(diffs), beam_width)

x,y = convolved_radius, deconvolved_radius
#ax.scatter(x,y,marker="+",color="red",label="cores")
h = ax.hexbin(x, y, bins=40, cmap='viridis', norm=LogNorm())
threshold = beam_width
ax.axvline(threshold, color="black")
ax.axhline(threshold, color="green")
ax.text(threshold - 0.005, np.mean(y)+0.4,rf'Threshold: $={threshold:.2f}$pc',
            rotation=90,va='center',ha='left',color='black',fontsize=11, transform=ax.get_xaxis_transform())
ax.plot([np.min(x), np.max(x)], [np.min(x), np.max(x)], color="black", label="y=x")

ax.plot(np.linspace(np.min(x),np.max(x),100), np.sqrt(np.power(np.linspace(np.min(x),np.max(x),100),2)-beam_width**2), color="red", label=r"$y^2=x^2-\mathrm{beam_width}^2$")

ax.axvspan(min(x), threshold,
           facecolor='none',
           edgecolor='black',
           hatch='//',
           alpha=0.3)
ax.fill_between(np.linspace(threshold, max(x),100), np.linspace(threshold, max(x),100) , max(y),
                facecolor='none',
                edgecolor='black',
                hatch='//',
                alpha=0.3)
ax.annotate("resolved",
            xy=(threshold, np.mean(y)),
            xytext=(threshold + 0.01, np.mean(y)),
            arrowprops=dict(arrowstyle="<-", color="black"))
ax.set_xlabel("Convolved radii")
ax.set_ylabel("Deconvolved radii")
ax.set_xlim([min(x),max(x)])
ax.set_ylim([min(y),max(y)])
plt.colorbar(h, ax=ax, label="Core counts")
ax.legend()

flags = deconvolved_radius > threshold

obs.get_cores(force_compute=True)
new_cores = []
for i in range(len(obs.cores)):
    if flags[i]:
        new_cores.append(obs.cores[i])
obs.cores = new_cores

obs.load("_cinn")
obs.load_error("cINN")
obs.plot_cores_error(correction=False)
obs.plot_dcmf(monte_carlo=30, method="constant", bins=20)

#Test sur catalogue ntormousi (garder que coeurs avec taille suffisante)
#Lire louvet 2021
"""


"""cores = obs.get_cores()
true_col_dens = np.array([c.get_center_density(column_density=True) for c in cores])
konyves_col_dens = np.array([c.data["peak_ncol"] for c in cores])
fig, ax = plt.subplots()
ax.scatter(true_col_dens, konyves_col_dens, color="black", marker="+")
ax.set_xscale("log")
ax.set_yscale("log")
ax.plot([np.min(true_col_dens), np.max(true_col_dens)], [np.min(true_col_dens), np.max(true_col_dens)], color="black")
ax.set_xlabel("konyves catalog Nh")
ax.set_ylabel("herschel map Nh")
"""

#obs.serialize_cores(suffixes=["_unet","_cinn","_ddpm"])
#obs.plot_density_distributions(what="data", monte_carlo=0)
#obs.plot_validity_with_model("batch_idefix_512_A_training")
#obs.plot_validity_with_model("batch_idefix_512_A_training", patch_size=(512,512), c_x=lambda x: np.std(np.log10(x)), c_y=lambda x: np.log10(np.mean(x)), logspace=False)

#obs.predict(trainer,patch_size=(128,128), overlap=0.5, downsample_factor=obs.find_scale(1.25,128,obs.distance), nan_value=1e20, apply_baseline=True)
#obs.save(suffix=f"_unet_idefix")
#obs.load(suffix=f"_unet_idefix")
#obs.plot(data=obs.prediction, norm=LogNorm(vmin=2e2), plot_skeleton=False)
#obs.plot_dcmf(monte_carlo=20, correction=False)


"""
sim_names = ["turb_sim_A"]#,"turb_sim_B","turb_sim_C"]
GENERATE_DATASET = False
number_per_face = int(128*128*.05)
if GENERATE_DATASET:
    datasets_to_merge:List['Dataset'] = []
    for s in sim_names:
        sim = Simulation_DC(s)
        maps = getSimulationSpectra(simulation=sim, axes=[0,1,2])
        for i, m in enumerate(maps):
            ds = getDataset("batch_spectra_"+str(s)+"_"+str(i))
            if ds is not None:
                datasets_to_merge.append(ds)
                continue
            m.generate_dataset(name="spectra_"+str(s)+"_"+str(i), number=number_per_face)
            datasets_to_merge.append(getDataset("batch_spectra_"+str(s)+"_"+str(i)))
    datasets_to_merge[0].merge(datasets_to_merge[1:], delete=False, save=True, name="spectra")
    ds = getDataset("batch_spectra")
    ds1, ds2 = ds.split(0.8)
    ds1.save()
    ds2.save()
training_set, validation_set = getDataset("batch_spectra_b1"), getDataset("batch_spectra_b2")

#training_set.check_sanity(remove=True)
#validation_set.check_sanity(remove=True)

def _plot_spectrum(ds:'Dataset',ds_index:int=0):
    data = ds.get(ds_index)
    env_spectra = data[ds.get_element_index("noisy_spectrum")]
    spectrum = env_spectra[len(env_spectra)//2][len(env_spectra)//2]
    spect = Spectrum(spectrum, name="test")
    spect.X = data[ds.get_element_index("channels")]
    gauss_params = data[ds.get_element_index("gaussians") ]
    spect.fit_settings = None , {'params':gauss_params, 'N':len(gauss_params)//3}
    spect.plot(show_fit=True, show_dendrogram=False)

trainer = Trainer(network=SpectraNetwork, training_set=training_set, validation_set=validation_set, model_name="Spectral_Fit")
#trainer = load_trainer("cached_model")
trainer.model_name = "Spectral_Fit"
trainer.network_settings['num_layers']=4
trainer.network_settings['out_features']=10*3
trainer.network_settings['base_filters']=64
trainer.network_settings['environment_dim']=3
trainer.network_settings['spectra_dim']=128

trainer.norms = { 
    "channels": (lambda x:x,lambda x:x),
    "noisy_spectrum": (lambda x:x, lambda x:x),
    "gaussians_amplitudes": (lambda x:x, lambda x:x),
    "gaussians_sigmas": (lambda x:x, lambda x:x),
    "gaussians_means": (lambda x:x, lambda x:x),
    "snr": (lambda x:x, lambda x:x),
}


import torch
import torch.nn.functional as F
def _gaussian(x, A, mu, sigma):
    return A * torch.exp(-.5*((x - mu) / (sigma))**2)

def _gaussian_sum(amps, means, sigmas, N):
    B = means.shape[0]
    x = torch.linspace(-1, 1, steps=128, device=means.device).view(1, 1, 128).expand(B, 1, 128)
    y = torch.zeros_like(x)
    for i in range(N):
        A = amps[:,i]
        mu = means[:,i]
        sigma = sigmas[:,i]

        gaussian = _gaussian(x, A, mu, sigma)
        y = y + gaussian

    return y

from scipy.optimize import linear_sum_assignment
def spectrum_loss(output, target):

    means = target[1].clone().squeeze(1)
    amps = target[0].clone().squeeze(1)
    sigmas = target[2].clone().squeeze(1)
    sorted_indexes = torch.argsort(means, dim=-1)
    means = torch.gather(means, -1, sorted_indexes)
    amps = torch.gather(amps, -1, sorted_indexes)
    sigmas = torch.gather(sigmas, -1, sorted_indexes)

    predicted_means = output[1].clone().squeeze(1)
    predicted_amps = output[0].clone().squeeze(1)
    predicted_sigmas = output[2].clone().squeeze(1)
    predicted_sorted_indexes = torch.argsort(predicted_means, dim=-1)
    predicted_means = torch.gather(predicted_means, -1, predicted_sorted_indexes)
    predicted_amps = torch.gather(predicted_amps, -1, predicted_sorted_indexes)
    predicted_sigmas = torch.gather(predicted_sigmas, -1, predicted_sorted_indexes)

    cost_mu = (predicted_means[:,:,None] - means[:,None,:])**2
    cost_sigma = (predicted_sigmas[:,:,None] - sigmas[:,None,:])**2
    cost_amp   = (predicted_amps[:,:,None] - amps[:,None,:])**2
    cost = cost_mu + cost_sigma + cost_amp

    reordered_pred_amp = []
    reordered_pred_mu = []
    reordered_pred_sigma = []


    B = predicted_means.shape[0]
    for b in range(B):

        row, col = linear_sum_assignment(cost[b].detach().cpu().numpy())

        reordered_pred_amp.append(predicted_amps[b, row])
        reordered_pred_mu.append(predicted_means[b, row])
        reordered_pred_sigma.append(predicted_sigmas[b, row])

        amps[b] = amps[b, col]
        means[b] = means[b, col]
        sigmas[b] = sigmas[b, col]

    predicted_amps = torch.stack(reordered_pred_amp)
    predicted_means = torch.stack(reordered_pred_mu)
    predicted_sigmas = torch.stack(reordered_pred_sigma)

    #y_true = _gaussian_sum(amps, means, sigmas, 10)
    #y_predict = _gaussian_sum(predicted_amps, predicted_means, predicted_sigmas, 10)
    #l_predict = F.mse_loss(y_predict, y_true)

    l_amps = F.mse_loss(predicted_amps,amps)
    l_means = F.mse_loss(predicted_amps*predicted_means,amps*means)
    l_sigmas = F.mse_loss(predicted_amps*predicted_sigmas,amps*sigmas)
    
    
    loss = l_amps+l_means+l_sigmas*10.

    return loss

#torch.autograd.set_detect_anomaly(True)
trainer.optimizer_name = "Adam"
trainer.learning_rate = 1e-6
trainer.init()
trainer.loss_method = spectrum_loss
trainer.validation_loss_method = spectrum_loss

trainer.training_random_transform = False
trainer.input_names = ["noisy_spectrum","snr", "channels"]
trainer.target_names = ["gaussians_amplitudes","gaussians_means","gaussians_sigmas"]
trainer.train(50, compute_validation=1, batch_number=256, early_stopping=False, training_mode="accumulation")
#trainer.save()
trainer.plot_losses()
val_batch = trainer.get_prediction_batch()
print(val_batch[3*10+0])
print(val_batch[3*10+1])
print(val_batch[3*10+2])
"""

#sim = SimulationArray(name="turb_sim_A")
#sim.plot(
#    plot_method=Simulation_DC.plot_pdf,
#    colors="viridis",
#    offset_method="none",
#    what="rho",
#    drawstyle=None, swap_axis=True
#)




#v_map = sim.compute_velocity_decomposition()
#v_map.plot()


#sim = Simulation_DC("orionHD_all_512")
#sim = Simulation_DC("turb_sim_A")
#sim.plot_power_spectrum(what_to_plot="rms_velocity", bins=30, energy=True )

"""
from POLARIScore.utils.physics_utils import dcmf_func, density_gaussian
from POLARIScore.utils.utils import plot_function
_dcmf_function = lambda M,amp,mu,sigma,alpha,cutoff: dcmf_func(M,amp,mu,sigma,alpha,cutoff, enable_cutoff=False)


pdf = compute_pdf(sim.data['RHO']/np.mean(sim.data['RHO']))
bin_centers = pdf[1][:100]
values = pdf[0]

popt, _ = curve_fit(_dcmf_function, (10**bin_centers), values,
                    p0=[np.max(values), np.mean(bin_centers), np.std(bin_centers), 1., 1])
func = lambda X: _dcmf_function(X, popt[0], popt[1], popt[2], popt[3], popt[4])

fig = plt.figure()
ax = fig.subplots()

M = 5
b = np.sqrt((np.exp(popt[2]**2)-1 )/M**2)

ax.plot(10**pdf[1][:100], pdf[0], marker="+", color="black", label=f"b={b:.3}")
plot_function(func, ax=ax, scatter=False, logspace=True, lims= (np.min(10**bin_centers), np.max(10**bin_centers)), color="red", linestyle="--")

ax.set_xscale("log")
ax.set_yscale("log")
fig.legend()

fig = plt.figure()
ax = fig.subplots()
def _plot_velocity(key, fit:bool=False):
    pdf = compute_pdf(sim.data[key], func=lambda x: x)
    bin_centers = pdf[1][:100]
    values = pdf[0]

    ax.plot(pdf[1][:100], pdf[0], marker="+", label=rf"{key}: $M=${np.sqrt(np.mean(np.power(sim.data[key],2)))*np.sqrt(3)/3.2591e+4:.2}")

    if fit:
        popt, _ = curve_fit(density_gaussian, bin_centers, values,
                            p0=[np.max(values), np.std(bin_centers), np.mean(bin_centers)])
        func = lambda X: density_gaussian(X, *popt)
        plot_function(func, ax=ax, scatter=False, logspace=False, lims= (np.min(bin_centers), np.max(bin_centers)), color="red", linestyle="--")

_plot_velocity('VX1')
_plot_velocity('VX2')
_plot_velocity('VX3')
fig.legend()


sim.plot_slice(slice=100,)
from POLARIScore.utils.utils import compute_mass_weighted_density, compute_volume_weighted_density
"""
plt.show()
