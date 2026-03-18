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

from POLARIScore.objects.Simulation_DC import Simulation_DC
from POLARIScore.utils.sim_utils import init_idefix,init_ramses
from POLARIScore.utils.utils import compute_mass_weighted_density

from POLARIScore.objects.SimulationArray import SimulationArray
from POLARIScore.objects.Dataset import Dataset, getDataset
from torch import nn

sim = SimulationArray(name="sim_512_A_3")
#sim = SimulationArray(simulations=[Simulation_DC("turb_sim_A"),Simulation_DC("adastra_512")], indexes=[192,512])
#sim = SimulationArray(simulations=[Simulation_DC("adastra_512_old"),Simulation_DC("adastra_512")], indexes=[0,1])

from POLARIScore.objects.SpectrumMap import SpectrumMap, getSimulationSpectra
from POLARIScore.objects.Spectrum import Spectrum
#sim.plot(plot_method=Simulation_DC.plot_pdf, what="rho", colors="viridis", drawstyle=None)
sim.plot(plot_method=Simulation_DC.plot, mode="slider", norm=LogNorm(vmin=1e3, vmax=1e6), method=compute_mass_weighted_density, label=r"$<n_H>_m$")
#sim.simulations[-1].plot_slice()

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

from POLARIScore.networks.Trainer import Trainer, load_trainer
from POLARIScore.networks.architectures.nn_SpectraNetwork import SpectraNetwork
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
