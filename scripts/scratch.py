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

#sim = SimulationArray(name="turb_sim_A")
sim = SimulationArray(simulations=[Simulation_DC("turb_sim_A"),Simulation_DC("adastra_512")], indexes=[0,1])
from POLARIScore.objects.SpectrumMap import SpectrumMap, getSimulationSpectra
from POLARIScore.objects.Spectrum import Spectrum
sim.plot(plot_method=Simulation_DC.plot_pdf, what="vel", colors="viridis", drawstyle=None)

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

from POLARIScore.networks.Trainer import Trainer
from POLARIScore.networks.architectures.nn_SpectraNetwork import SpectraNetwork
trainer = Trainer(network=SpectraNetwork, training_set=training_set, validation_set=validation_set, model_name="Spectral_Fit")
trainer.network_settings['num_layers']=3
trainer.network_settings['out_features']=10*3
trainer.network_settings['base_filters']=32
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
        A = amps[:,:,i]
        mu = means[:,:,i]
        sigma = sigmas[:,:,i]

        gaussian = _gaussian(x, A, mu, sigma)
        y = y + gaussian

    return y

def spectrum_loss(output, target):

    means = target[1]
    amps = target[0]
    sigmas = target[2]

    y_true = _gaussian_sum(amps, means, sigmas, 10)
    y_predict = _gaussian_sum(output[0], output[1], output[2], 10)
    
    loss = F.mse_loss(y_predict, y_true)

    return loss

#torch.autograd.set_detect_anomaly(True)
trainer.optimizer_name = "SGD"
trainer.learning_rate = 1e-3
trainer.init()
trainer.loss_method = spectrum_loss
trainer.validation_loss_method = spectrum_loss

trainer.training_random_transform = False
trainer.input_names = ["noisy_spectrum","snr", "channels"]
trainer.target_names = ["gaussians_amplitudes","gaussians_means","gaussians_sigmas"]
trainer.train(1000, compute_validation=1, batch_number=512, early_stopping=False)

val_batch = trainer.get_prediction_batch()
print(val_batch)
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
