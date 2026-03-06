from POLARIScore.utils.vtk_io import readVTKCart
from POLARIScore.utils import compute_pdf
from POLARIScore.config import DATA_FOLDER
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib
from scipy.optimize import curve_fit

from POLARIScore.objects.Simulation_DC import Simulation_DC
from POLARIScore.utils.sim_utils import init_idefix,init_ramses
from POLARIScore.utils.utils import compute_mass_weighted_density

from POLARIScore.objects.SimulationArray import SimulationArray
from POLARIScore.objects.Dataset import Dataset, getDataset
from torch import nn

sim = Simulation_DC("turb_sim_C")
from POLARIScore.objects.SpectrumMap import SpectrumMap, getSimulationSpectra
from POLARIScore.objects.Spectrum import Spectrum
maps = getSimulationSpectra(simulation=sim, axes=[0])
#maps[0].plot(fit='dendrogram')
#map = maps[0]
#pca = map.pca(plot=True)
#sim.plot_pdf(offset_method="none", what="cdens")

#ds = maps[0].generate_dataset(name="spectra_C", number=128*10)
ds = getDataset("batch_spectra_C")
def _plot_spectrum(ds:'Dataset',ds_index:int=0):
    data = ds.get(ds_index)
    env_spectra = data[ds.get_element_index("noisy_spectrum")]
    spectrum = env_spectra[len(env_spectra)//2][len(env_spectra)//2]
    spect = Spectrum(spectrum, name="test")
    spect.X = data[ds.get_element_index("channels")]
    gauss_params = data[ds.get_element_index("gaussians") ]
    spect.fit_settings = None , {'params':gauss_params, 'N':len(gauss_params)//3}
    spect.plot(show_fit=True, show_dendrogram=False)

training_set, validation_set = ds.split(0.9)
from POLARIScore.networks.Trainer import Trainer
from POLARIScore.networks.architectures.nn_SpectraNetwork import SpectraNetwork
trainer = Trainer(network=SpectraNetwork, training_set=training_set, validation_set=validation_set, model_name="Spectral_Fit")
trainer.network_settings['num_layers']=3
trainer.network_settings['out_features']=10*3
trainer.network_settings['base_filters']=16
trainer.network_settings['environment_dim']=3
trainer.network_settings['spectra_dim']=128

trainer.norms = { 
    "channels": (lambda x:x,lambda x:x),
}


import torch
import torch.nn.functional as F
def _gaussian(x, A, mu, sigma):
    return torch.abs(A) * torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2))


def _gaussian_sum(x, params, N):
    y = torch.zeros_like(x)

    for i in range(N):
        A = params[:,:,3 * i]
        mu = params[:,:,3 * i + 1]
        sigma = params[:,:,3 * i + 2]

        gaussian = _gaussian(x, A, mu, sigma)
        gaussian = torch.nan_to_num(gaussian, nan=0)
        y = y + gaussian

    return y


def _chi_squared(params, x, y_true, N):
    y_model = _gaussian_sum(x, params, N)
    return torch.sum((y_true - y_model) ** 2 / (y_model + 1e-8))

def spectrum_loss(output, target):
    """
    output: predicted gaussian parameters [A1, mu1, sigma1, ...]
    target: true gaussian parameters
    """


    channels = output[1]

    mse = F.mse_loss(output[0], target)

    pred_params = torch.exp(output[0])-1
    true_params = torch.exp(target)-1
    y_true = _gaussian_sum(channels, true_params, 10)
    chisq = torch.mean(_chi_squared(pred_params, channels, y_true, 10))

    loss = mse

    return loss

trainer.optimizer_name = "SGD"
trainer.learning_rate = 1e-3
trainer.init()
trainer.loss_method = spectrum_loss
trainer.validation_loss_method = spectrum_loss

trainer.training_random_transform = False
trainer.input_names = ["noisy_spectrum","snr","channels"]
trainer.target_names = ["gaussians"]
trainer.train(1000, batch_number=128, early_stopping=True)

val_batch = trainer.get_prediction_batch()
print(val_batch)

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
