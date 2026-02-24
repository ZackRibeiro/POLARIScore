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
from POLARIScore.objects.Dataset import getDataset

sim_names=[
    "turb_sim_A","turb_sim_B","turb_sim_C","turb_sim_E"
]
spectra_dim = 3
enable_dataset_gen = False

"""
if enable_dataset_gen:
    for name in sim_names:
        sims = SimulationArray(simulations=[] ,name=name)

        sims.generate_dataset(name=name,what_to_compute={"cospectra":"pca", "vdens":compute_mass_weighted_density}, number=100, axes=[0,1])
        ds = getDataset("batch_"+name)
        ds.downsample(channel_names=["cospectra"], target_sizes=spectra_dim, methods="first", replace=True)
        ds.transform(channel_names="cospectra", method="split")

        #validation dataset
        sims.generate_dataset(name=name+"_v",what_to_compute={"cospectra":"pca", "vdens":compute_mass_weighted_density}, number=100, axes=[2])
        ds = getDataset("batch_"+name+"_v")
        ds.downsample(channel_names=["cospectra"], target_sizes=spectra_dim, methods="first", replace=True)
        ds.transform(channel_names="cospectra", method="split")
        
    training_datasets = [getDataset("batch_"+name) for name in sim_names]
    validation_datasets = [getDataset("batch_"+name+"_v") for name in sim_names]
    training_datasets[0].merge(training_datasets[1:], delete=False, name="idefix_training_"+str(spectra_dim), save=True)
    validation_datasets[0].merge(validation_datasets[1:], delete=False, name="idefix_validation_"+str(spectra_dim), save=True)
training_ds = getDataset("batch_idefix_training_"+str(spectra_dim))
validation_ds = getDataset("batch_idefix_validation_"+str(spectra_dim))


#sim = Simulation_DC("orionMHD_lowB_0.39_512", global_size=66.0948)
#sim.generate_dataset(name="orion_training",what_to_compute={"cospectra":"pca", "vdens":compute_mass_weighted_density}, number=100, axes=[0,1])
#sim.generate_dataset(name="orion_validation",what_to_compute={"cospectra":"pca", "vdens":compute_mass_weighted_density}, number=100, axes=[2])
training_ds = getDataset("batch_orion_training")
validation_ds = getDataset("batch_orion_validation")
#training_ds.downsample(channel_names=["cospectra"], target_sizes=spectra_dim, methods="first", replace=True)
#training_ds.transform(channel_names="cospectra", method="split")
#validation_ds.downsample(channel_names=["cospectra"], target_sizes=spectra_dim, methods="first", replace=True)
#validation_ds.transform(channel_names="cospectra", method="split")

from POLARIScore.networks.Trainer import Trainer, load_trainer
from POLARIScore.networks.architectures.nn_MultiNet import MultiNet
from POLARIScore.networks.architectures.nn_UNet import UNet
from torch import nn
trainer = Trainer(MultiNet, training_set=training_ds, validation_set=validation_ds, model_name="MultiNet_ID_13CO_PCA"+str(spectra_dim))
#trainer = load_trainer("cached_model")
trainer.validation_set = validation_ds
trainer.training_set = training_ds
trainer.validation_loss_method = nn.MSELoss()
trainer.learning_rate = 1e-2
trainer.network_settings["base_filters"] = 32
trainer.network_settings["branch_filters"] = 32
trainer.network_settings["num_layers"] = 3
trainer.network_settings["channel_dimensions"]=[2 for _ in range(spectra_dim+1)]
trainer.input_names = ["cdens",*["cospectra"+str(i) for i in range(spectra_dim)]]
trainer.target_names = ["vdens"]
trainer.network_settings["channel_modes"] = [None for _ in range(spectra_dim+1)]
trainer.training_random_transform = False
trainer.init()
trainer.train(750, batch_number=8, compute_validation=10,early_stopping=False)
trainer.save()
trainer.plot(save=False)
trainer.plot_validation(save=False)
trainer.model.plot_channel_weights(channel_names=trainer.input_names, cmap='viridis')
"""

sim = Simulation_DC("turb_sim_C")
from POLARIScore.objects.SpectrumMap import SpectrumMap, getSimulationSpectra
maps = getSimulationSpectra(simulation=sim)
maps[0].plot()
#map = maps[0]
#pca = map.pca(plot=True)



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
