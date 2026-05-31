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
from POLARIScore.networks.architectures.nn_SAUNet import SizeAwareUNet
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
#sim = Simulation_DC("orionMHD_lowB_0.39_512", global_size=66.0948)
#sim = openSimulation("orionMHD_lowB_multi_", global_size=66.0948+0.12,keys=['RHO'],cache_name="orion") #offset bcs without dense cores have an offset :/
#sim.plot(plot_pdf=False, norm=LogNorm(vmin=1e21, vmax=3e24))
#sim.plot(norm=LogNorm(vmin=1e21, vmax=3e24))


#sim = Simulation_AMR("orion_MHD_lowB_AMR", global_size=66.0948, init=False)
#sim.init(init_datacubes=False)
#sim.init_datacubes(res=512, keys=['p_RHO'])
#sim.plot(norm=LogNorm(vmin=1e21, vmax=3e24), plot_pdf=False)
#sim.plot_slice()
#sim.generate_dataset(name="amr_256px", img_size=256, number=1000, size=1.*2)
#ds = getDataset("amr_256px")
#ds1, ds2 = ds.split(0.8)
#ds1.save()
#ds2.save()

#sim = Simulation_AMR("orionMHD_lowB_AMR", global_size=66.0948+0.12, init_datacubes=False)
#sim.generate_dataset(name="amr_2pc", size=4., img_size=256, number=300)
#ds2 = getDataset("amr_2pc")
#ds1_2, ds2_2 = getDataset("amr_2pc_b1"), getDataset("amr_2pc_b2")#ds2.split(0.8)
#ds1_2.save()
#ds2_2.save()

#training_ds = getDataset("amr_1pc_b1")
#validation_ds = getDataset("amr_1pc_b2")

#training_ds.merge([ds1, ds1_2], name="sa_training_set", save=True)
#validation_ds.merge([ds2, ds2_2], name="sa_validation_set", save=True)

#training_ds = getDataset("amr_256px_b1")
#validation_ds = getDataset("amr_256px_b2")


#def classic_log_mse(output, target):
#    output_phys = DATA_NORMALIZATION_VDENS_TORCH[1](output)
#    target_phys = DATA_NORMALIZATION_VDENS_TORCH[1](target)
#    output_log = torch.log(output_phys)
#    target_log = torch.log(target_phys)
#    mse = torch.mean((output_log - target_log) ** 2)
#    return mse


#trainer = Trainer(UNet, training_ds, validation_ds, "highres_unet")
trainer = load_trainer("SizeAware_Unet")
#trainer.pred_type = "v"
trainer.norms = { 
#    "cdens": DATA_NORMALIZATION_CDENS,
#    "vdens": DATA_NORMALIZATION_VDENS,
    "physize": (lambda x:x, lambda x:x)
}

#trainer.ema = True
#trainer.validation_loss_method = classic_log_mse
#trainer.ema_warmup = 30
#trainer.learning_rate = 1e-4
#trainer.network_settings["base_filters"] = 64
#trainer.network_settings["num_layers"] = 5
#trainer.network_settings["attention_layers"] = [2]
#trainer.network_settings["attention_heads"] = [8]
#trainer.network_settings["filter_function"] = "linear"
#trainer.training_random_transform = True
#trainer.optimizer_name = "Adam"
#trainer.target_names = ["vdens"]
#trainer.input_names = ["cdens"]
#trainer.auto_save = 250
#trainer.scheduler = None
#trainer.init()
#trainer.train(1000,batch_number=8,compute_validation=10,early_stopping=False)
#trainer.save()
#trainer.get_validation_error()

#trainer.plot_residuals()
#trainer.plot(save=True)
#trainer.plot_validation(save=True, number=16, number_per_row=4)



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
obs = Observation("OrionB", "column_density_map")

#obs.catalog_name = "Ntormousi & Hennebelle"
#trainer = load_trainer("DDPM", trainer_class=DDPTrainer)
#trainer.norms = {
#    "cdens": DATA_NORMALIZATION_CDENS,
#    "vdens": DATA_NORMALIZATION_VDENS,
#}
#trainer.get_validation_error()
#trainer.plot_residuals()
#3.30474
#print(obs.find_scale(3.,256,obs.distance))
#_, error = obs.predict(trainer, method="mean", repeat=0, overlap=0.5, downsample_factor=obs.find_scale(3.,256,obs.distance), nan_value=1., apply_baseline=False, kernel="uniform", save_samples=None, skip_using_saved_samples=False, only_error=False, patch_size=(256,256))
#obs.save("_saunet")
#obs.prediction = compute_mass_weighted_density(sim.data['RHO'], axis=AXIS)
obs.load("_ddpm")
fig, ax = obs.plot_cores_error(show_errors=True,label="none",correction=None, log_average=30)
obs.plot_cores_error(show_errors=True,ax=ax,label="fixed",correction="fixed", log_average=30)
obs.plot_cores_error(show_errors=True,ax=ax,label="blurred",correction="blurred", log_average=30)


#obs.load_error("cINN")
#obs.prediction = obs.rectify_error_baseline()
#_, ax =obs.plot_cores_error(correction="fixed", label="fixed")
#obs.load("_ddpm_2")
#_, ax =obs.plot_cores_error(ax=ax, correction="blurred", label="blurred")
#obs.plot(obs.prediction, norm=LogNorm(1e2, 3e5))
#_, ax =obs.plot_cores_error(correction="blurred", label="blurred")
#obs.convolved_data = np.load(os.path.join(CACHES_FOLDER,"convolved_orionb.npy"))
#obs.plot(obs.convolved_data, norm=None)
#obs.plot_dcmf(correction="fixed")
#obs.plot_cores_error(ax=ax, correction="fixed", label="fixed")

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


plt.show()
breakpoint()