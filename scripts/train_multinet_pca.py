
import matplotlib.pyplot as plt
from POLARIScore.utils.utils import compute_mass_weighted_density
from POLARIScore.objects.SimulationArray import SimulationArray
from POLARIScore.objects.Dataset import getDataset
import torch


sim_names=[
    "turb_sim_A","turb_sim_B","turb_sim_C","turb_sim_E"
]
spectra_dim = 3
enable_dataset_gen = False

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
training_ds = getDataset("batch_idefix_training_"+str(15))
validation_ds = getDataset("batch_idefix_validation_"+str(15))


#sim = Simulation_DC("orionMHD_lowB_0.39_512", global_size=66.0948)
#sim.generate_dataset(name="orion_training",what_to_compute={"cospectra":"pca", "vdens":compute_mass_weighted_density}, number=100, axes=[0,1])
#sim.generate_dataset(name="orion_validation",what_to_compute={"cospectra":"pca", "vdens":compute_mass_weighted_density}, number=100, axes=[2])
#training_ds = getDataset("batch_orion_training")
#validation_ds = getDataset("batch_orion_validation")
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
trainer.learning_rate = 1e-3
trainer.network_settings["base_filters"] = 64
trainer.network_settings["branch_filters"] = 32
trainer.network_settings["num_layers"] = 4
#trainer.network_settings["channel_dimensions"]=[2 for _ in range(spectra_dim+1)]
trainer.network_settings["channel_dimensions"] = [2,2]
#trainer.input_names = ["cdens",*["cospectra"+str(i) for i in range(spectra_dim)]]
trainer.input_names = ["cdens","cospectra"]
trainer.target_names = ["vdens"]
trainer.network_settings["channel_inchannels"] = [1, 15]
trainer.network_settings["channel_modes"] = [None, None]
trainer.ema = True
trainer.ema_warmup = 2000
#trainer.network_settings["channel_modes"] = [None for _ in range(spectra_dim+1)]
trainer.training_random_transform = True
trainer.init()
#trainer.scheduler = torch.optim.lr_scheduler.StepLR(trainer.optimizer, 50, 0.1)
trainer.train(1000, batch_number=8, compute_validation=10,early_stopping=False)
trainer.save()
trainer.plot(save=False)
trainer.plot_validation(save=False)
#trainer.model.plot_channel_weights(channel_names=trainer.input_names, cmap='viridis')

plt.show()