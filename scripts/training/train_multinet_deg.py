
import matplotlib.pyplot as plt
from POLARIScore.utils.utils import compute_mass_weighted_density
from POLARIScore.objects.SimulationArray import SimulationArray
from POLARIScore.objects.Dataset import getDataset
import torch



sim_names=[
    "turb_sim_A","turb_sim_B","turb_sim_C","turb_sim_E"
]
enable_dataset_gen = False

if enable_dataset_gen:
    for name in sim_names:
        sims = SimulationArray(simulations=[] ,name=name)

        density_methods = {"skewness":compute_density_pdf_skewness,"kurtosis":compute_density_pdf_kurtosis}
        sims.generate_dataset(name=name,what_to_compute={"cospectra":True, "vdens":compute_mass_weighted_density, "density_methods":density_methods}, number=100, axes=[0,2])
        #validation dataset
        sims.generate_dataset(name=name+"_v",what_to_compute={"cospectra":True, "vdens":compute_mass_weighted_density, "density_methods": density_methods}, number=100, axes=[1])
        
    training_datasets = [getDataset("batch_"+name) for name in sim_names]
    validation_datasets = [getDataset("batch_"+name+"_v") for name in sim_names]
    training_datasets[0].merge(training_datasets[1:], delete=True, name="idefix_training_13CO", save=True)
    validation_datasets[0].merge(validation_datasets[1:], delete=True, name="idefix_validation_13CO", save=True)
training_ds = getDataset("batch_idefix_training_13CO")
validation_ds = getDataset("batch_idefix_validation_13CO")

from POLARIScore.networks.Trainer import Trainer, load_trainer, plot_models_residuals_extended
from POLARIScore.networks.architectures.nn_MultiNet import MultiNet
from POLARIScore.networks.architectures.nn_UNet import UNet
from torch import nn



trainer = Trainer(MultiNet, training_set=training_ds, validation_set=validation_ds, model_name="MultiNet_13CO")
#trainer = load_trainer("cached_model")
trainer.validation_set = validation_ds
trainer.training_set = training_ds
trainer.validation_loss_method = nn.MSELoss()
trainer.learning_rate = 1e-3
trainer.network_settings["base_filters"] = 32
trainer.network_settings["branch_filters"] = 16
trainer.network_settings["num_layers"] = 4
#trainer.network_settings["channel_dimensions"]=[2 for _ in range(spectra_dim+1)]
trainer.network_settings["channel_dimensions"] = [2]
#trainer.input_names = ["cdens",*["cospectra"+str(i) for i in range(spectra_dim)]]
trainer.input_names = ["cospectra"]
trainer.target_names = ["vdens"]
trainer.network_settings["channel_inchannels"] = [15]
trainer.network_settings["channel_modes"] = [None]
#trainer.ema = True
#trainer.ema_warmup = 2000
#trainer.network_settings["channel_modes"] = [None for _ in range(spectra_dim+1)]
trainer.training_random_transform = True
trainer.init()
#trainer.scheduler = torch.optim.lr_scheduler.StepLR(trainer.optimizer, 100, 0.1)
trainer.train(500, batch_number=8, compute_validation=10,early_stopping=True)
trainer.save()
trainer.plot_validation()
trainer.plot()

#trainer = load_trainer("MultiNet_ID_13CO_PCA"+str(spectra_dim))
#trainer_wout_co = load_trainer("MultiNet_ID_wout_13CO_PCA"+str(spectra_dim))
#trainers = [trainer, trainer_wout_co]
#for t in trainers:
#    t.plot_validation()
#plot_models_residuals_extended(trainers=trainers)

#trainer.model.plot_channel_weights(channel_names=trainer.input_names, cmap='viridis')

plt.show()