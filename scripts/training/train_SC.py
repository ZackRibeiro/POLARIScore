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

        sims.generate_dataset(name=name,what_to_compute={"cospectra":True, "vdens":compute_mass_weighted_density}, number=100, axes=[0,2])
        #validation dataset
        sims.generate_dataset(name=name+"_v",what_to_compute={"cospectra":True, "vdens":compute_mass_weighted_density}, number=100, axes=[1])
        
    training_datasets = [getDataset("batch_"+name) for name in sim_names]
    validation_datasets = [getDataset("batch_"+name+"_v") for name in sim_names]
    training_datasets[0].merge(training_datasets[1:], delete=True, name="idefix_training_13CO", save=True)
    validation_datasets[0].merge(validation_datasets[1:], delete=True, name="idefix_validation_13CO", save=True)
training_ds = getDataset("batch_idefix_training_13CO")
validation_ds = getDataset("batch_idefix_validation_13CO")

from POLARIScore.networks.Trainer import Trainer, load_trainer, plot_models_residuals_extended
from POLARIScore.networks.architectures.nn_SC_2 import SC_2
from torch import nn

#trainer = Trainer(SC_2, training_set=training_ds, validation_set=validation_ds, model_name="SC_2")
trainer = load_trainer("cached_model")
trainer.validation_set = validation_ds
trainer.training_set = training_ds
trainer.validation_loss_method = nn.MSELoss()
trainer.learning_rate = 1e-4
trainer.network_settings["encoder_filters"] = 16
trainer.network_settings["latent_features"] = 16
trainer.network_settings["encoder_layers"] = 3
trainer.network_settings["hidden_features"] = 64
trainer.network_settings["spectra_dim"] = 128
trainer.input_names = ["cdens","cospectra"]
trainer.target_names = ["vdens"]

#trainer.ema = True
#trainer.ema_warmup = 2000
trainer.training_random_transform = True
#trainer.init()
#trainer.scheduler = torch.optim.lr_scheduler.StepLR(trainer.optimizer, 100, 0.1)
trainer.train(100, batch_number=2, compute_validation=10,early_stopping=False)
trainer.save()
trainer.plot_validation()
trainer.plot()

plt.show()