
from POLARIScore.utils.utils import compute_mass_weighted_density
from POLARIScore.objects.SimulationArray import SimulationArray
from POLARIScore.objects.Dataset import getDataset

sim_names=[
    "turb_sim_A","turb_sim_B","turb_sim_C","turb_sim_E"
]
enable_dataset_gen = True

if enable_dataset_gen:
    for name in sim_names:
        sims = SimulationArray(simulations=[] ,name=name)

        sims.generate_dataset(name=name,what_to_compute={"cospectra":"gaussians", "vdens":compute_mass_weighted_density}, number=100, axes=[0,2])
        #validation dataset
        sims.generate_dataset(name=name+"_v",what_to_compute={"cospectra":"gaussians", "vdens":compute_mass_weighted_density}, number=100, axes=[1])
        
    training_datasets = [getDataset("batch_"+name) for name in sim_names]
    validation_datasets = [getDataset("batch_"+name+"_v") for name in sim_names]
    training_datasets[0].merge(training_datasets[1:], delete=False, name="idefix_training_gaussians", save=True)
    validation_datasets[0].merge(validation_datasets[1:], delete=False, name="idefix_validation_gaussians", save=True)
training_ds = getDataset("batch_idefix_training_gaussians")
validation_ds = getDataset("batch_idefix_validation_gaussians")

"""
from POLARIScore.networks.Trainer import Trainer, load_trainer
from POLARIScore.networks.architectures.nn_SC_1 import SC_1
from torch import nn
trainer = Trainer(SC_1, training_set=training_ds, validation_set=validation_ds, model_name="MultiNet_ID_13CO_GAUSSIANS")
#trainer = load_trainer("cached_model")
trainer.validation_set = validation_ds
trainer.training_set = training_ds
trainer.validation_loss_method = nn.MSELoss()
trainer.learning_rate = 1e-3
trainer.network_settings["base_filters"] = 64
trainer.network_settings["num_layers"] = 3
trainer.network_settings["gaussian_features"] = 10
trainer.input_names = ["cdens","cospectra"]
trainer.target_names = ["vdens"]
trainer.training_random_transform = True
trainer.init()
trainer.train(750, batch_number=8, compute_validation=10,early_stopping=False)
trainer.save()
trainer.plot(save=False)
trainer.plot_validation(save=False)
trainer.model.plot_channel_weights(channel_names=trainer.input_names, cmap='viridis')
"""