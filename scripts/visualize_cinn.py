from POLARIScore.config import *
import os
from POLARIScore.utils.utils import *
from POLARIScore.objects.Dataset import Dataset
from POLARIScore.objects.Simulation_DC import openSimulation
from POLARIScore.networks.INNTrainer import INNTrainer
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize

trainer:INNTrainer = INNTrainer.load("cINN")
INDEX = 2
new_ds = trainer.validation_set.clone()

indexes_to_remove = []
for i, k in enumerate(list(new_ds.batch.keys())):
    if i != INDEX:
        indexes_to_remove.append(k)
new_ds.remove(indexes_to_remove)
trainer.validation_set = new_ds


trainer.norms = { 
    "cdens": DATA_NORMALIZATION_CDENS,
    "vdens": DATA_NORMALIZATION_VDENS,
}

#trainer.plot_latent_physical_plane(grid_n=128)
trainer.plot_sampling(1000, bins=None)

new_ds.delete()
plt.show()
