from POLARIScore.config import *
import os
from POLARIScore.utils.utils import *
from POLARIScore.objects.Dataset import Dataset
from POLARIScore.objects.Simulation_DC import openSimulation
from POLARIScore.networks.DDPTrainer import DDPTrainer
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize

trainer:DDPTrainer = DDPTrainer.load("DDPM")
#trainer.set_scheduler(timesteps=1000, beta_schedule="quadratic", beta_start=1e-4, beta_end=0.2)
trainer.inference_timestep =  20

INDEX = 2

new_ds = trainer.validation_set.clone()

indexes_to_remove = []
for i, k in enumerate(list(new_ds.batch.keys())):
    if i != INDEX:
        indexes_to_remove.append(k)
new_ds.remove(indexes_to_remove)
trainer.validation_set = new_ds

trainer.save_intermediaries_steps = False
trainer.norms = { 
    "cdens": DATA_NORMALIZATION_CDENS,
    "vdens": DATA_NORMALIZATION_VDENS,
}


#trainer.plot_intermediaries_steps()
#trainer.plot_pdf_trajectory()
trainer.plot_sampling(1000, bins=None)
#trainer.plot_degeneracy(100, bins=10)

new_ds.delete()
plt.show()

"""
in_tensor = trainer.validation_set.get(-1)[trainer.validation_set.get_element_index("cdens")]
pred_batch = trainer.get_prediction_batch()
target_tensor, pred_tensor = pred_batch[-1]
pred_noisy_steps = trainer.intermediaries_steps #(B, H, W)
sim = openSimulation("orionMHD_lowB_multi_", global_size=66.0948+0.12,keys=['RHO'],cache_name="orion")
fig, ax = sim.plot_correlation()
FOLLOWED_PIXELS = [(64,64)]
cmap = get_cmap("plasma") 
n_steps = len(pred_noisy_steps)
norm = Normalize(vmin=0, vmax=n_steps-1)
for pixel_id, (px_x, px_y) in enumerate(FOLLOWED_PIXELS):
    value_cdens = in_tensor[px_y,px_x]
    pred_values_vdens = DATA_NORMALIZATION_VDENS[1](np.array([t[0,px_y, px_x] for t in pred_noisy_steps]))
    true_value_vdens = target_tensor[px_y, px_x]

    for t in range(n_steps - 1):

        color = cmap(norm(t))

        ax.plot([value_cdens, value_cdens],[pred_values_vdens[t], pred_values_vdens[t+1]],
            color=color,linewidth=2,alpha=0.9,
        )

        ax.scatter([value_cdens],[pred_values_vdens[t]],
            color=color,s=25,
        )

    ax.scatter([value_cdens],[pred_values_vdens[0]],
        color="blue",marker="s",s=80,label="Start" if pixel_id == 0 else None,zorder=5
    )

    ax.scatter(
        [value_cdens],[pred_values_vdens[-1]],
        color="blue",marker="X",s=100,label="Final pred" if pixel_id == 0 else None,zorder=5
    )

    ax.scatter(
        [value_cdens],[true_value_vdens],
        color="red",marker="o",s=100,label="True" if pixel_id == 0 else None,zorder=5
    )
"""