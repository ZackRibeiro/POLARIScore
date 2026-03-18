from POLARIScore.objects.Dataset import Dataset, getDataset
from POLARIScore.networks.DDPTrainer import DDPTrainer
from POLARIScore.networks.architectures.nn_DDPM import DDPMUnet
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from POLARIScore.objects.Simulation_DC import Simulation_DC
from typing import List
from POLARIScore.objects.SpectrumMap import getSimulationSpectra

sim_names = ["sim_256_A_5"]#,"turb_sim_B","turb_sim_C"]
prefix = "no_env"
GENERATE_DATASET = False
number_per_face = int(256*256*.01)
if GENERATE_DATASET:
    datasets_to_merge:List['Dataset'] = []
    for s in sim_names:
        sim = Simulation_DC(s)
        maps = getSimulationSpectra(simulation=sim, axes=[0,1,2])
        for i, m in enumerate(maps):
            ds = getDataset("batch_spectra_"+prefix+"_"+str(s)+"_"+str(i))
            if ds is not None:
                datasets_to_merge.append(ds)
                continue
            m.generate_dataset(name="spectra_"+prefix+"_"+str(s)+"_"+str(i), number=number_per_face, environment=0, what_to_compute={"gaussians": False})
            datasets_to_merge.append(getDataset("batch_spectra_"+prefix+"_"+str(s)+"_"+str(i)))
    datasets_to_merge[0].merge(datasets_to_merge[1:], delete=False, save=True, name="spectra_"+prefix)
    ds = getDataset("batch_spectra_"+prefix)
    ds1, ds2 = ds.split(0.8)
    ds1.save()
    ds2.save()
training_set, validation_set = getDataset("batch_spectra_"+prefix+"_b1"), getDataset("batch_spectra_"+prefix+"_b2")

def _accuracy(output, target):
    if isinstance(output, (list, tuple)):
        output = output[0]
    if isinstance(target, (list, tuple)):
        target = target[0]

    corrects = (torch.abs(output-target) <= 0.05)
    acc = corrects.float().mean(dim=(1, 2))
    acc = torch.mean(acc)

    return acc

def _loss(output, target):
    if isinstance(output, (list, tuple)):
        output = output[0]
    if isinstance(target, (list, tuple)):
        target = target[0]
    return F.mse_loss(output, target)

trainer = DDPTrainer(DDPMUnet, training_set, validation_set, model_name="Denoising_DDPM", timesteps=500, beta_schedule='cosine')
trainer.pred_type = "x0"
trainer.dimension = 1
trainer.norms = {
    "spectrum": (lambda x:x, lambda x:x),
    "noisy_spectrum": (lambda x:x, lambda x:x)
}

trainer.ema = True
trainer.loss_method = _loss
trainer.validation_loss_method = _accuracy
trainer.ema_warmup = 50
trainer.network_settings["dim"] = 1
trainer.network_settings["base_filters"] = 64
trainer.network_settings["num_layers"] = 4
trainer.network_settings["attention_layers"] = [3]
trainer.network_settings["attention_heads"] = [8]
trainer.training_random_transform = False
trainer.optimizer_name = "Adam"
trainer.target_names = ["spectrum"]
trainer.input_names = ["noisy_spectrum"]
trainer.scheduler = None
trainer.init()
trainer.train(100, batch_number=256, compute_validation=10, early_stopping=False)
trainer.save()
trainer.plot_validation(inter=(0, 16))


plt.show()