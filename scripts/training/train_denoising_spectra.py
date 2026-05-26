from POLARIScore.objects.Dataset import Dataset, getDataset
from POLARIScore.networks.Trainer import load_trainer
from POLARIScore.networks.DDPTrainer import DDPTrainer, Trainer
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
            m.generate_dataset(name="spectra_"+prefix+"_"+str(s)+"_"+str(i), number=number_per_face, environment=0, what_to_compute={"gaussians": False}
                               , snr=[10,10])
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

    diff = torch.abs(output - target)
    acc = torch.exp(-diff / 0.02)
    return 1 - acc.mean()

def _loss(output, target):
    if isinstance(output, (list, tuple)):
        output = output[0]
    if isinstance(target, (list, tuple)):
        target = target[0]

    #int_loss = F.mse_loss(torch.sum(output, dim=2), torch.sum(target, dim=2))

    loss = F.mse_loss(output, target)

    return loss

#trainer = DDPTrainer(DDPMUnet, training_set, validation_set, model_name="Denoising_DDPM", timesteps=500, beta_schedule='cosine')
trainer = load_trainer("Denoising_DDPM", trainer_class=DDPTrainer)
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
trainer.network_settings["base_filters"] = 32
trainer.network_settings["num_layers"] = 4
trainer.network_settings["attention_layers"] = [4]
trainer.network_settings["attention_heads"] = [8]
trainer.training_random_transform = False
trainer.optimizer_name = "Adam"
trainer.target_names = ["spectrum"]
trainer.input_names = ["noisy_spectrum"]
trainer.scheduler = None
#trainer.init()
#trainer.train(50, batch_number=256, compute_validation=10, early_stopping=False)
#trainer.save()
trainer.get_prediction_batch(batch_number=1)
#trainer.plot_validation(inter=(0, 8),number_per_row=4)
#trainer.plot_losses(log10=False)

inter =(0,128)
pred_batch = trainer.get_prediction_batch()[inter[0]:inter[1]]

from sklearn.decomposition import PCA
import numpy as np

def pca_denoise(batch_spectra, n_components=20):
    """
    batch_spectra: torch.Tensor (B, C, L) or (B, L)
    """
    if isinstance(batch_spectra, torch.Tensor):
        data = batch_spectra.detach().cpu().numpy()
    else:
        data = batch_spectra

    if data.ndim == 3:
        B, C, L = data.shape
        data = data.reshape(B, C * L)
    elif data.ndim == 2:
        B, L = data.shape
    else:
        raise ValueError("Unexpected shape")

    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(data)
    reconstructed = pca.inverse_transform(transformed)

    if 'C' in locals():
        reconstructed = reconstructed.reshape(B, C, L)

    return torch.tensor(reconstructed, dtype=torch.float32)
noise_spectra = torch.stack([torch.tensor(p[0]) for p in pred_batch])
pred_spectra = torch.stack([torch.tensor(p[1]) for p in pred_batch])
pca_pred = pca_denoise(noise_spectra, n_components=10)

from POLARIScore.objects.Spectrum import Spectrum
def _plot_spectrum(index:int=0):
    ds = trainer.validation_set
    data = ds.get(list(ds.batch.keys())[index+inter[0]])
    spectrum = pred_batch[index][0]
    spect = Spectrum(spectrum, name="test")
    spect.X = data[ds.get_element_index("channels")]
    _, ax =spect.plot(show_fit=False, show_dendrogram=False, label="noisy spectrum (input, snr=10)")
    spect.spectrum = data[ds.get_element_index("spectrum")]
    spect.plot(ax=ax, color="red", show_dendrogram=False, label="true spectrum")
    spect.spectrum = pred_batch[index][1]
    spect.plot(ax=ax, color="green", show_dendrogram=False, label="DDPM prediction")
    spect.spectrum = pca_pred[index]
    spect.plot(ax=ax, color="purple", show_dendrogram=False, label="PCA")

_plot_spectrum(0)
_plot_spectrum(1)
_plot_spectrum(2)
_plot_spectrum(3)
_plot_spectrum(4)


plt.show()