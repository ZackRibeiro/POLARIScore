from .Trainer import Trainer, load_trainer
from POLARIScore.networks.ProbabilisticTrainer import ProbabilisticTrainer
import torch
from torch.optim.lr_scheduler import StepLR
import math
from POLARIScore.networks.architectures.nn_cINN import cINN
from POLARIScore.utils.utils import printProgressBar, plot_rect_bg
import torch.nn as nn
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from typing import *
from POLARIScore.config import *
import matplotlib.axes
from matplotlib.colors import Normalize, LogNorm
from scipy.stats import lognorm

class INNTrainer(ProbabilisticTrainer):
    """
    Extension of Trainer class to train and use Invertible Neural Networks.
    """
    def __init__(self, *args, **kwargs):
        """To know args and variables, please refers to '~.Trainer.__init__'"""
        super(INNTrainer, self).__init__(*args, **kwargs)
        self.loss_method = self.max_likelihood_loss
        self.validation_loss_method = nn.MSELoss()

    @staticmethod
    def max_likelihood_loss(output, target):
        """Conditional maximum likelihood loss"""
        z, log_det_J = output
        if z.dim() > 2:
            z = z.view(z.size(0), -1)
        nll = 0.5 * torch.sum(z ** 2, dim=[1]) - (log_det_J)
        const = 0.5 * z.size(1) * math.log(2 * math.pi)
        nll += const

        loss = torch.mean(nll)
        return loss

    def _train_model(self, model, input, target):
        if(type(input) is list):
            input = input[0]
        if(type(target) is list):
            target = target[0]

        #To test if the model is indeed invertible
        #with torch.no_grad():
        #    z, _ = model(target, input)
        #    y_rec = model.inverse(z, input)  
        #    recon_mse = torch.mean((y_rec - target)**2).item()
        #    print("Reconstruction MSE:", recon_mse)
            
        output = model(target,input)
        return output
    
    def _infer_model(self, model, input):
        if(type(input) is list):
            input = input[0]
        B,_,_,_ = input.shape
        C, H, W = model.z_shape
        
        with torch.no_grad(): 
            z = torch.randn((B, C, H, W), device=input.device)
            output = model.inverse(z, input)

            return output
        
    def plot_latent_physical_plane(self, grid_n=20):
        zs = []

        input_key = self.validation_set.get_element_index(self.input_names[0])
        target_key = self.validation_set.get_element_index(self.target_names[0])

        for i in range(len(self.validation_set.batch)):

            sample = self.validation_set.get(list(self.validation_set.batch.keys())[i])

            c = torch.tensor(
                self.norms[self.input_names[0]][0](sample[input_key])
            ).float().unsqueeze(0).unsqueeze(0).cuda()

            y = torch.tensor(
                self.norms[self.target_names[0]][0](sample[target_key])
            ).float().unsqueeze(0).unsqueeze(0).cuda()

            with torch.no_grad():
                z, _ = self.model(y, c)

            zs.append(z.flatten().cpu().numpy())

        zs = np.stack(zs)

        pca = PCA(n_components=2)
        pca.fit(zs)

        z_mean = zs.mean(axis=0)


        xs = np.linspace(-1, 1, grid_n)
        ys = np.linspace(-1, 1, grid_n)

        observable = np.zeros((grid_n, grid_n))

        sample = self.validation_set.get(0)

        c = torch.tensor(self.norms[self.input_names[0]][0](sample[input_key])).float().unsqueeze(0).unsqueeze(0).cuda()

        for i,a in enumerate(xs):
            for j,b in enumerate(ys):
                printProgressBar(i*len(ys)+j, len(xs)*len(ys), "Predicting...")

                z = (z_mean+ a*pca.components_[0]+ b*pca.components_[1])
                z = torch.tensor(z).float().reshape(1,*self.model.z_shape).cuda()

                with torch.no_grad():
                    x = self.model.inverse(z, c)

                observable[j,i] = x.mean().item()

        plt.figure(figsize=(6,5))

        plt.imshow(observable,extent=[xs.min(), xs.max(), ys.min(), ys.max()],
            origin="lower",aspect="auto")

        plt.xlabel("Latent PCA 1")
        plt.ylabel("Latent PCA 2")

        plt.colorbar(label="Mean density")

        plt.tight_layout()

    @staticmethod
    def load(model_name, load_model=True):
        return load_trainer(model_name, load_model, trainer_class=INNTrainer)
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from POLARIScore.objects.Dataset import getDataset
    from POLARIScore.config import DATA_NORMALIZATION_CDENS, DATA_NORMALIZATION_VDENS, DATA_NORMALIZATION_CDENS_TORCH, DATA_NORMALIZATION_VDENS_TORCH
    ds1 = getDataset("batch_highres_2_b1")
    ds2 = getDataset("batch_highres_2_b2")
    #ds1, ds2 = ds.split(0.7)

    spectra_dim = 5

    def classic_log_mse(output, target):
        output = output[0]
        target = target[0]
        output_phys = DATA_NORMALIZATION_VDENS_TORCH[1](output)
        target_phys = DATA_NORMALIZATION_VDENS_TORCH[1](target)
        output_log = torch.log(output_phys)
        target_log = torch.log(target_phys)
        mse = torch.mean((output_log - target_log) ** 2)
        return mse

    #trainer = INNTrainer(cINN, ds1, ds2, model_name="cINN")
    trainer = load_trainer("cINN", trainer_class=INNTrainer)
    trainer.norms = {
        "cdens": DATA_NORMALIZATION_CDENS,
        "vdens": DATA_NORMALIZATION_VDENS,
    }
    trainer.ema = True
    trainer.validation_loss_method = classic_log_mse
    trainer.ema_warmup = 50
    trainer.learning_rate = 1e-4
    trainer.training_set = ds1
    trainer.validation_set = ds2
    trainer.network_settings["img_dim"] = 128
    trainer.network_settings["base_filters"] = 32
    trainer.network_settings["num_layers"] = 3
    trainer.network_settings["coupling_block_per_layer"] = 3
    #trainer.network_settings["attention_layers"] = [3,4]
    trainer.network_settings["num_encoders"] = 1#+spectra_dim
    trainer.training_random_transform = True
    trainer.optimizer_name = "Adam"
    trainer.target_names = ["vdens"]
    trainer.input_names = ["cdens"]#,*["cospectra"+str(i) for i in range(spectra_dim)]]
    trainer.auto_save = 500
    #trainer.scheduler = None#StepLR(trainer.optimizer, 250, 0.1)
    #trainer.init()

    #unet_encoder = load_trainer("UNet").model.encoders
    #trainer.model.encoder.encoders = unet_encoder

    #trainer.train(3000,batch_number=8,compute_validation=10,early_stopping=False)
    #trainer.save()
    trainer.plot(save=False)
    trainer.plot_validation(save=False, number=8, number_per_row=4)
    trainer.get_validation_error()

    plt.show()