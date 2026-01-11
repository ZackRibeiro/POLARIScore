from .Trainer import Trainer, load_trainer
import torch
from torch.optim.lr_scheduler import StepLR
import math
from POLARIScore.networks.architectures.nn_cINN import cINN
import torch.nn as nn

class INNTrainer(Trainer):
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
    
    
    @staticmethod
    def load(model_name, load_model=True):
        return load_trainer(model_name, load_model, trainer_class=INNTrainer)
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from POLARIScore.objects.Dataset import getDataset
    from POLARIScore.config import DATA_NORMALIZATION_CDENS, DATA_NORMALIZATION_VDENS, DATA_NORMALIZATION_CDENS_TORCH, DATA_NORMALIZATION_VDENS_TORCH
    #ds1 = getDataset("batch_training")
    #ds2 = getDataset("batch_validation")
    ds = getDataset("batch_highres_2")
    ds1, ds2 = ds.split(0.7)

    def classic_log_mse(output, target):
        output_phys = DATA_NORMALIZATION_VDENS_TORCH[1](output)
        target_phys = DATA_NORMALIZATION_VDENS_TORCH[1](target)
        output_log = torch.log(output_phys)
        target_log = torch.log(target_phys)
        mse = torch.mean((output_log - target_log) ** 2)
        return mse


    #trainer = INNTrainer(cINN, ds1, ds2, model_name="Fixed_cINN")
    trainer = load_trainer("cINN", trainer_class=INNTrainer)
    trainer.norms = {
        "cdens": DATA_NORMALIZATION_CDENS,
        "vdens": DATA_NORMALIZATION_VDENS,
    }
    #ds = getDataset("batch_highres_2")
    #ds1, ds2 = ds.split(0.7)
    #trainer.validation_set = ds2
    #trainer.get_validation_error()

    trainer.ema = True
    trainer.validation_loss_method = classic_log_mse
    trainer.ema_warmup = 100
    trainer.learning_rate = 1e-3
    trainer.training_set = ds1
    trainer.validation_set = ds2
    #trainer.network_settings["img_dim"] = 128
    #trainer.network_settings["base_filters"] = 48
    #trainer.network_settings["num_layers"] = 4
    #trainer.network_settings["coupling_block_per_layer"] = 3
    trainer.training_random_transform = True
    trainer.optimizer_name = "Adam"
    trainer.target_names = ["vdens"]
    trainer.input_names = ["cdens"]
    trainer.auto_save = 500
    trainer.scheduler = None#StepLR(trainer.optimizer, 250, 0.1)
    #trainer.init()

    #unet_encoder = load_trainer("UNet").model.encoders
    #trainer.model.encoder.encoders = unet_encoder

    trainer.train(1500,batch_number=8,compute_validation=10,early_stopping=False)
    trainer.save()
    trainer.plot(save=True)
    trainer.plot_validation(save=True, number=8, number_per_row=4)
    trainer.get_validation_error()

    plt.show()