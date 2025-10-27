from .Trainer import Trainer, load_trainer
import torch
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
        nll = 0.5 * torch.sum(z ** 2, dim=1) - log_det_J

        # normalization term
        const = 0.5 * z.size(1) * math.log(2 * math.pi)
        nll += const

        loss = torch.mean(nll)
        return loss

    def _train_model(self, model, input, target):
        if(type(input) is list):
            input = input[0]
        if(type(target) is list):
            target = target[0]
        output = model(target,input) #target is true data, input is condition data
        return output
    
    def _infer_model(self, model, input):
        if(type(input) is list):
            input = input[0]
        B,_,_,_ = input.shape
        C, H, W = model.z_shape
        z = torch.randn((B, C, H, W), device=input.device)
        output = model.inverse(z, input)
        return output
    
    @staticmethod
    def load(model_name, load_model=True):
        return load_trainer(model_name, load_model, trainer_class=INNTrainer)
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from POLARIScore.objects.Dataset import getDataset
    from POLARIScore.config import DATA_NORMALIZATION_CDENS, DATA_NORMALIZATION_VDENS
    ds1 = getDataset("batch_training_32px")
    ds2 = getDataset("batch_validation_32px")


    trainer = INNTrainer(cINN, ds1, ds2, model_name="cINN")
    #trainer = INNTrainer.load("cINN")
    #trainer.norms = {
    #    "cdens": DATA_NORMALIZATION_CDENS,
    #    "vdens": DATA_NORMALIZATION_VDENS,
    #}
    trainer.learning_rate = 1e-3
    trainer.training_set = ds1
    trainer.validation_set = ds2
    trainer.network_settings["img_dim"] = 32
    trainer.network_settings["base_filters"] = 32
    trainer.network_settings["num_layers"] = 4
    trainer.training_random_transform = True
    trainer.optimizer_name = "Adam"
    trainer.target_names = ["vdens"]
    trainer.input_names = ["cdens"]
    trainer.init()
    trainer.train(500,batch_number=256,compute_validation=10,early_stopping=False)
    trainer.save()
    trainer.plot(save=True)
    trainer.plot_validation(save=True)

    plt.show()