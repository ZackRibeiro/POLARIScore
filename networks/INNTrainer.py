from .Trainer import Trainer, load_trainer
import torch
import math
from POLARIScore.networks.architectures.nn_cINN import cINN
import torch.nn as nn

class INNTrainer(Trainer):
    """
    Extension of Trainer class to train and use Invertible Neural Networks.
    """
    def __init__(self, **kwargs):
        """To know args and variables, please refers to '~.Trainer.__init__'"""
        if kwargs["network"] is None:
            kwargs["network"] = cINN
        super(INNTrainer, self).__init__(**kwargs)
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
        output = model(target,input) #target is true data, input is condition data
        return output
    
    def _infer_model(self, model, input):
        z = torch.randn(model.z_shape, device=input.device).unsqueeze(0)
        output = model.inverse(z, input)
        return output
    
    @staticmethod
    def load(model_name, load_model=True):
        return load_trainer(model_name, load_model, trainer_class=INNTrainer)