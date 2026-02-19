import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from POLARIScore.networks.architectures.nn_BaseModule import BaseModule
from POLARIScore.networks.architectures.nn_SAUNet import SizeAwareUNet
from POLARIScore.networks.architectures.nn_cINN import cINN
from POLARIScore.networks.architectures.nn_DDPM import DDPMUnet
from POLARIScore.utils.batch_utils import *
from POLARIScore.config import *
import uuid
from POLARIScore.networks.architectures.nn_UNet import *
from POLARIScore.networks.architectures.nn_CAUNet import ContextAwareUNet
from POLARIScore.networks.architectures.nn_MultiNet import MultiNet
from POLARIScore.networks.architectures.nn_PPV import PPV, Test
from POLARIScore.networks.architectures.nn_KNet import *
from POLARIScore.networks.utils.nn_utils import compute_batch_accuracy
from POLARIScore.utils.utils import moving_average, applyBaseline
from POLARIScore.networks.addons.ExpMA import ExponentialMovingAverage
from POLARIScore.networks.addons.EarlyStopping import EarlyStopping
import json
from POLARIScore.objects.Dataset import getDataset, Dataset
import shutil
from typing import List, Dict, Union, Tuple, Callable

NETWORK_OPTIONS = {
    "UNet" : UNet,
    "KNet" : KNet,
    "UneK": UneK,
    "MultiNet": MultiNet,
    "PPV": PPV,
    "SAUnet": SizeAwareUNet,
    "CAUNet": ContextAwareUNet,
    "JustKAN": JustKAN,
    "cINN": cINN,
    "DDPMUnet": DDPMUnet,
    "None": None
}

CONVBLOCK_OPTIONS = {
    "DoubleConvBlock": DoubleConvBlock,
    "ResConvBlock":ResConvBlock,
    "KanConvBlock":KanConvBlock,
    "ConvBlock":ConvBlock
}


class Trainer():
    """Allows training of models and experiments with them."""
    def __init__(self,network=None,training_set:Dataset=None,validation_set:Dataset=None,model_name:str=None,segmentation:bool=False,auto_save:int=0):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        """device: gpu or cpu"""

        self.auto_save:int = auto_save
        """The model is saved each 'auto_save' epochs during training."""

        self.model_name:str = model_name
        if model_name is None:
            self.model_name = str(uuid.uuid4())

        self._has_target_in_train_output = False

        LOGGER.log(f"{self.device} is used for {self.model_name}")
        
        self.network_type:str = "None"
        """Network class name used for the model"""
        if not(network is None):
            self.network_type = network.__name__
        self.learning_rate:float = 0.001
        """Not constant step size at each iteration during training while moving toward a minimum of a loss function."""
        self.training_set:Dataset = training_set
        """Dataset with the data used for training, i.e used for training/rectify the model weights."""
        self.validation_set:Dataset = validation_set
        """Dataset with the data used for validation, i.e not seen during training."""

        self.prediction_batch:Tuple[List[np.ndarray],List[np.ndarray]] = None
        """data channel 0, prediction channel 0"""

        self.network = network
        """Network class used"""
        self.network_settings: Dict ={}
        """Network settings"""

        self.target_names:Union[List[str],str] = ["vdens"]
        self.input_names:Union[List[str],str] = ["cdens"]
        self.norms:Dict[str,Tuple[Callable[[np.ndarray],np.ndarray],Callable[[np.ndarray],np.ndarray]]] = {} 
        """Normalizations, dict of tuple of function which map physical quantity to normed and invert"""

        self.segmentation:bool = segmentation
        """Segmentation mode, if true the trainer will use classical settings of a classification problem and change some parameters in compatible models
          else the goal is assumed to be a regression."""
        if self.segmentation:
            LOGGER.warn("Segmentation mode is on.")

        self.training_random_transform:bool = False

        self.model:'BaseModule' = None
        """Instance of self.network"""
        self.optimizer = None
        """Instance of an optimizer, by default Adam if optimizer_name was not changed."""
        self.optimizer_name:str = str(type(torch.optim.Adam))
        self.scheduler = None
        """Instance of a scheduler"""
        self.weight_decay:float = 0.
        self.cache_threshold:float = 2.
        """If validation error is lower that this value and if cache option is enable then the model is saved as 'cache_model'
        (and each time the validation error is lower than the previous minimum)."""

        self.ema:bool = False
        """Enable Exponential Moving Average for model weights"""
        self.ema_handler:Union[ExponentialMovingAverage,None] = None
        """Handler of Expoential Moving Average if enabled"""
        self.ema_warmup:int = 200
        """Ema warmup, number of epochs before ema begins"""

        self.early_stopping = EarlyStopping()
        
        self.baseline:Tuple[List[float],List[float]] = None
        """Baseline : (...predictions, ...residuals)"""

        if not(self.network is None):
            #self.network_settings["segmentation"] = self.segmentation
            self.model = network(**self.network_settings).to(self.device)
            if self.optimizer_name in (str(type(torch.optim.Adam)),"Adam"):
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            elif self.optimizer_name in (str(type(torch.optim.SGD)),"SGD"):
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.5)
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=50, factor=0.75, threshold=0.005)

        self.loss_method = nn.MSELoss() if not(self.segmentation) else nn.CrossEntropyLoss()
        self.validation_loss_method:Union[Callable, None] = None
        """If set to None, use the same loss method on validation and training."""

        self.training_losses:List[float] = []
        self.validation_losses:List[float] = []
        self.last_epoch:int = 0
        self.inference_time:Union[float,None] = None

    def init(self, model=None)->bool:
        """
        Init the model, use this function if you changed settings as for example self.network,self.optimizer or self.scheduler. 
        By default, when creating a Trainer instance a model is created with default settings with Adam for the optimizer.
        Args:
            model: Network instance, can be None if you want to let the code create the instance using self.network_settings.
        Returns:
            bool: If a model was created.
        """
        if(self.network is None and model is None):
            LOGGER.warn(f"Can't init model {self.model_name}, check if network is defined or model is not None.")
            return False
        self.network_type = self.network.__name__
        if model is None:
            self.model = self.network(**self.network_settings).to(self.device)
        else:
            self.model = model.to(self.device)
        if self.optimizer_name in (str(type(torch.optim.Adam)),"Adam") or "Adam" in self.optimizer_name:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_name in (str(type(torch.optim.SGD)),"SGD") or "SGD" in self.optimizer_name:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.5)
        if self.scheduler is not None:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=50, factor=0.75, threshold=0.005)
        if self.validation_loss_method is None:
            self.validation_loss_method = self.loss_method
        return True

    def _infer_model(self, model, input):
        """Used in inference"""
        return model(*input)
    
    def _train_model(self, model, input, target):
        """Used in training, e.g for INN this function is not the same as self._infer_model."""
        return model(*input)
    
    def _get_eval_model(self, epoch:Union[int, None]=None):
        if self.ema and self.ema_handler is not None and (epoch is None or epoch > self.ema_warmup):
            ema_model =self.ema_handler.copy_ema_model(self.model)
            #ema_model.eval()
            eval_model = ema_model
        else:
            eval_model = self.model
        return eval_model

    def train(self, epoch_number:int=100, batch_number:int=32, compute_validation:int=10, cache:bool=True, early_stopping:bool=True, training_mode:Literal["accumulation","normal"]="normal"):
        """
        Train the model (check trainer variables for settings)

        Args:
            epoch_number(int, default: 50): train the model for x epochs.
            batch_number(int, default: 32): How many images will be processed at a time in the GPU/CPU.
            compute_validation(int, default:10): compute validation losses each x epochs.
            cache(bool, default:True): If the validation loss is less than a previous epoch, the model will be saved in a cache.
            early_stopping(bool, default:True): Stop the training when the model isn't better in a timeframe. You can change the settings of early stopping (e.g patience and delta) by define a new ES: self.early_stopping = EarlyStopping(your_settings).
            training_mode(str, default:"normal"): If is 'accumulation': gradients are sum over mini batches, if 'normal': optimizer is applied in each mini batch.
        """
        LOGGER.log(f"Training started with {str(epoch_number)} epochs on network {self.network_type} with mini-batch of size {batch_number}, model has {sum(p.numel() for p in self.model.parameters())} parameters.")
   
        self.model.train()

        def _format_time(seconds:float)->str:
            hours, rem = divmod(seconds, 3600)
            minutes, seconds = divmod(rem, 60)
            return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

        def _random_transform(tensors_input, tensors_target):
            k = torch.randint(0, 4, (1,)).item()
            #Maybe need to change this: See if it works ! 
            tensors_input = [torch.rot90(t, k, [-2, -1]) for t in tensors_input]
            tensors_target = [torch.rot90(t, k, [-2, -1]) for t in tensors_target]

            if torch.rand(1).item() > 0.5:
                tensors_input = [torch.flip(t, [-2]) for t in tensors_input]
                tensors_target = [torch.flip(t, [-2]) for t in tensors_target]

            if torch.rand(1).item() > 0.5:
                tensors_input = [torch.flip(t, [-1]) for t in tensors_input]
                tensors_target = [torch.flip(t, [-1]) for t in tensors_target]

            return tensors_input, tensors_target


        total_epoch = self.last_epoch
        l_ep = self.last_epoch
        batch_size = len(self.training_set.batch)
        validation_batch_size = len(self.validation_set.batch)
        start_time = time.time()

        minimimum_validation_loss = self.cache_threshold

        if self.validation_loss_method is None:
            self.validation_loss_method = self.loss_method

        for epoch in range(epoch_number):
            total_epoch += 1
            epoch_loss = 0
            shuffled_indices = torch.randperm(batch_size)
            if training_mode == "accumulation":
                self.optimizer.zero_grad()
            minbatch_nbr = int(np.floor(batch_size/batch_number))
            epoch_time = time.time()
            for b in range(minbatch_nbr if minbatch_nbr > 1 else 1):
                printProgressBar(b, minbatch_nbr, length=10, prefix=f"{b}/{minbatch_nbr}")
                if training_mode == "normal":
                    self.optimizer.zero_grad()
                used_batch = self.training_set.get(indexes=shuffled_indices[b*batch_number:(b+1)*batch_number if minbatch_nbr > 1 else -1])
                if batch_number == 1:
                    used_batch = [used_batch]
                t_input, t_target = self.model.shape_batch(used_batch, self.training_set.get_element_index(self.target_names), self.training_set.get_element_index(self.input_names),
                                                          target_names=self.target_names, input_names=self.input_names, norms=self.norms, segmentation=self.segmentation)
                if not(type(t_input) is list):
                    t_input = [t_input]

                if self.training_random_transform:
                    t_input, t_target = _random_transform(t_input, t_target)
                output = self._train_model(self.model,t_input,t_target)
                target = t_target

                #In some cases the train function can also returns the target (like in DDPMs where it returns the noise added).
                if self._has_target_in_train_output:
                    output, target = output
                loss = 0
                try:
                    loss = self.loss_method(output, target)
                except:
                    for tt in range(len(target)):
                        if type(output) is list:
                            loss += self.loss_method(output[tt], target[tt])
                        else:
                            loss += self.loss_method(output, target[tt])
                if training_mode == "accumulation":
                    loss = loss / max(minbatch_nbr, 1)
                loss.backward()
                if training_mode == "normal":
                    self.optimizer.step()
                    if self.ema:
                        if total_epoch > self.ema_warmup:
                            if(self.ema_handler is None):
                                self.ema_handler = ExponentialMovingAverage()
                                self.ema_handler.register_model(self.model)
                            self.ema_handler.update(self.model)
            
                epoch_loss += loss.item()
            if training_mode == "accumulation":
                self.optimizer.step()
                if self.ema:
                    if total_epoch > self.ema_warmup:
                        if(self.ema_handler is None):
                            self.ema_handler = ExponentialMovingAverage()
                            self.ema_handler.register_model(self.model)
                        self.ema_handler.update(self.model)
            if self.scheduler is not None:
                self.scheduler.step(epoch_loss)
            if training_mode == "normal":
                epoch_loss /= max(minbatch_nbr,1)
            self.training_losses.append((total_epoch, epoch_loss))
            val_total_loss = None
            if compute_validation>0 and total_epoch % compute_validation == 0:
                minbatch_nbr = int(np.floor(validation_batch_size/batch_number))
                with torch.no_grad():
                    val_total_loss = 0
                    eval_model = self._get_eval_model(epoch=total_epoch)
                    for b in range(minbatch_nbr if minbatch_nbr > 1 else 1):
                        printProgressBar(b, minbatch_nbr, length=10, prefix=f"{b}/{minbatch_nbr}")
                        used_batch = self.validation_set.get(indexes=list(range(len(self.validation_set.batch)))[b*batch_number:(b+1)*batch_number if minbatch_nbr > 1 else -1])
                        if batch_number == 1:
                            used_batch = [used_batch]
                        v_input_tensor, v_target_tensor = self.model.shape_batch(used_batch, self.validation_set.get_element_index(self.target_names), self.validation_set.get_element_index(self.input_names),
                                                                                target_names=self.target_names, input_names=self.input_names, norms=self.norms, segmentation=self.segmentation)
                        if not(type(v_input_tensor) is list):
                            v_input_tensor = [v_input_tensor]
                        validation_output = self._infer_model(eval_model, v_input_tensor)
                        v_loss = 0
                        try:
                            v_loss = self.validation_loss_method(validation_output,v_target_tensor).item()
                        except:
                            for tt in range(len(v_target_tensor)):
                                if type(validation_output) is list:
                                    v_loss += self.validation_loss_method(validation_output[tt],v_target_tensor[tt]).item()
                                else:
                                    v_loss += self.validation_loss_method(validation_output,v_target_tensor[tt]).item()
                        val_total_loss += v_loss
                val_total_loss /= minbatch_nbr if minbatch_nbr > 0 else 1
                self.validation_losses.append((total_epoch,val_total_loss))
                if early_stopping and self.early_stopping is not None:
                    self.early_stopping(val_total_loss, epoch=total_epoch)
            if early_stopping and self.early_stopping is not None:
                if self.early_stopping.early_stop:
                    LOGGER.warn(f"Early stop at epoch: {total_epoch}.")
                    break

            if self.auto_save > 0 and total_epoch % self.auto_save == 0:
                self.last_epoch = total_epoch
                self.save()
            if cache and not(val_total_loss is None) and val_total_loss < minimimum_validation_loss:
                minimimum_validation_loss = val_total_loss
                self.last_epoch = total_epoch
                self.save(is_cache=True)

            actual_time = time.time()
            epoch_time = actual_time - epoch_time
            time_left = (actual_time-start_time) / (epoch+1) * (epoch_number-(epoch+1))
            LOGGER.print(f'Epoch {total_epoch}/{l_ep + epoch_number} | Elapsed: {_format_time(actual_time-start_time)} | Time Left: {_format_time(time_left)} | Training Loss: {epoch_loss}, Validation loss: {val_total_loss if val_total_loss else "Not computed"}', type="training", level=1, color="34m")
            
        self.last_epoch = total_epoch
        self.learning_rate = self.scheduler.get_last_lr()[0] if self.scheduler is not None else self.learning_rate

    def get_validation_error(self, n:int=100, conf_lvl=0.9, save=True):
        LOGGER.log("Computing validation errors.")
        batch = self.get_prediction_batch()
        d_target = np.array([np.log10(b[0]) for b in batch]).flatten()
        d_prediction = np.array([np.log10(b[1]) for b in batch]).flatten()
        residuals = d_prediction-d_target
        sorted_indexes = np.argsort(d_target)
        d_prediction = d_prediction[sorted_indexes]
        residuals = residuals[sorted_indexes]

        quantiles = [(1-conf_lvl)/2, 1.-(1-conf_lvl)/2, 0.5]

        bins = np.linspace(d_prediction.min(), d_prediction.max(), n+1)
        bin_indices = np.digitize(d_prediction, bins) - 1
        bin_centers = .5*(bins[:-1]+bins[1:])
        qvals = {q: np.full(n, np.nan) for q in quantiles}

        for i in range(n):
            mask = bin_indices == i
            if np.any(mask):
                res_bin = residuals[mask]
                for q in quantiles:
                    qvals[q][i] = np.quantile(res_bin, q)

        if save:
            if not(os.path.exists(MODEL_FOLDER)):
                os.mkdir(MODEL_FOLDER)

            model_path = os.path.join(MODEL_FOLDER,self.model_name.rsplit("_epoch",1)[0])
            if not(os.path.exists(model_path)):
                os.mkdir(model_path)
        
            np.save(os.path.join(model_path,"validation_error.npy"), np.array([bin_centers,*qvals.values()]))
            LOGGER.log("Validation errors saved in model folder.")

        return bin_centers, qvals

    
    def create_baseline(self,n:int=1000, force_compute:bool=False, log:bool=True)->Tuple[List[float],List[float]]:
        """
        Create a baseline using moving average method.
        Args:
            n(int): Moving average step
            force_compute(bool): Erase the previous computed baseline and compute a new one.
            log(bool): Log to the console.
        Returns:
            (List of predictions, List of residuals)
        """
        if not(force_compute) and not(self.baseline is None):
            return self.baseline
        if log:
            LOGGER.log(f"Computing baseline for model: {self.network_type}")
        batch = self.get_prediction_batch()
        d_target = np.array([np.log10(b[0]) for b in batch]).flatten()
        d_prediction = np.array([np.log10(b[1]) for b in batch]).flatten()
        residuals = d_prediction-d_target

        mask = np.isfinite(d_prediction) & np.isfinite(residuals) & np.isfinite(d_target)
        d_target = d_target[mask]
        d_prediction = d_prediction[mask]
        residuals = residuals[mask]

        sorted_indexes = np.argsort(d_target)
        d_prediction = d_prediction[sorted_indexes]
        residuals = residuals[sorted_indexes]



        mresiduals = moving_average(residuals, n=n)
        mx = moving_average(d_prediction, n=n)

        self.baseline = (mx,mresiduals)

        return self.baseline
    
    def apply_baseline(self, prediction:np.ndarray, log:bool=True)->np.ndarray:
        """
        Apply the baseline to data (prediction).
        Args:
            prediction: data to be processed
            log (bool): log output in console.
        Returns:
            modified prediction: Prediction minus residuals.
        """
        if log:
            LOGGER.log(f"Applying baseline for prediction {prediction.shape[0]}x{prediction.shape[1]}")
        if self.baseline is None:
            self.create_baseline(log=log)

        H,W = prediction.shape
        d_prediction = np.array(np.log10(prediction)).flatten()


        d_prediction = applyBaseline(self.baseline[0],self.baseline[1],d_prediction,d_prediction)
        d_prediction = d_prediction.reshape((H,W))

        d_prediction = np.exp(d_prediction * np.log(10))

        return d_prediction

    def get_prediction_batch(self,force_compute=False):
        """
        Args:
            force_compute(bool): If trainer has already a prediction batch computed then if this is True, this will be computed again.
        Returns:
            prediction_batch:[(target_img1, prediction_img1), (target_img2, prediction_img2), ...]
        """
        
        if not(self.prediction_batch is None or force_compute):
            return self.prediction_batch
        
        start_time = time.time()
        self.prediction_batch = self.predict(self.validation_set)
        end_time = time.time()

        self.inference_time = (end_time - start_time)/len(self.prediction_batch[0])

        return self.prediction_batch

    def predict(self, dataset:Dataset, batch_number:int=1)->List[Tuple[List[np.ndarray],List[np.ndarray]]]:
        """Apply the model on a dataset
        Args:
            dataset: the dataset
            batch_number(int): How many pairs of images/arrays send to the gpu and computed at the same time.
        Returns:
            List: list of (targets,outputs) where targets and outputs are lists in the dataset order, use self.target_names to know what are the quantities.
        """

        self.model.eval()

        result_batch = []

        batch_size = len(dataset.batch)
        minbatch_nbr = int(np.floor(batch_size/batch_number))
        for b in range(minbatch_nbr if minbatch_nbr > 1 else 1):
            #printProgressBar(b, minbatch_nbr, length=10, prefix=f"{b}/{minbatch_nbr}")
            used_batch = dataset.get(indexes=list(range(batch_size))[b*batch_number:(b+1)*batch_number if minbatch_nbr > 1 else -1])
            if batch_number == 1:
                used_batch = [used_batch]
            input_tensor, target_tensor = self.model.shape_batch(used_batch, dataset.get_element_index(self.target_names), dataset.get_element_index(self.input_names),
                                                                target_names=self.target_names, input_names=self.input_names, norms=self.norms, segmentation=self.segmentation)
            if not(type(input_tensor) is list):
                input_tensor = [input_tensor]
            if not(type(target_tensor) is list):
                target_tensor = [target_tensor]
            output = self._infer_model(self._get_eval_model(), input_tensor)
            target_tensor = [self.model.shape_tensor(t, reverse=True, name=self.target_names[ti], norms=self.norms, segmentation=self.segmentation) for ti,t in enumerate(target_tensor)] 
            output = output if type(output) is list else [output]
            output = [self.model.shape_tensor(o, reverse=True, name=self.target_names[oi], norms=self.norms, segmentation=self.segmentation) for oi,o in enumerate(output)]
            for i in range(len(target_tensor)):
                l1 = target_tensor[i]
                l2 = output[i]
                result_batch.append((l1,l2))

        return result_batch
    
    def predict_tensor(self, inputs:Union[np.ndarray,List[np.ndarray]], input_names:Union[str,List[str],None]=None, output_names:Union[str,List[str],None]=None, return_tensor:bool=False):
        """
        Apply a model to inputs
        Args:
            inputs: List of arrays for example [col_dens,spectra]
            input_names: if not None, use the names to find a normalization if this is defined
            output_names: if not None, use the names to find a normalization if this is defined
        Returns:
            List of output, for example [vol_dens]
        """
        
        self.model.eval()

        inputs = inputs if type(inputs) is list else [inputs]
        input_names = input_names if type(input_names) is list else [input_names]
        output_names = output_names if type(output_names) is list else [output_names]
        input_tensors = [self.model.shape_tensor(inputs[i], name=input_names[i] if input_names else None, norms=self.norms) for i in range(len(inputs))]
        outputs = self._infer_model(self._get_eval_model(), input_tensors)
        outputs = outputs if type(outputs) is list else [outputs]
        if return_tensor:
            return outputs
        return [self.model.shape_tensor(outputs[i], name=output_names[i] if output_names else None, reverse=True, segmentation=self.segmentation, norms=self.norms) for i in range(len(outputs))]

    #-------PLOT-------

    def plot_losses(self, ax=None ,log10=True, save=False):
        """
        Plot training and validation losses.

        Args:
            ax (matplotlib.axes.Axes, default:None): Axis to plot on. If None, a new figure is created.
            log10 (bool, default:True): Whether to express losses in log10 scale. Default is True.

        Returns:
            figure and ax
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        x_training = [i[0] for i in self.training_losses]
        x_validation = [i[0] for i in self.validation_losses]
        y_training = [i[1] for i in self.training_losses]
        y_validation = [i[1] for i in self.validation_losses]
        
        loss_method_name = getattr(self.loss_method, "_get_name", lambda: "Custom")()
        ax.set_ylabel(loss_method_name)
        if log10:
            y_training = np.log10(y_training)
            y_validation = np.log10(y_validation)
            ax.set_ylabel(loss_method_name + " (log10)")

        ax.scatter(x_training, y_training, label="training losses")
        ax.scatter(x_validation, y_validation, label="validation losses")
        ax.plot(x_training, y_training)
        ax.plot(x_validation, y_validation)

        ax.set_xlabel("epoch")
        ax.legend()

        if save:
            self.save_fig(fig, fig_name='losses')
        
        return fig, ax

    def plot_validation(self, inter=(None,None), number_per_row=8, number=16, same_limits=True, save=False):
        """
        Show target and model prediction images
        """
        fig, axes = plot_batch(self.get_prediction_batch()[0 if inter[0] is None else inter[0]: -1 if inter[1] is None else inter[1]], same_limits=same_limits, number_per_row=number_per_row, number=number)
        if save:
            self.save_fig(fig, fig_name='validation')

    def plot_residuals(self, batch=None, ax=None, plot_distribution=True, color="blue", bins_inter=(None,None), save=False):
        """
        Plot model predictions residuals

        Args:
            ax (matplotlib.axes.Axes, optional): Axis to plot on. If None, a new figure is created.
            plot_distribution (bool, optional): Whether to plot the residuals distribution. Default is True.
            color (str, optional): Residuals distribution color. Default is blue
            bins_inter (tuple (x,x), optional): Set the plot min and max (min,max), min can be None when max takes a value.

        Returns:
            figure and ax
        """

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        if batch is None:
            batch = self.get_prediction_batch()
        d_target = np.array([np.log10(b[0]) for b in batch]).flatten()
        d_prediction = np.array([np.log10(b[1]) for b in batch]).flatten()

        residuals = d_prediction-d_target
        violin_num_bins = 5
        bins = np.linspace(min(d_target) if bins_inter[0] is None else bins_inter[0], max(d_target) if bins_inter[1] is None else bins_inter[1], violin_num_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2 
        bin_indices = np.digitize(d_target, bins) - 1 
        binned_residuals = [residuals[bin_indices == i] for i in range(violin_num_bins)]

        re_bin_centers = []
        re_binned_residuals = []
        mean_residuals = []
        for i, res in enumerate(binned_residuals):
            if res.size <= 0:
                continue
            re_bin_centers.append(bin_centers[i])
            re_binned_residuals.append(res)
            mean_residuals.append(np.mean(residuals[bin_indices == i]))
        bin_centers = re_bin_centers
        binned_residuals = re_binned_residuals

        if plot_distribution:
            vp = ax.violinplot(binned_residuals, positions=bin_centers, showmeans=False, showmedians=True)
            for i, body in enumerate(vp['bodies']):
                body.set_facecolor(color)
                body.set_alpha(0.5)
            for part in ['cbars', 'cmins', 'cmaxes', 'cmedians']:
                vp[part].set_edgecolor('black')
                vp[part].set_linewidth(1.0)
        ax.axhline(0, color='red', linestyle='--')
        ax.plot(bin_centers, mean_residuals, marker='o', linestyle='-', color='black', alpha=0.7)

        ax.set_xticks(bin_centers)
        ax.set_xlabel("Pixel value (log10)")
        ax.set_ylabel("Residuals (prediction-target)")
        ax.grid(True, linestyle="--", alpha=0.5)

        if(save):
            self.save_fig(fig, fig_name='residuals')

        return fig, ax
    
    @DeprecationWarning
    def plot_sim_validation(self, simulation, plot_total=False, save=False):
        sim_col_dens = simulation._compute_c_density()
        sim_mass_dens = simulation._compute_v_density(method=compute_mass_weighted_density)
        raw_sim_batch =  [(sim_col_dens,sim_mass_dens)]
        d_m_s_col = divide_matrix_to_sub(sim_col_dens)
        d_m_s_mass = divide_matrix_to_sub(sim_mass_dens)
        divided_sim_batch = rebuild_batch(d_m_s_col, d_m_s_mass)
        pred_raw_batch = self.predict(raw_sim_batch)
        pred_divided_batch = self.predict(divided_sim_batch)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [1, 1]})
        fig.suptitle("Simulation Validation")

        raw_fig, raw_axes = plot_batch(pred_raw_batch, number_per_row=1, same_limits=True)
        raw_axes = np.array(raw_axes).flatten()

        divided_fig, divided_axes = plot_batch(rebuild_batch([group_matrix([p[0] for p in pred_divided_batch])],[group_matrix([p[1] for p in pred_divided_batch])]), number_per_row=1, same_limits=True)
        divided_axes = np.array(divided_axes).flatten()

        if plot_total:
            raw_fig.canvas.draw()
            raw_axes_image = np.array(raw_fig.canvas.renderer.buffer_rgba())

            divided_fig.canvas.draw()
            divided_axes_image = np.array(divided_fig.canvas.renderer.buffer_rgba())

            axes[0].imshow(raw_axes_image)
            axes[0].set_title("Raw Data")
            axes[0].axis("off")

            axes[1].imshow(divided_axes_image)
            axes[1].set_title("Divided Data")
            axes[1].axis("off")

            if save:
                self.save_fig(fig, fig_name='sim_validation')
        else:
            fig = None
    
    def plot_validation_spatial_error(self,number_per_row=4,number=8,log=True,save=False):
        batch = self.get_prediction_batch()[:number]
        if log:
            error = (np.log10(np.array([b[0] for b in batch]))-np.array(np.log10([b[1] for b in batch])))
        else:
            error = np.abs(np.array([b[0] for b in batch])-np.array([b[1] for b in batch]))
        fig, axes = plt.subplots(int(np.ceil(len(error)/number_per_row)),number_per_row,figsize=(14, 9))
        for i,e in enumerate(error):
            if len(axes.shape) > 1:
                im = axes[(i//number_per_row)][i%number_per_row].imshow(e, cmap="jet")
            else:
                im = axes[i].imshow(e, cmap="jet")
            plt.colorbar(im)
        fig.subplots_adjust( left=None, bottom=None,  right=None, top=None, wspace=None, hspace=None)
        if save:
            self.save_fig(fig, fig_name='spatial_error')
        return fig, axes

    def plot_prediction_correlation(self,ax=None,factors=[0], save=False):
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        batch = self.get_prediction_batch()
        target_imgs = np.array([np.log10(b[0]) for b in batch]).flatten()
        min_t, max_t = np.min(target_imgs), np.max(target_imgs)
        plot_batch_correlation(batch, ax=ax)
        ax.set_xlabel("Target (log10)")
        ax.set_ylabel("Prediction (log10)")
        
        X = np.linspace(min_t,max_t,10)
        colors = FIGURE_CMAP(range(len(factors)))
        for i,f in enumerate(factors):
            if f == 0:
                continue
            f = np.abs(f)
        
            ax.plot(X,np.log10(f)+X,color=colors[i],linestyle="dashdot",label=fr"y=${f}\times x$")
            ax.plot(X,np.log10(1/f)+X,color=colors[i],linestyle="dashdot",label=fr"y=${1/f:.1f}\times x$")

        plt.legend()

        if save:
            self.save_fig(fig, fig_name='pred_correlation')

        return fig, ax

    def plot(self, save=False):
        """
        Plot all in one figure: validation correlation, residuals and losses.
        """
        plt.figure()
        plt.suptitle(self.model_name)
        
        ax1 = plt.subplot(2,2,1)
        self.plot_prediction_correlation(ax=ax1, save=False)

        ax2 = plt.subplot(2,2,2)
        self. plot_residuals(ax=ax2, save=False)

        ax3 = plt.subplot(2,2,3)
        self.plot_losses(ax=ax3, save=save)

        plt.tight_layout()

    #-------SAVE-------

    def save_fig(self, fig, fig_name='default'):
        if not(os.path.exists(MODEL_FOLDER)):
            os.mkdir(MODEL_FOLDER)
        model_path = os.path.join(MODEL_FOLDER,self.model_name.rsplit("_epoch",1)[0])
        fig_path = os.path.join(model_path,fig_name+'.jpg')
        fig.savefig(fig_path)

    def _modify_saved_settings(self, settings):
        """Override this method if you want to add new settings to be saved, don't forget to also override _modify_loaded_settings"""
        return settings
    
    def _modify_loaded_settings(self, settings):
        """Override this method if you want to add new settings to be auto loaded, don't forget to also override _modify_saved_settings"""
        return

    def save(self, model_name=None, is_cache=False):
        """
        Save model and model settings in a new folder

        Args:
            model_name (str, optional): set model name, if None it will take the trainer model_name if user had set it or a random uuid.
            is_cache (bool, default:False): Save the model as a cache (just one cached model can exists).

        Returns:
            bool: If this is a success
        """

        if(not(model_name is None)):
            self.model_name = model_name
        
        if not(os.path.exists(MODEL_FOLDER)):
            os.mkdir(MODEL_FOLDER)

        #while os.path.exists(os.path.join(MODEL_FOLDER,self.model_name)):
        #    self.model_name = str(uuid.uuid4())

        model_path = os.path.join(MODEL_FOLDER,self.model_name.rsplit("_epoch",1)[0])
        if is_cache:
            model_path = os.path.join(MODEL_FOLDER, "cached_model")
        if not(os.path.exists(model_path)):
            os.mkdir(model_path)
        elif is_cache:
            LOGGER.warn(f"A previous cached model was removed.")
            shutil.rmtree(model_path)
            os.mkdir(model_path)


        ep = self.last_epoch
        model_name = self.model_name.rsplit("_epoch",1)[0]+"_epoch"+str(ep) if not(is_cache) else "cached_model"
        if os.path.exists(os.path.join(model_path,model_name+".pth")):
            LOGGER.warn(f"Can't save {model_name} with epoch: {ep}")
            return
        
        torch.save(self.model.state_dict(),os.path.join(model_path,model_name+".pth"))
        if self.ema and self.ema_handler is not None:
            ema_path = os.path.join(model_path, f"{model_name}_ema.pth")
            torch.save(self.ema_handler.state_dict(), ema_path)

        loss_method_name = ""
        try:
            loss_method_name = self.loss_method._get_name()
        except AttributeError:
            loss_method_name = "Custom"

        cloned_network_settings = self.network_settings.copy()

        if "convBlock" in self.network_settings:
            cloned_network_settings["convBlock"] = self.network_settings["convBlock"].__name__ if not(type(self.network_settings["convBlock"]) is str) else self.network_settings["convBlock"]

        settings = {
            "model_name": self.model_name,
            "network": self.network_type,
            "network_settings": cloned_network_settings,
            "loss_method": loss_method_name,
            "optimizer": str(type(self.optimizer)),
            "learning_rate": str(self.learning_rate),
            "is_segmentation": self.segmentation,
            "scheduler": str(type(self.scheduler)),
            "total_epoch": str(self.last_epoch),
            "input_names": self.input_names,
            "target_names": self.target_names,
            "normalizations": str(self.norms),
            "training_set": str(self.training_set.name),
            "validation_set": str(self.validation_set.name),
            "system": get_system_info(),
            "training_losses": self.training_losses,
            "validation_losses": self.validation_losses,
        }

        settings = self._modify_saved_settings(settings)

        with open(os.path.join(model_path,'settings.json'), 'w') as file:
            json.dump(settings, file, indent=4)

        LOGGER.log(f"{self.model_name} saved.")

        return True

    @staticmethod
    def load(model_name, load_model=True):
        return load_trainer(model_name, load_model)

def load_trainer(model_name, load_model=True, trainer_class=Trainer)->Trainer:

    folder_model_name = model_name

    model_path = os.path.join(MODEL_FOLDER, model_name)
    if(not(os.path.exists(model_path))):
        LOGGER.error(f"Can't load {model_name}, file {model_path} doesn't exist.")
        return
    
    settings = {}
    with open(os.path.join(model_path,'settings.json')) as file:
        settings = json.load(file)
    
    trainer = trainer_class(model_name=settings["model_name"], segmentation=bool(settings["is_segmentation"] if "is_segmentation" in settings else False))
    if "training_set" in settings and not(settings["training_set"] is None):
        try:
            trainer.training_set = getDataset(settings["training_set"])
        except Exception as e:
            LOGGER.warn(f"Couldn't load training set: {e}")
    if "validation_set" in settings and not(settings["validation_set"] is None):
        try:
            trainer.validation_set = getDataset(settings["validation_set"])
        except Exception as e:
            LOGGER.warn(f"Couldn't load validation set: {e}")

    #TODO, Remove network options and convblock options by an auto handler
    try:
        network_options = NETWORK_OPTIONS
        network_convblock_options = CONVBLOCK_OPTIONS
    except:
        network_options = {}
        network_convblock_options = {}
    network_settings = settings["network_settings"] if "network_settings" in settings else {}
    if "convBlock" in network_settings:
        network_settings["convBlock"] = network_convblock_options[network_settings["convBlock"]]

    trainer.network_type = settings["network"]
    trainer.network_settings = network_settings
    trainer.network = network_options[settings["network"]]
    trainer.learning_rate = float(settings["learning_rate"])
    trainer.last_epoch = int(settings["total_epoch"])
    trainer.training_losses = settings["training_losses"]
    trainer.validation_losses = settings["validation_losses"]
    trainer.input_names = settings["input_names"] if "input_names" in settings else ["cdens"]
    trainer.target_names = settings["target_names"] if "target_names" in settings else (settings["target_name"] if "target_name" in settings else "vdens")
    trainer.optimizer_name = settings["optimizer"]

    trainer._modify_loaded_settings(settings)

    ema_state_path = os.path.join(model_path, f"{folder_model_name}_ema.pth")
    if os.path.exists(ema_state_path):
        trainer.ema = True

    if load_model:
        model = trainer.network(**trainer.network_settings)
        try:
            model.load_state_dict(torch.load(os.path.join(model_path,trainer.model_name+".pth"), map_location=trainer.device), strict=False)
        except FileNotFoundError:
            try:
                model.load_state_dict(torch.load(os.path.join(model_path,folder_model_name+".pth"), map_location=trainer.device), strict=False)
            except FileNotFoundError:
                print(os.path.join(model_path,trainer.model_name+f"_epoch{trainer.last_epoch}.pth"))
                model.load_state_dict(torch.load(os.path.join(model_path,trainer.model_name+f"_epoch{trainer.last_epoch}.pth"), map_location=trainer.device), strict=False)
        model.to(trainer.device)
        trainer.init(model=model)
        if os.path.exists(ema_state_path):
            if(trainer.ema_handler is None):
                trainer.ema_handler = ExponentialMovingAverage()
                trainer.ema_handler.register_model(trainer.model)
            trainer.ema_handler.load_state_dict(torch.load(ema_state_path, map_location=trainer.device))
            LOGGER.log(f"EMA state loaded for {model_name}.")

    LOGGER.log(f"{model_name} loaded")

    return trainer

def plot_models_residuals(trainers:List['Trainer'] = [], ax=None, colors:Union[List[str],None]=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    models_residuals = []
    models_targets = []
    for t in trainers:
        batch = t.get_prediction_batch()
        d_target = np.array([np.log(b[0])/np.log(10) for b in batch]).flatten()
        d_prediction = np.array([np.log(b[1])/np.log(10) for b in batch]).flatten()
        models_residuals.append(d_prediction-d_target)
        models_targets.append(d_target)
    models_name = [t.model_name for t in trainers]

    positions = np.arange(len(models_name))

    vp = ax.violinplot(models_residuals, positions=positions, showmeans=True, showmedians=True)
    colors = FIGURE_CMAP(np.linspace(FIGURE_CMAP_MIN, FIGURE_CMAP_MAX, len(models_name))) if colors is None else colors
    for i, body in enumerate(vp['bodies']):
        body.set_facecolor(colors[i])
        body.set_alpha(0.8)
    vp["cmedians"].set_edgecolor('black')
    vp["cmedians"].set_linestyle('dashed')
    vp["cmedians"].set_linewidth(1.2)
    for part in ['cbars', 'cmins', 'cmaxes', 'cmeans']:
        vp[part].set_edgecolor('black')
        vp[part].set_linewidth(1.2)
    ax.set_xticks(positions)
    ax.set_xticklabels(models_name)
    ax.set_ylabel("Residuals (log(prediction)-log(target))")
    ax.grid(True, linestyle="--", alpha=0.5)

    return fig, ax, colors

def plot_models_residuals_extended(trainers:List['Trainer'] = [], colors=None):
    fig = plt.figure()

    ax1 = plt.subplot2grid((len(trainers), 2), (0, 0), rowspan=len(trainers))
    _,_, colors = plot_models_residuals(trainers=trainers, ax=ax1, colors=colors)
    #ax1.set_title("Comparison of different models")

    for i,t in enumerate(trainers):
        ax = plt.subplot(len(trainers),2, (i+1)*2)
        t.plot_residuals(ax=ax, color=colors[i], bins_inter=(1.5,7.5))
        ax.set_ylim((-1.25, 1.25))
        ax.set_ylabel("")
        if i != len(trainers)-1:
            ax.set_xlabel("")

    plt.tight_layout()
    return fig, ax
    

def plot_accuracy(trainers=[], ax=None, sigmas=(0., 1., 20), show_errors=False, bins=None, col_dens=None,
                   use_linestyles=False, linestyle=None, color="black", marker=None, legend=True, xlabel="Error allowed (in log10)", ylabel="Accuracy" ):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    sigmas = np.linspace(sigmas[0], sigmas[1], sigmas[2])
    colors = FIGURE_CMAP(np.linspace(FIGURE_CMAP_MIN, FIGURE_CMAP_MAX, len(trainers)))
    linestyles = ['-', '--', '-.', ':'] 

    for i, t in enumerate(trainers):
        if bins is None:
            accuracies, accuracies_error = [], []
            for s in sigmas:
                acc_mean, acc_std = compute_batch_accuracy(t.get_prediction_batch(), sigma=s, bins=None, col_dens=col_dens)
                accuracies.append(acc_mean)
                accuracies_error.append(acc_std)
            accuracies = np.array(accuracies)
            accuracies_error = np.array(accuracies_error)

            style = linestyles[i % len(linestyles)] if use_linestyles else linestyle
            color = color if use_linestyles or linestyle is not None else colors[i]

            if show_errors:
                ax.fill_between(
                    sigmas,
                    np.clip(accuracies - accuracies_error, 0., 1.),
                    np.clip(accuracies + accuracies_error, 0., 1.),
                    color=color,
                    alpha=0.15 if use_linestyles else 0.2
                )
            ax.plot(sigmas, accuracies, marker=marker, linestyle=style, color=color, label=t.model_name)

        else:
            # Multiple bins per trainer
            all_bin_means, all_bin_stds = [], []
            for s in sigmas:
                result = compute_batch_accuracy(t.get_prediction_batch(), sigma=s, bins=bins, col_dens=col_dens)
                means = [r[0] for r in result]
                stds = [r[1] for r in result]
                all_bin_means.append(means)
                all_bin_stds.append(stds)

            all_bin_means = np.array(all_bin_means)
            all_bin_stds = np.array(all_bin_stds)
            n_bins = all_bin_means.shape[1]

            for b in range(n_bins):
                acc_mean = all_bin_means[:, b]
                acc_std = all_bin_stds[:, b]
                style = linestyles[b % len(linestyles)] if use_linestyles else '-'
                color = color if use_linestyles else plt.cm.tab20b((i*(n_bins-1)+b)/len(trainers)/(n_bins - 1))
                label = f"{t.model_name} - bin {b+1}"

                if show_errors:
                    ax.fill_between(
                        sigmas,
                        np.clip(acc_mean - acc_std, 0., 1.),
                        np.clip(acc_mean + acc_std, 0., 1.),
                        color=color,
                        alpha=0.1 if use_linestyles else 0.15
                    )
                ax.plot(sigmas, acc_mean, marker=marker, linestyle=style, color=color, label=label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if legend:
        ax.legend()
    ax.grid()

    return fig, ax

import re
from scipy.interpolate import griddata
import matplotlib.colors as mcolors
def heatmap(root_name, validation_batch, X, Y, ax=None):

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    Z = []
    X_flat = []
    Y_flat = []
    for l in Y:
        for bf in X:
            trainer = load_trainer(root_name+f"_l{str(l)}_bf{str(bf)}")
            if trainer is None:
                continue
            trainer.validation_batch = validation_batch
            acc, std = compute_batch_accuracy(trainer.get_prediction_batch(),sigma=0.3)
            Z.append(acc)
            Y_flat.append(l)
            X_flat.append(bf)
            trainer = None

    X = np.array(X_flat)
    Y = np.array(Y_flat)
    Z = np.array(Z)
    X_unique = np.sort(np.unique(X))
    Y_unique = np.sort(np.unique(Y))
    Z_grid = np.full((len(X_unique), len(Y_unique)), np.nan)
    for x, y, z in zip(X, Y, Z):
        x_idx = np.where(X_unique == x)[0][0]
        y_idx = np.where(Y_unique == y)[0][0] 
        Z_grid[x_idx, y_idx] = z
    
    
    grid_x, grid_y = np.meshgrid(X_unique, Y_unique, indexing='ij')
    all_points = np.array([(x, y) for x, y in zip(grid_x.ravel(), grid_y.ravel())])

    known_points = np.array([(x, y) for x, y in zip(X, Y)])
    known_values = Z

    interpolation_method = 'nearest'

    interpolated_values = griddata(known_points, known_values, all_points, method=interpolation_method)

    Z_grid = interpolated_values.reshape(len(X_unique), len(Y_unique))
    for i in range(len(Z_grid)):
        for j in range(len(Z_grid[i])):
            if np.isnan(Z_grid[i,j]):
                Z_grid[i,j] = 0

    cmap = plt.get_cmap("viridis", 100)
    norm = mcolors.Normalize(vmin=np.min(Z), vmax=np.max(Z))
    #cm = plt.pcolormesh(grid_x, grid_y, Z_grid.T, shading='auto', cmap=cmap, alpha=0.75, norm=norm)
    cm = ax.pcolormesh(X_unique, Y_unique, Z_grid.T, shading='auto', cmap=cmap, alpha=0.75, norm=norm)
    for i in range(len(X)):
        sc = ax.scatter(X[i], Y[i], color=cmap(norm(Z[i])), marker="o" , edgecolors='k', norm=norm)
    cbar = plt.colorbar(cm,label=r"Accuracy for $\sigma=0.3$",ax=ax)
    ax.set_xlabel("Base Filters")
    ax.set_ylabel("Layers")

    return fig, ax