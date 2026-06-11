from POLARIScore.networks.Trainer import Trainer
from POLARIScore.objects.Dataset import Dataset, getDataset
from POLARIScore.networks.addons import EarlyStopping
from POLARIScore.config import *
from POLARIScore.utils.utils import NumpyEncoder, numpy_decoder, ask_to_user
from typing import *
from torch.nn import MSELoss
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes
import time
import json
import shutil
from ast import literal_eval
import re
from matplotlib.widgets import Slider

"""Parameters example:

{"num_layers":{"type":int, "clamp":(1, 5)}, "base_filters":{"type":int, "clamp": (1, 4), "transform":(lambda x: 16*x, lambda x: int(x/16))}}

"""

class HyperParameterFineTuning():
    def __init__(self, trainer:Trainer, 
                 parameters:Dict[str,Dict]={"learning_rate":{"type":float}}):
        self.trainer = trainer
        self.parameters = parameters
        self.metric = MSELoss()
        if trainer is not None:
            self.validation_set = trainer.validation_set
        self.result:Tuple[np.ndarray, 'Trainer'] = None
        self.losses :list = []
        self.cache:Dict = {}

    def plot_losses(self, ax:Optional[matplotlib.axes.Axes]=None):
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        assert len(self.losses) > 0, LOGGER.error("No losses found.")

        ax.plot(range(len(self.losses)), self.losses, marker="+")
        ax.grid()
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")

        return fig, ax
    
    def plot_heatmap_3d(self, cmap="viridis"):

        assert len(self.parameters) == 3

        keys = list(self.parameters.keys())

        xs = sorted(set(k[0] for k in self.cache))
        ys = sorted(set(k[1] for k in self.cache))
        zs = sorted(set(k[2] for k in self.cache))

        X = {v: i for i, v in enumerate(xs)}
        Y = {v: i for i, v in enumerate(ys)}

        fig, ax = plt.subplots(figsize=(8, 6))
        plt.subplots_adjust(bottom=0.20)

        def build_heatmap(z_value):
            heatmap = np.full((len(ys), len(xs)), np.nan)

            for (x, y, z), loss in self.cache.items():
                if z == z_value:
                    heatmap[Y[y], X[x]] = loss

            return heatmap

        z0 = zs[0]

        heatmap = build_heatmap(z0)

        im = ax.imshow(
            heatmap,
            origin="lower",
            aspect="auto",
            cmap=cmap,
        )

        ax.set_xticks(np.arange(len(xs)))
        ax.set_xticklabels(xs)

        ax.set_yticks(np.arange(len(ys)))
        ax.set_yticklabels(ys)

        ax.set_xlabel(keys[0])
        ax.set_ylabel(keys[1])

        title = ax.set_title(f"{keys[2]} = {z0}")

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Loss")

        slider_ax = plt.axes([0.15, 0.05, 0.7, 0.03])

        slider = Slider(
            slider_ax,
            keys[2],
            0,
            len(zs) - 1,
            valinit=0,
            valstep=1,
        )

        fig._slider = slider

        marker = None

        def update(idx):

            nonlocal marker

            idx = int(idx)
            z_value = zs[idx]

            heatmap = build_heatmap(z_value)

            im.set_data(heatmap)

            if marker is not None:
                marker.remove()

            candidates = {
                k: v
                for k, v in self.cache.items()
                if k[2] == z_value
            }

            if candidates:
                best = min(candidates, key=candidates.get)

                marker = ax.scatter(
                    X[best[0]],
                    Y[best[1]],
                    c="red",
                    marker="x",
                    s=200,
                    linewidths=3,
                )

            title.set_text(f"{keys[2]} = {z_value}")

            fig.canvas.draw_idle()

        slider.on_changed(update)

        update(0)

        return fig, ax, slider
    
    def plot_heatmap(self, ax: Optional[matplotlib.axes.Axes] = None, cmap: str = "viridis"):

        assert len(self.parameters) == 2, ("Heatmap only supported for exactly 2 parameters.")

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.figure

        keys = list(self.parameters.keys())

        xs = sorted(set(k[0] for k in self.cache.keys()))
        ys = sorted(set(k[1] for k in self.cache.keys()))

        X = {v: i for i, v in enumerate(xs)}
        Y = {v: i for i, v in enumerate(ys)}

        heatmap = np.full((len(ys), len(xs)), np.nan)

        for params, loss in self.cache.items():
            x, y = params
            heatmap[Y[y], X[x]] = loss

        im = ax.imshow(heatmap,origin="lower",aspect="auto",cmap=cmap,)

        ax.set_xticks(np.arange(len(xs)))
        ax.set_xticklabels(xs)

        ax.set_yticks(np.arange(len(ys)))
        ax.set_yticklabels(ys)

        ax.set_xlabel(keys[0])
        ax.set_ylabel(keys[1])

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Loss")

        best = min(self.cache, key=self.cache.get)
        ax.scatter(X[best[0]],Y[best[1]],c="red",marker="x",s=200,linewidths=3,)

        return fig, ax
    
    def save(self, name:Optional[str]=None, force:bool=True):
        if name is None:
            name = self.trainer.network_type
        if not(os.path.exists(TUNING_FOLDER)):
            os.mkdir(TUNING_FOLDER)
        folder_path = os.path.join(TUNING_FOLDER, name)
        if os.path.exists(folder_path) and force:
            shutil.rmtree(folder_path)
            LOGGER.warn(f"Previous tuning folder named {name} was removed.")
        os.mkdir(folder_path)
        
        model_losses = {}
        for k in self.cache.keys():
            model_losses[str(k)] = self.cache[k]
        losses_dict = {
            "model_losses": model_losses, 
            "losses": self.losses,
        }
        settings_dict = {
            "dataset": self.validation_set.name,
            "result": self.result[0],
            "parameters_tuned": str(self.parameters),
        }
        with open(os.path.join(folder_path, 'losses.json'), 'w') as file:
            json.dump(losses_dict, file, indent=4, cls=NumpyEncoder)
        with open(os.path.join(folder_path, 'settings.json'), 'w') as file:
            json.dump(settings_dict, file, indent=4, cls=NumpyEncoder)

        LOGGER.log(f"Tuning {name} saved")

    def load(self, name:Optional[str]=None, blacklist:List[str]=[]):
        """!!! This overwrite parameters"""
        if name is None:
            assert self.trainer is not None, LOGGER.error("If name is not given, then a trainer is needed.")
            name = self.trainer.network_type

        folder_path = os.path.join(TUNING_FOLDER, name)
        assert os.path.exists(folder_path), LOGGER.error(f"Can't load tuning {name} because there is no folder with such name.")

        with open(os.path.join(folder_path,'settings.json')) as file:
            settings = json.load(file, object_hook=numpy_decoder)
            if "dataset" in settings and "dataset" not in blacklist:
                self.validation_set = getDataset(settings["dataset"])
            if "result" in settings and "result" not in blacklist:
                self.result = (settings["result"], self.trainer)
            #if "parameters_tuned" in settings and "parameters_tuned" not in blacklist:
            #    self.parameters = settings["parameters_tuned"]
        with open(os.path.join(folder_path, 'losses.json')) as file:
            losses = json.load(file,  object_hook=numpy_decoder)
            if "model_losses" in losses and "model_losses" not in blacklist:
                self.cache = {}
                for k in losses["model_losses"]:
                    k_true = re.sub(r"np\.int64\((\d+)\)", r"\1", k)
                    k_true = re.sub(r"np\.float64\(([^()]*)\)", r"\1", k_true)
                    self.cache[literal_eval(k_true)] = losses["model_losses"][k]
            if "losses" in losses and "losses" not in blacklist:
                self.losses = losses["losses"]
        LOGGER.log(f"Tuning {name} loaded.")

    def init_tuning(self, test_dataset:Dataset, metric:Optional[Callable[[np.ndarray, np.ndarray], float]]=None,):
        self.metric = self.metric if metric is None else metric
        trainer = self.trainer
        trainer.validation_set = test_dataset
        for i, p_name in enumerate(self.parameters.keys()):
            if "index" not in self.parameters[p_name] or self.parameters[p_name]['index'] != i:
                self.parameters[p_name]['index'] = i

            if p_name in trainer.network_settings and "network_setting" not in self.parameters[p_name]:
                self.parameters[p_name]["network_setting"] = True
        
        #Init
        parameters = []
        for p_name in self.parameters.keys():
            if "network_setting" in self.parameters[p_name] and self.parameters[p_name]["network_setting"]:
                p_value = trainer.network_settings[p_name]
            else:
                p_value = vars(trainer)[p_name]

            
            if "transform" in self.parameters[p_name]:
                p_value = self.parameters[p_name]["transform"][1](p_value)
            parameters.append(p_value)
        
        parameters = np.array(parameters)
        parameters_prev = parameters.copy()

        loss = np.inf
        loss_prev = np.inf

        return trainer, (parameters, parameters_prev), (loss, loss_prev)
    
    def init_trainer_using_params(self, trainer:'Trainer', params):
        for i, p in enumerate(self.parameters.keys()):
            param = self.parameters[p]
            p_value = params[i]
            if "type" in param:
                p_value = param["type"](p_value)
            if "clamp" in param:
                p_min, p_max = param["clamp"]
                p_value = min(max(p_value, p_min), p_max)
            if "transform" in param:
                p_value = param["transform"][0](p_value)

            if "network_setting" in param and param["network_setting"]:
                trainer.network_settings[p] = p_value
            elif not("method_setting" in param) or not(param["method_setting"]):
                vars(trainer)[p] = p_value
        trainer.last_epoch = 0
        trainer.early_stopping = EarlyStopping.EarlyStopping()
        trainer.training_losses = []
        trainer.validation_losses = []
        trainer.init()
        return trainer 

    def evaluate_parameters(self, params, epoch_number:int, use_caching:bool=True, log_training:bool=False, early_stop_training:bool=False):
        if use_caching and tuple(params) in self.cache:
            return self.cache[tuple(params)]

        local_trainer = self.init_trainer_using_params(self.trainer, params)

        try:
            loss = float(ask_to_user(f"If you know the loss of model using {params}, you have 10s to give it: ", seconds=10, default=np.inf))
            if np.isfinite(loss):
                return loss
        except:
            LOGGER.warn("User query failed.")

        if not log_training:
            LOGGER.disable()

        i = 0
        try:
            while i<5:
                try:
                    local_trainer.train(
                        epoch_number=epoch_number, batch_number=params[self.parameters['batch_number']['index']] if "batch_number" in self.parameters else 16,
                        cache=False, early_stopping=early_stop_training, print_epoch_progress_bar=False if log_training else f"Training {params}",
                        training_mode="normal"
                    )
                    break
                except AssertionError:
                    i += 1                
        except torch.OutOfMemoryError:
            LOGGER.enable()
            LOGGER.error(f"Out Of Memory for model {params}.")
            self.cache[tuple(params)] = np.inf
            return np.inf


        LOGGER.enable()

        val_loss = self.metric(local_trainer)
        self.cache[tuple(params)] = val_loss

        return val_loss
        
    def discrete_tuning(self, test_dataset:Dataset, step_number:int=100, epoch_number=100, metric:Optional[Callable[['Trainer'], float]]=None,
                     optimizer_settings={"step_size":1, "scheduler":5, "eps":1e-3}, use_caching:bool=True, early_stop_training:bool=True, log_training:bool=False,
                     method:Literal["full_gradient","partial_gradient","random_walk","coordinate_descent"]="full_gradient"):
        opt_settings = {"step_size":1, "scheduler":5, "eps":1e-3}
        for k in opt_settings.keys():
            if k not in optimizer_settings:
                continue
            opt_settings[k] = optimizer_settings[k]

        method = method.lower()

        LOGGER.log(f"Discrete tuning of parameters {list(self.parameters.keys())} began.")

        def _format_time(seconds:float)->str:
            hours, rem = divmod(seconds, 3600)
            minutes, seconds = divmod(rem, 60)
            return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"


        if isinstance(opt_settings["step_size"], int):
            opt_settings["step_size"] = [opt_settings["step_size"] for _ in range(len(self.parameters))]
        opt_settings["step_size"] = np.array(opt_settings["step_size"])
        assert len(opt_settings["step_size"]) == len(self.parameters.keys()), LOGGER.error("Step size parameter need to be of the size of the parameters")
        
        trainer, p, l = self.init_tuning(test_dataset, metric)
        parameters, parameters_prev = p
        loss, loss_prev = l

        self.losses = []
        grad_losses = []

        step = 0
        scheduler_step = 0
        start_time = time.time()
        while step < step_number and loss>opt_settings['eps']:

            if log_training:
                LOGGER.border(f"Parameters: {parameters}")

            step_time = time.time()
            step += 1

            new_loss = self.evaluate_parameters(parameters, epoch_number=epoch_number, use_caching=use_caching, log_training=log_training, early_stop_training=early_stop_training)
            loss_prev = loss
            loss = new_loss
            
            count = 0
            if np.isfinite(loss):
                self.losses.append(loss)
                if len(self.losses) > 1 and np.isfinite(loss_prev):
                    grad_losses.append(loss-loss_prev)
                    last_grads = np.array(grad_losses[-6:])
                    count = np.sum(last_grads > 0)

            if count > 2:
                scheduler_step += 1
                if scheduler_step > opt_settings["scheduler"]:
                    scheduler_step = 0
                    if np.any(opt_settings["step_size"] > 1):
                        opt_settings["step_size"] = np.maximum(opt_settings["step_size"] - 1, 1)

            if method == "full_gradient":
                if np.isfinite(loss) and np.isfinite(loss_prev):
                    delta = parameters - parameters_prev
                    grad = np.divide(loss - loss_prev,delta,out=np.zeros_like(delta, dtype=float),where=delta != 0,)
                else:
                    grad = np.random.choice([-1, 1], size=len(parameters))
            elif method == "partial_gradient":

                grad = np.zeros(len(parameters), dtype=float)

                for i in range(len(parameters)):
                    p_plus = parameters.copy()
                    p_minus = parameters.copy()

                    p_plus[i] += 1
                    p_minus[i] -= 1

                    p_key = list(self.parameters.keys())[i]
                    if "clamp" in self.parameters[p_key]:
                        c_min, c_max = self.parameters[p_key]["clamp"]
                        p_plus[i] = np.clip(p_plus[i], c_min, c_max)
                        p_minus[i] = np.clip(p_minus[i], c_min, c_max)
                    
                    if p_plus[i] != parameters[i]:
                        loss_plus = self.evaluate_parameters(p_plus, epoch_number=epoch_number, use_caching=use_caching, log_training=log_training, early_stop_training=early_stop_training)
                        loss_minus = np.inf
                    else:
                        loss_plus = np.inf
                        loss_minus = self.evaluate_parameters(p_minus, epoch_number=epoch_number, use_caching=use_caching, log_training=log_training, early_stop_training=early_stop_training) if p_minus[i] != parameters[i] else np.inf

                    if np.isfinite(loss_plus) and np.isfinite(loss_minus):
                        grad[i] = (loss_plus - loss_minus) / 2.0
                    elif np.isfinite(loss_plus):
                        grad[i] = loss_plus - loss
                    elif np.isfinite(loss_minus):
                        grad[i] = loss - loss_minus
                    else:
                        grad[i] = np.random.choice([-1, 1])
            elif method == "random_walk":
                grad = np.random.choice([-1, 1], size=len(parameters))
            
            direction = np.sign(grad)
            direction[direction == 0] = 0
            new_parameters = parameters - direction * opt_settings["step_size"]

            cl_max = []
            cl_min = []
            for p_key in self.parameters.keys():
                if "clamp" in self.parameters[p_key]:
                    c_min, c_max = self.parameters[p_key]["clamp"]
                else:
                    c_min, c_max = -9999, 9999
                cl_min.append(c_min)
                cl_max.append(c_max)
            new_parameters = parameters.copy() - direction*opt_settings["step_size"]
            new_parameters = np.maximum(np.minimum(new_parameters, cl_max), cl_min)
            parameters_prev = parameters.copy()
            parameters = new_parameters.copy()

            actual_time = time.time()
            step_time = actual_time - step_time
            time_left = (actual_time-start_time) / (step+1) * (step_number-(step+1))


            LOGGER.print(
                f"Step {step}/{step_number} | "
                f"Elapsed: {_format_time(actual_time-start_time)} | "
                #f"Time Left: {_format_time(time_left)} | "
                f"Loss: {loss:.2e}, Parameters: {parameters_prev}->{parameters}",
                type="tuning",
                level=1,
            )

        LOGGER.log(f"Discrete tuning ended with parameters: {dict(zip(self.parameters.keys(), parameters_prev))}")

        self.result = (parameters_prev, trainer)

        self.save()

        return self.result

    def adam_tuning(self, test_dataset:Dataset, step_number:int=100, epoch_number=100, metric:Optional[Callable[[np.ndarray, np.ndarray], float]]=None,
                     optimizer_settings={"beta_1":0.9, "beta_2":0.999, "eps":1e-8, "step_size":1e-3}):
        opt_settings={"beta_1":0.9, "beta_2":0.999, "eps":1e-8, "step_size":1e-3}
        for k in opt_settings.keys():
            if k not in optimizer_settings:
                continue
            opt_settings[k] = optimizer_settings[k]

        trainer, p, l = self.init_tuning(test_dataset, metric)
        parameters, parameters_prev = p
        loss, loss_prev = l

        m = np.zeros_like(parameters)
        m_prev = np.zeros_like(parameters)
        v = np.zeros_like(parameters)
        v_prev = np.zeros_like(parameters)                   

        step = 0
        while step < step_number and loss>opt_settings['eps']:
            step += 1
            
            trainer = self.init_trainer_using_params(trainer, parameters)
            trainer.train(epoch_number=epoch_number, batch_number=parameters[self.parameters['batch_number']['index']] if "batch_number" in self.parameters else 32, early_stopping=False, cache=False)
            pred_batch = trainer.get_prediction_batch(force_compute=True)
            target_tensors = np.array([b[0] for b in pred_batch])
            pred_tensors = np.array([b[1] for b in pred_batch])

            new_loss = self.metric(target_tensors, pred_tensors)
            loss_prev = loss
            loss = new_loss

            if step == 1:
                g = np.zeros_like(parameters) #Use two inferences (parameters+epsilon) to set a initial gradient.
            else:
                g = (loss - loss_prev) / (parameters - parameters_prev + 1e-8)

            m = opt_settings["beta_1"]*m_prev+(1-opt_settings["beta_1"])*g
            v = opt_settings["beta_2"]*v_prev+(1-opt_settings["beta_2"])*g**2
            alpha_t = opt_settings["step_size"]*np.sqrt(1-opt_settings["beta_2"]**step)/(1-opt_settings["beta_1"]**step)
            new_parameters = parameters-alpha_t*m/(np.sqrt(v)+opt_settings["eps"])
            parameters_prev = parameters.copy()
            parameters = new_parameters.copy()
            m_prev = m.copy()
            v_prev = v.copy()

        self.result = (parameters, trainer)
        return self.result

        

