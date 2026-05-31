from POLARIScore.networks.Trainer import Trainer
from POLARIScore.objects.Dataset import Dataset
from POLARIScore.networks.addons import EarlyStopping
from POLARIScore.config import *
from typing import *
from torch.nn import MSELoss
import numpy as np

"""Parameters example:

{"num_layers":{"type":int, "clamp":(1, 5)}, "base_filters":{"type":int, "clamp": (1, 4), "transform":(lambda x: 16*x, lambda x: int(x/16))}}

"""
class HyperParameterFineTuning():
    def __init__(self, init_trainer:Trainer, 
                 parameters:Dict[str,Dict]={"learning_rate":{"type":float}}):
        self.init_trainer = init_trainer
        self.parameters = parameters
        self.metric = MSELoss()
        self.optimizer = "ADAM"
        self.validation_set = init_trainer.validation_set
    def start_tuning(self, test_dataset:Dataset, step_number:int=100, epoch_number=100, metric:Optional[Callable[[np.ndarray, np.ndarray], float]]=None,
                     optimizer_settings={"beta_1":0.9, "beta_2":0.999, "eps":1e-8, "step_size":1e-3}):
        self.metric = self.metric if metric is None else metric
        trainer = self.init_trainer
        trainer.validation_set = test_dataset
        opt_settings={"beta_1":0.9, "beta_2":0.999, "eps":1e-8, "step_size":1e-3}
        for k in opt_settings.keys():
            if k not in optimizer_settings:
                continue
            opt_settings[k] = optimizer_settings[k]

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
                p_value = self.parameters["transform"][1](p_value)
            parameters.append(p_value)
        
        parameters = np.array(parameters)
        parameters_prev = parameters

        loss = np.inf
        loss_prev = np.inf

        m = np.zeros_like(parameters)
        m_prev = np.zeros_like(parameters)
        v = np.zeros_like(parameters)
        v_prev = np.zeros_like(parameters)

        def _init_network_using_params(trainer:'Trainer', params):
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
            return trainer                    

        step = 0
        while step < step_number and loss>optimizer_settings['eps']:
            step += 1
            
            trainer = _init_network_using_params(trainer, parameters)
            trainer.init()
            trainer.train(epoch_number=epoch_number, batch_number=parameters[self.parameters['batch_number']['index']] if "batch_number" in self.parameters else 32, early_stopping=False, cache=False)
            pred_batch = trainer.get_prediction_batch(force_compute=True)
            target_tensors = np.array([b[0] for b in pred_batch])
            pred_tensors = np.array([b[1] for b in pred_batch])

            new_loss = self.metric(target_tensors, pred_tensors)
            loss_prev = loss
            loss = new_loss

            if step == 1:
                g = np.zeros_like(parameters)
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

        self.result = parameters
        return self.result

        

