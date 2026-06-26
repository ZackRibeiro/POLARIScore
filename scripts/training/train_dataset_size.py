import matplotlib.pyplot as plt
import numpy as np
from POLARIScore.config import *
import POLARIScore.networks.Trainer as Trainer
from POLARIScore.networks.INNTrainer import INNTrainer, cINN
from POLARIScore.networks.DDPTrainer import DDPTrainer, DDPMUnet
from POLARIScore.objects.Dataset import Dataset, getDataset
from POLARIScore.networks.utils.nn_utils import compute_batch_accuracy
from POLARIScore.utils.utils import *
import glob
import re
import argparse
from typing import *
from POLARIScore.networks.utils.nn_utils import *
import shutil

def classic_log_mse(output, target):
    output = output[0]
    target = target[0]

    output_phys = DATA_NORMALIZATION_VDENS_TORCH[1](output)
    target_phys = DATA_NORMALIZATION_VDENS_TORCH[1](target)
    output_log = torch.log(output_phys)
    target_log = torch.log(target_phys)
    mse = torch.mean((output_log - target_log) ** 2)
    return mse

#50 STEPS FOR UNET
def train_model_varying_dataset_size(trainer:'Trainer.Trainer', sizes:List[float]=np.linspace(0.1,1.,20), validation_method=None):

    training_dataset = trainer.training_set
    model_root_name = trainer.model_name

    metric_results = []
    real_sizes_trained = []
    cache_folder = os.path.join(MODEL_FOLDER, "cached_model")

    for j,s in enumerate(sizes):
        training_dataset = getDataset(training_dataset.name)
        size = s*len(training_dataset.batch)
        size = max(min(size, len(training_dataset.batch)),1)

        if os.path.exists(cache_folder):
            shutil.rmtree(cache_folder)
        
        new_training_batch = {}
        for i,k in enumerate(training_dataset.batch.keys()):
            if i < size:
                new_training_batch[k] = training_dataset.batch[k]
            else:
                break
        training_dataset.batch = new_training_batch

        size = len(training_dataset.batch)
        real_sizes_trained.append(size)
        model_name = model_root_name+f"_t{str(size)}"
        model_path = os.path.join(MODEL_FOLDER, model_name)
        if(os.path.exists(model_path)):
            LOGGER.warn(model_name+f" already exists, delete the folder if you want to train a new model with these settings.")
            trainer = Trainer.load_trainer(model_name, trainer_class=trainer.__class__)
            trainer.validation_loss_method = validation_method
            trainer.norms = {
                "cdens": DATA_NORMALIZATION_CDENS,
                "vdens": DATA_NORMALIZATION_VDENS,
            }

        else:
            LOGGER.log(f"Now training: {size}({str(np.round((j+1)/(len(sizes))*100,3))}%)")
            
            #trainer = DDPTrainer(DDPMUnet, training_ds, validation_ds, f"DDPM_size_{str(int(t[0]))}l_{str(int(t[1]))}f")
            trainer = INNTrainer(cINN, training_ds, validation_ds, f"cINN_size_{str(int(t[0]))}l_{str(int(t[1]))}f")
            #trainer.pred_type = "v"
            trainer.norms = { 
                "cdens": DATA_NORMALIZATION_CDENS,
                "vdens": DATA_NORMALIZATION_VDENS,
            }
            trainer.learning_rate = 1e-3
            trainer.network_settings["base_filters"] = int(t[1]*16)
            trainer.network_settings["num_layers"] = t[0]
            #trainer.network_settings["attention_layers"] = [2]
            #trainer.network_settings["attention_heads"] = [8]
            #trainer.ema = True
            #trainer.ema_warmup = 30
            trainer.network_settings["coupling_block_per_layer"] = 3
            #trainer.validation_loss_method = classic_log_mse
            trainer.training_random_transform = True
            trainer.optimizer_name = "Adam"
            trainer.target_names = ["vdens"]
            trainer.input_names = ["cdens"]

            #trainer.scheduler = None
            trainer.training_set = training_dataset
            trainer.validation_loss_method = validation_method
            #trainer.learning_rate = 1e-3
            trainer.model_name = model_name
            trainer.early_stopping = Trainer.EarlyStopping()
            trainer.training_losses = []
            trainer.validation_losses = []
            trainer.last_epoch = 0
            trainer.inference_time = None
            trainer.cache_threshold = 10.
            trainer.init()
            LOGGER.disable()
            trainer.train(1500, batch_number=8, cache=True, early_stopping=True, print_epoch_progress_bar=True, modify_learning_rate=False, training_mode="accumulation")
            LOGGER.enable()

            if os.path.exists(cache_folder):
                trainer = Trainer.load_trainer("cached_model", trainer_class=trainer.__class__)
                trainer.norms = { 
                    "cdens": DATA_NORMALIZATION_CDENS,
                    "vdens": DATA_NORMALIZATION_VDENS,
                }
                trainer.model_name = model_name

            trainer.save()
        
        #accuracy = trainer.validation_losses[-1][1]
        accuracy = find_error_for_batch_accuracy(trainer.get_prediction_batch(remove_residuals_baseline=True), accuracy=0.8, sigma1=1.)
        #print(accuracy)
        metric_results.append(accuracy)
    
    real_sizes_trained = np.array(real_sizes_trained)
    metric_results = np.array(metric_results)

    sorted_indexes = np.argsort(real_sizes_trained)

    return real_sizes_trained[sorted_indexes], metric_results[sorted_indexes]

training_ds = getDataset("batch_highres_2_b1")
validation_ds = getDataset("batch_highres_2_b2")

tested_models = [(3,2)]
fig, ax = plt.subplots()
colors = ["tab:blue","tab:orange","tab:green"]
for i,t in enumerate(tested_models):
    #trainer = Trainer.Trainer(Trainer.UNet, training_ds, validation_ds, f"UNet_size_{str(int(t[0]))}l_{str(int(t[1]))}f")
    #trainer = DDPTrainer(DDPMUnet, training_ds, validation_ds, f"DDPM_size_{str(int(t[0]))}l_{str(int(t[1]))}f")
    trainer = INNTrainer(cINN, training_ds, validation_ds, f"cINN_size_{str(int(t[0]))}l_{str(int(t[1]))}f")
    #trainer.pred_type = "v"
    trainer.norms = { 
        "cdens": DATA_NORMALIZATION_CDENS,
        "vdens": DATA_NORMALIZATION_VDENS,
    }
    trainer.learning_rate = 1e-3
    trainer.network_settings["base_filters"] = int(t[1]*16)
    trainer.network_settings["num_layers"] = t[0]
    #trainer.network_settings["attention_layers"] = [2]
    #trainer.network_settings["attention_heads"] = [8]
    #trainer.ema = True
    #trainer.ema_warmup = 30
    trainer.network_settings["coupling_block_per_layer"] = 2
    trainer.training_random_transform = True
    trainer.optimizer_name = "Adam"
    trainer.target_names = ["vdens"]
    trainer.input_names = ["cdens"]

    x, y = train_model_varying_dataset_size(trainer, validation_method=classic_log_mse)
    X, Y, X_err, Y_err = bin_mean(x, y, nbins=20, method="mean", return_deviation=True, logspace=False)

    ax.plot(X,Y, color=colors[i])
    ax.scatter(x, y, color=colors[i], marker="+")
    ax.set_xlabel("Training-set size")
    ax.set_ylabel("MSE Loss")

plt.show()

"""
trainer = INNTrainer(cINN, training_ds, validation_ds, "cINN_low")
#trainer = load_trainer("SizeAware_Unet")
#trainer.pred_type = "v"
trainer.norms = { 
    "cdens": DATA_NORMALIZATION_CDENS,
    "vdens": DATA_NORMALIZATION_VDENS,
#    "physize": (lambda x:x, lambda x:x)
}

#trainer.ema = True
#trainer.validation_loss_method = classic_log_mse
#trainer.ema_warmup = 30
trainer.learning_rate = 1e-7
#trainer.network_settings["base_filters"] = 32
#trainer.network_settings["num_layers"] = 2

#trainer.network_settings["img_dim"] = 128
trainer.network_settings["base_filters"] = 32
trainer.network_settings["num_layers"] = 3
trainer.network_settings["coupling_block_per_layer"] = 2
#trainer.network_settings["attention_layers"] = [3,4]

#trainer.network_settings["attention_layers"] = [2]
#trainer.network_settings["attention_heads"] = [8]
#trainer.network_settings["filter_function"] = "linear"
trainer.training_random_transform = True
trainer.optimizer_name = "SGD"
trainer.target_names = ["vdens"]
trainer.input_names = ["cdens"]
#trainer.auto_save = 250
#trainer.scheduler = None
trainer.init()
#trainer.train(1000,batch_number=8,compute_validation=10,early_stopping=False)
#trainer.save()
#trainer.get_validation_error()

trainer.validation_loss_method = classic_log_mse
"""