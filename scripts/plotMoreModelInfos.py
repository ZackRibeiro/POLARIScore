import matplotlib.pyplot as plt
from POLARIScore.config import *
from POLARIScore.networks.Trainer import load_trainer, Trainer
from POLARIScore.networks.utils.nn_utils import compute_batch_accuracy
import glob
import re
def plot_modelset(root_name, validation_batch=None, prefix="t", ax=None):

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    models_paths = glob.glob(os.path.join(MODEL_FOLDER,root_name)+f"_{prefix}*")
    X = []
    names = []
    for p in models_paths:
        p = p.split('/')[-1]
        match = re.search(r"_t(\d+)", p)
        if match:
            X.append(int(match.group(1)))
            names.append(p)
        else:
            LOGGER.warn(f"Can't read property in model name: {p}.")
            continue

    indexes = np.argsort(X)
    X = np.array(X)[indexes]
    names = np.array(names)[indexes]

    Y = []
    for n in names:
        trainer = load_trainer(n)
        if trainer is None:
            continue
        if not(validation_batch is None):
            trainer.validation_set = validation_batch
        acc, std = compute_batch_accuracy(trainer.get_prediction_batch(),sigma=0.3)
        Y.append(acc)
        del trainer
    
    ax.plot(X, Y)
    ax.scatter(X, Y)
    ax.grid(True)
    ax.set_xlabel("Size of the training set")
    ax.set_ylabel(r"Accuracy for $\sigma=0.3$")
    fig.tight_layout()

    return fig, ax

def generate_model_map(root_name, train_batch, validation_batch, network, layers=[2,3,4,5], base_filters=[8,16,32,48,64,80]):

    for i,l in enumerate(layers):
        for j,bf in enumerate(base_filters):
            model_path = os.path.join(MODEL_FOLDER, root_name+f"_l{str(l)}_bf{str(bf)}")
            if(os.path.exists(model_path)):
                LOGGER.warn(root_name+f"_l{str(l)}_bf{str(bf)}"+f" already exists, delete the folder if you want to traina new model with these settings.")
                continue
            LOGGER.log(f"Now training: {l}, {bf} ({str(np.round((i*len(base_filters)+j)/(len(base_filters)*len(layers))*100,3))}%)")
            trainer  = Trainer(network, train_batch,validation_batch, model_name=root_name+f"_l{str(l)}_bf{str(bf)}")
            trainer.network_settings["base_filters"] = bf
            trainer.network_settings["num_layers"] = l
            trainer.training_random_transform = True
            trainer.network_settings["attention"] = True
            trainer.init()
            trainer.train(int(2000+1000*(1-i/len(layers))*(1-i/len(base_filters))))
            trainer.save()

def generate_model_training_map(root_name, train_batch, validation_batch, network, training=[8,16,32,48,64,80]):

    for i,l in enumerate(training):
        if l > 1:
            l = l/len(train_batch.batch)
        dataset, _ = train_batch.split(l)
        l = len(dataset.batch)
        model_path = os.path.join(MODEL_FOLDER, root_name+f"_t{str(l)}")
        if(os.path.exists(model_path)):
            LOGGER.warn(root_name+f"_t{str(l)}"+f" already exists, delete the folder if you want to train a new model with these settings.")
            continue
        LOGGER.log(f"Now training: {l}({str(np.round((i)/(len(training))*100,3))}%)")
        trainer  = Trainer(network, dataset ,validation_batch, model_name=root_name+f"_t{str(l)}")
        trainer.network_settings["base_filters"] = 64
        trainer.network_settings["num_layers"] = 4
        #trainer.network_settings["convBlock"] = DoubleConvBlock
        trainer.training_random_transform = True
        trainer.network_settings["attention"] = True
        trainer.target_names = "vdens"
        trainer.input_names = ["cdens"]
        trainer.optimizer_name = "SGD"
        trainer.init()
        trainer.train(1000, batch_number=16, cache=False)
        trainer.save()