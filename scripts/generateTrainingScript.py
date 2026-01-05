import inspect
from pathlib import Path
import ast
import os
from POLARIScore.config import EXPORT_FOLDER, LOGGER

from POLARIScore.objects.Dataset import Dataset


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--arch", required=False, default="UNet", help="Network architecture (default: 'UNet')")
parser.add_argument("--trainer", required=False, default=None, help="Specified Trainer for Network (default: None)")
parser.add_argument("--output", required=False, default=EXPORT_FOLDER, help="Where the script will be generated (default: '{EXPORT_FOLDER}')")
args = parser.parse_args()

def extract_functions_and_classes(path):
    source = Path(path).read_text()
    tree = ast.parse(source)

    blocks = []

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            start = node.lineno - 1
            end = node.end_lineno
            block = "\n".join(source.splitlines()[start:end])
            blocks.append(block)

    return "\n\n".join(blocks)

def extract_non_import_code(path):
    source = Path(path).read_text()
    lines = source.splitlines()

    tree = ast.parse(source)

    blocks = []

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue

        start = node.lineno - 1
        end = node.end_lineno
        blocks.append("\n".join(lines[start:end]))

    return "\n\n".join(blocks) 

#make an arg parser, e.g to choose which trainer to use/which network...
TEMPLATE = """
import os, sys, ast, re, platform, inspect, datetime, uuid, glob, ast, copy, math
import numpy as np
import time
import shutil
import json
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.cm as cm
from matplotlib.widgets import Slider
import torch, scipy
from torch.nn import init
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import rotate
from scipy.stats import lognorm
from typing import Any, List, Tuple, Union, Callable, Literal, Optional, Dict
from scipy.interpolate import griddata
import matplotlib.colors as mcolors

{{UTILS_METHODS}}
{{ARCHITECTURE_CLASS}}
{{DATASET_CLASS}}
{{TRAINER_CLASS}}

#-------------------------------------
#SETUP PART

if __name__ == "__main__":
    LOGGER.warn("Actually using the setup example.")
    ds = getDataset("batch_default") #All generated datasets using POLARIScore have a prefix: 'batch_'
    ds1, ds2 = ds.split(0.7) #Split the dataset into training set (70%) and validation set (30%)

    trainer = Trainer(UNet, ds1, ds2, model_name="My_UNet") #Create a trainer using UNet architecture
    #trainer.norms = {
    #    "cdens": DATA_NORMALIZATION_CDENS,
    #    "vdens": DATA_NORMALIZATION_VDENS,
    #} #If you need to normalize the data
    trainer.network_settings["base_filters"] = 64 #Change architecture settings
    trainer.network_settings["num_layers"] = 4
    trainer.training_random_transform = True #Apply random transformations (no deformations) on data on each epoch.
    trainer.optimizer_name = "Adam" #Use Adam as an optimizer (can also be 'SGD', i.e stochastic gradient descend).
    trainer.target_names = ["vdens"] #The neural network will try to predict 'vdens' in the dataset.
    trainer.input_names = ["cdens"] #The neural network will use 'cdens' in the dataset.
    trainer.init() #Init the network with the new settings.
    trainer.train(150,batch_number=16,compute_validation=10,early_stopping=False) #Train the neural network for 150 epochs
    trainer.save() #Save the trainer and neural network
    trainer.plot(save=True) #Plot loss curves and residuals
    #trainer.plot_validation(save=True) #Plot validation images
    plot_models_accuracy([trainer], sigmas=(0,1,20), bins=[0,2,4,8], use_linestyles=True) #Plot accuracy of the model.
"""

script = TEMPLATE

#Utils block
from POLARIScore import utils, config, Logger
block_utils = "\n" + extract_non_import_code(Logger.__file__)
block_utils += "\n" + extract_non_import_code(config.__file__)
block_utils += "\n" + extract_non_import_code(utils.utils.__file__)
block_utils += "\n" + extract_non_import_code(utils.batch_utils.__file__)
block_utils += "\n" + extract_non_import_code(utils.physics_utils.__file__)
script = script.replace("{{UTILS_METHODS}}",block_utils)

#Architecture block
from POLARIScore import networks
block_architecture = extract_non_import_code(networks.utils.nn_utils.__file__)

for pyfile in Path(networks.addons.__file__).parent.glob("*.py"):
    if pyfile.name != "__init__.py":
        block_architecture += "\n" + extract_non_import_code(pyfile)

block_architecture += "\n" + extract_non_import_code(networks.architectures.nn_BaseModule.__file__) 

architecture_path = Path(networks.architectures.__file__).parent /  ("nn_"+args.arch+".py")
assert os.path.exists(architecture_path), LOGGER.error(f"No architecture found: {architecture_path} ")
block_architecture += "\n" + extract_functions_and_classes(architecture_path)
script = script.replace("{{ARCHITECTURE_CLASS}}", block_architecture)

#Dataset block
from POLARIScore.objects import Dataset
block_dataset = "\n" + extract_functions_and_classes(Dataset.__file__)
script = script.replace("{{DATASET_CLASS}}", block_dataset)

#Trainer block
from POLARIScore.networks import Trainer
block_trainer = "\n" + extract_functions_and_classes(Trainer.__file__)
if args.trainer is not None:
    trainer_path = Path(networks.__file__).parent / (args.trainer+"Trainer.py")
    assert os.path.exists(trainer_path), LOGGER.error(f"No trainer found at {trainer_path}")
    block_trainer += "\n" + extract_functions_and_classes(trainer_path)
script = script.replace("{{TRAINER_CLASS}}", block_trainer)

output_path = os.path.join(args.output,"training_script.py")
Path(output_path).write_text(script)

LOGGER.log(f"Success: Training script was generated at {os.path.join(args.output,"training_script.py")}")