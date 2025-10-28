import os
import numpy as np
import torch

"""
This file contains all utils variables, like for simulation, plots...
"""
SIM_DATA_NAME = "datacube.fits"
"""Name of the file where the simulation data is stored"""

EXPORT_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)),"export/")
"""Where all the objects saves are stored (ex: models, training_batchs)"""

TRAINING_BATCH_FOLDER = os.path.join(EXPORT_FOLDER,"training_batchs")
"""Path to the training batchs"""
MODEL_FOLDER = os.path.join(EXPORT_FOLDER,"models")
"""Path to the models folder"""
SPECTRA_FOLDER = os.path.join(EXPORT_FOLDER,"spectra")
"""Path to the spectra folder"""

RANDOM_BATCH_SCORE_offset = 1.
RANDOM_BATCH_SCORE_fct = lambda x: 1./(1+np.exp(-2*(x-RANDOM_BATCH_SCORE_offset)))
"""To generate batch, we use a score that'll go through this function. If a random number between 0. and 1. is lower that this function, then the generate training image is keeped. By default this is a sigmoid"""

import matplotlib.cm as cm
FIGURE_CMAP = cm.Dark2
FIGURE_CMAP_MIN = 0.
FIGURE_CMAP_MAX = 1.0

from .Logger import Logger
LOGGER = Logger(level=2, auto_save=0)

FIGURE_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)),"paper/figure/")

CACHES_FOLDER = os.path.join(EXPORT_FOLDER,"caches/")

DATA_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)),"data/")

OBSERVATIONS_FOLDER = os.path.join(DATA_FOLDER,"observations/")

#Normalization functions are tuples of (lambda,lambda). First one is to normalize, second one is the invert function to recover the physical scale
DATA_NORMALIZATION_CDENS = (lambda x: (np.log10(x)-19.)/6.*2.-1., lambda y: np.power(10.,(1.+y)/2*6.+19.) )
DATA_NORMALIZATION_VDENS = (lambda x: np.log10(x)/8.*2.-1., lambda y: np.power(10,(1.+y)/2*8.))
DATA_NORMALIZATION_CDENS_TORCH = (lambda x: (torch.log10(x)-19.)/6.*2.-1., lambda y: torch.pow(10.,(1.+y)/2*6.+19.) )
DATA_NORMALIZATION_VDENS_TORCH = (lambda x: torch.log10(x)/8.*2.-1., lambda y: torch.pow(10,(1.+y)/2*8.))