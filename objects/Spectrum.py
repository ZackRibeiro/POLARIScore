import os
import sys
if __name__ == "__main__":
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(parent_dir)
from POLARIScore.config import CACHES_FOLDER, LOGGER
import numpy as np
from POLARIScore.utils.physics_utils import *
import uuid
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from typing import Tuple, List, Union

class Spectrum():
    """
    Object for spectra, this can also contains map of spectrum but a lot of functions will not work, use SpectrumMap instead. 
    """
    def __init__(self,spectrum:np.ndarray, name:bool=None):
        self.name = "spectrum_"+str(uuid.uuid4()) if name is None else name
        if not("spectrum" in self.name):
            self.namename = "spectrum_"+self.name
        self.spectrum:np.ndarray = spectrum
        """np.ndarray : 1D tensor"""
        self.derivatives:Tuple[Union[np.ndarray,None],Union[np.ndarray,None]] = [None, None]

    def save(self,folder:str=None, replace:bool=False, log:bool=True):
        folder = CACHES_FOLDER if folder is None else folder
        if not(os.path.exists(folder)):
            os.mkdir(folder)
        path = os.path.join(folder,self.name+".npy")
        if os.path.exists(path):
            if not(replace):
                LOGGER.error(f"Can't save spectrum {self.name} because there is already a spectrum called this way in the folder and replace is set to False")
                return
            os.remove(path)
        if log:
            LOGGER.log(f"Spectrum {self.name} saved")
        np.save(path,self.spectrum)

    def plot(self, ax=None, channels=None):
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        channels = np.arange(len(self.spectrum)) if channels is None else channels
        ax.plot(channels,self.spectrum)
        ax.set_xlabel("Velocity [m/s]")
        ax.set_ylabel("Intensity [K]")
        ax.grid()
        return fig, ax
    
    def getX(self, output_settings):
        if output_settings is None:
            LOGGER.error("Can't get x axis of spectrum because there is no output settings")
            return
        for key in ["v_function","lsr_velocity","velocity_channels","velocity_resolution"]:
            if not(key in output_settings):
                LOGGER.error(f"Can't get x axis of spectrum because there is no key {key} in output settings")
                return
        return output_settings["v_function"](output_settings["lsr_velocity"],output_settings["velocity_channels"],output_settings["velocity_resolution"])

    def get_derivatives(self, force_compute=False):
        if self.derivatives[0] is None or force_compute:
            self.derivatives[0] = np.gradient(self.spectrum)
        if self.derivatives[1] is None or force_compute:
            self.derivatives[1] = np.gradient(self.derivatives[0])
        return self.derivatives

    def fit(self, max_components=10, ax=None, score_threshold=50, X=None):
        Y = self.spectrum
        X = np.arange(len(Y), dtype=float) if X is None else X

        def _gaussian_sum(x, params, N):
            y = np.zeros_like(x)
            for i in range(N):
                A, mu, sigma = params[3*i], params[3*i+1], params[3*i+2]
                y += np.abs(A) * np.exp(-((x - mu)**2) / (2 * sigma**2))
            return y

        def _chi_squared(params, x, y, N):
            y_model = _gaussian_sum(x, params, N)
            return np.sum((y - y_model)**2/(y_model+1e-8))
                
        best_result = None
        results = []

        best_score = np.inf


        p_res = []
        if np.sum(Y) > 1e-5:
            for N in range(1, max_components+1):
                guess = []
                for _ in range(N):
                    guess.extend([max(Y)/2, X[int(len(X)/2)], np.random.uniform(10,100)])
                
                res = minimize(_chi_squared, guess, args=(X, Y, N), method='L-BFGS-B')
                k = len(res.x)
                chi2 = _chi_squared(res.x, X, Y, N)
                score = 2.*k + chi2
                results.append((N, res, score))
                if score < best_score:
                    best_score = score
                    best_result = (N, res)
                if best_score < score_threshold:
                    break

            N_best, res = best_result
            y_fit = _gaussian_sum(X, res.x, N_best)
            p_res = res.x
        else:
            y_fit = X*0.
            N_best = 0.
            

        if not(ax is None):
            ax.plot(X, y_fit, 'r-', label=f'Fit (N={N_best})')
            ax.legend()
            ax.set_title(f'N = {N_best}')

        return (N_best, p_res)

def loadSpectrum(name, folder=None, absolute_path=None):
    folder = CACHES_FOLDER if folder is None else folder
    path = os.path.join(folder,name.split(".npy")[0]+".npy")
    if not(absolute_path is None):
        path = absolute_path
    if not(os.path.exists(path)):
        LOGGER.error(f"Can't load spectrum because the file is not found: {path}")
        return 
    return Spectrum(spectrum=np.load(path))

def _method_getMoment(args, m=0):
    data = np.array(args["data"])
    output_settings = args["output"]
    X = output_settings["v_function"](output_settings["lsr_velocity"],output_settings["velocity_channels"],output_settings["velocity_resolution"])
    moment = 0
    for i,d in enumerate(data):
        moment += np.power(X[i],m)*d
    moment /= len(data)
    return moment

def _method_getComponentsNumber(args):
    spectrum = Spectrum(args["data"])
    N, _ = spectrum.fit(max_components=7)
    return N