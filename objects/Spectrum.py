import os
import sys
if __name__ == "__main__":
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(parent_dir)
from POLARIScore.config import CACHES_FOLDER, LOGGER
import numpy as np
from POLARIScore.utils.physics_utils import *
from POLARIScore.utils.utils import *
import uuid
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from typing import Tuple, List, Union, Literal, Optional, cast
from POLARIScore.objects.tools.Graph import Graph, Node
from POLARIScore.objects.tools.Dendrogram import Dendrogram
import copy

def _gaussian(x, A, mu, sigma):
    return np.abs(A) * np.exp(-((x - mu)**2) / (2 * sigma**2))

def _gaussian_sum(x, params, N):
    y = np.zeros_like(x)
    for i in range(N):
        A, mu, sigma = params[3*i], params[3*i+1], params[3*i+2]
        y += _gaussian(x, A, mu, sigma)
    return y

def _chi_squared(params, x, y, N):
    y_model = _gaussian_sum(x, params, N)
    return np.sum((y - y_model)**2/(y_model+1e-8))

class Spectrum():
    """
    Object for spectra, this can also contains map of spectrum but a lot of functions will not work, use SpectrumMap instead. 
    """
    def __init__(self,spectrum:np.ndarray, name:bool=None):
        self.name = "spectrum_"+str(uuid.uuid4()) if name is None else name
        if not("spectrum" in self.name):
            self.namename = "spectrum_"+self.name
        self.X = None
        self.spectrum:np.ndarray = spectrum
        """np.ndarray : 1D tensor"""
        self.derivatives:Tuple[Union[np.ndarray,None],Union[np.ndarray,None]] = [None, None]

        self.dendro:Optional[Dendrogram] = None
        self.fit_settings:Optional[Tuple[np.ndarray,Dict]]=None

        self.host_map = None
        self.host_position = None

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

    def add_noise(self, SNR: float, seed: Optional[int] = None):
        """
        Add white Gaussian noise to the spectrum to reach a target SNR.
        """
        if seed is not None:
            np.random.seed(seed)

        signal = self.spectrum.astype(float)
        T_peak = np.max(signal)
        sigma_noise = T_peak / SNR
        noise = np.random.normal(loc=0.0, scale=sigma_noise, size=signal.shape)
        self.spectrum = signal + noise
        
        return self.spectrum

    def plot(self, ax=None, channels:Optional[np.ndarray]=None, show_fit:bool=False, show_fit_gaussians:bool=False, show_dendrogram:bool=True):
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        channels = self.get_X() if channels is None else channels

        ax.plot(channels,self.spectrum, color="black", label="data")
        ax.set_xlabel("Velocity [m/s]")
        ax.set_ylabel("Intensity [K]")

        if show_dendrogram:
            dendrogram = self.dendrogram()
            dendrogram.plot(ax=ax)

        if show_fit:
            if self.fit_settings is None:
                LOGGER.warn("Can't plot spectrum fit because there is no fit done. Launch self.fit() first.")
            else:
                y_fit, props = self.fit_settings
                gaussian_params = props['params']
                if y_fit is None:
                    y_fit = _gaussian_sum(channels, gaussian_params, props["N"])
                    self.fit_settings = y_fit, props
                ax.plot(channels, y_fit, 'r-', label=r'fit=$\sum^{'+str(props["N"])+r'}_i G_i$')
                if show_fit_gaussians:
                    colormap = plt.get_cmap("viridis")
                    for i in range(props['N']):
                        color = colormap((i+1) / (props['N']+1))
                        A, mu, sigma = gaussian_params[3*i],gaussian_params[3*i+1],gaussian_params[3*i+2]
                        g_fit = _gaussian(channels, A, mu, sigma)
                        ax.plot(channels, g_fit, color=color, label=rf"$G_{i}$")

        ax.grid()
        ax.legend()
        return fig, ax
    
    def get_X(self, output_settings:Optional[Dict]=None, force_compute:bool=False):
        if self.X is not None and not(force_compute):
            return self.X
        if output_settings is None:
            LOGGER.warn("Can't get x axis of spectrum because there is no output settings")
            return range(len(self.spectrum))
        for key in ["v_function","lsr_velocity","velocity_channels","velocity_resolution"]:
            if not(key in output_settings):
                LOGGER.warn(f"Can't get x axis of spectrum because there is no key {key} in output settings")
                return range(len(self.spectrum))
        self.X = output_settings["v_function"](output_settings["lsr_velocity"],output_settings["velocity_channels"],output_settings["velocity_resolution"])
        return self.X

    def get_derivatives(self, force_compute=False):
        if self.derivatives[0] is None or force_compute:
            self.derivatives[0] = np.gradient(self.spectrum)
        if self.derivatives[1] is None or force_compute:
            self.derivatives[1] = np.gradient(self.derivatives[0])
        return self.derivatives
    
    def get_borders(self, X:Optional[np.ndarray]=None, threshold:float=0.01)->Tuple[float, float]:
        Y = self.spectrum
        X = np.arange(len(Y), dtype=float) if X is None else X
        thr = threshold * np.max(Y)

        valid = np.where(Y > thr)[0]
        if len(valid) == 0:
            return X[0], X[-1]

        return valid[0], valid[-1]
        
    def dendrogram(self, force_compute:bool=False)->Dendrogram:

        if self.dendro is not None and not(force_compute):
            return self.dendro

        Y = self.spectrum
        X = self.get_X()

        def _dendro(x,y,graph:Optional[Dendrogram]=None,connect_node:Optional[Node]=None)->Dendrogram:
            x_mean = np.sum(x*y)/np.sum(y)
            
            if graph is None:
                graph = Dendrogram()

            if connect_node is not None:
                root_node = graph.add_node([x_mean, connect_node.position[1]])
                graph.add_edge(connect_node, root_node)
            else:
                root_node = graph.add_node([x_mean, np.min(y)]) 
            root_node.properties["x_borders"] = (np.min(x),np.max(x))
            
            dy = np.empty_like(y, dtype=float)
            dy[0] = 0.0 
            dy[1:] = (y[1:] - y[:-1])
            #ddy = np.gradient(dy)
            roots = find_roots(x, dy, interp=None)

            min_y = np.max(y)
            min_y_idx = np.argmax(y)
            is_a_minima = False
            for r in roots:
                if r == 0 or r == len(x)-1:
                    continue
                if not(y[r+1] > y[r] and y[r-1] > y[r]):
                    continue
                is_a_minima=True
                if y[r] < min_y:
                    min_y = y[r]
                    min_y_idx = r

            if is_a_minima:
                min_node = graph.add_node([x[min_y_idx],min_y])
                graph.add_edge(root_node,min_node)
                root_node.properties["is_root"] = True
                _dendro(x[0:min_y_idx],y[0:min_y_idx],graph=graph,connect_node=min_node)
                _dendro(x[min_y_idx:],y[min_y_idx:],graph=graph,connect_node=min_node)
            else:
                leaf_node = graph.add_node([x[min_y_idx],y[min_y_idx]])
                leaf_node.properties["is_leaf"] = True
                leaf_node.properties["x_borders"] = (np.min(x),np.max(x))
                graph.add_edge(root_node, leaf_node)

            return graph

        i1, i2 = self.get_borders(X=X)
        dendro = _dendro(X[i1:i2+1], Y[i1:i2+1])
        self.dendro = dendro

        return dendro
    
    def fit(self, method:Literal['minimal','dendrogram','clean','iterative'], force_compute:bool=False, **args)->Tuple[np.ndarray, Dict]:
        X = self.get_X()
        Y = self.spectrum

        if (self.fit_settings is not None 
            and 'method' in self.fit_settings[1] and self.fit_settings[1]['method'] == method
            and not(force_compute)):
            return self.fit_settings

        if method=="minimal":
            y_fit, props = self.fit_minimize(**args)
        elif method=="dendrogram":
            y_fit, props = self.fit_dendrogram(**args)
        elif method=="iterative":
            y_fit, props = self.fit_iterative(**args)
        
        props['method'] = method
        self.fit_settings = (y_fit, props)

        return self.fit_settings
    
    
    def fit_iterative(self, distance:int=2, max_iteration:int=10, early_stop:bool=True):
        LOGGER.log("Iterative fitting launched")
        assert self.host_map is not None, LOGGER.error("The iterative fitting method need the spectrum to be part of a spectrum map.")
        assert self.host_position is not None, LOGGER.error("The iterative fitting method requires the spectrum to have a position.")
        x_min = max(0, self.host_position[0]-distance)
        x_max = min(len(self.host_map.map)-1,self.host_position[0]+distance+1)
        y_min = max(0, self.host_position[1]-distance)
        y_max =  min(len(self.host_map.map[0])-1,self.host_position[1]+distance+1)
        s_map=self.host_map.get_spectra(map=self.host_map.map[x_min:x_max, y_min:y_max])
        s_map = cast(List[List[Spectrum]], s_map)

        pos_X, pos_Y = self.host_position[0] - x_min, self.host_position[1] - y_min
        
        for i in range(len(s_map)):
            for j in range(len(s_map[0])):
                spectrum:Spectrum = s_map[i][j]
                if spectrum.fit_settings is None:
                    spectrum.fit("dendrogram")
        
        def _get_fit_props(key="N"):
            props = []
            for i in range(len(s_map)):
                props.append([])
                for j in range(len(s_map[0])):
                    props[i].append(s_map[i][j].fit_settings[1][key])
            return props
        
        iteration = 0   
        scores = [[np.inf for __ in range(len(s_map[0]))] for _ in range(len(s_map))]
        scores = np.array(scores)
        
        
        a1=lambda x: x*1.
        a2=lambda x: x*0.1
        a3=lambda x: x*0.1

        
        mean_scores = []
        start_time = time.time()
        while iteration < max_iteration:
            iteration_time = time.time()
            temp_s_map = copy.deepcopy(s_map)

            component_matrix = np.array(_get_fit_props("N"))
            chi_matrix = np.array(_get_fit_props("CHI"))
            print(component_matrix)
            for i in range(len(s_map)):
                for j in range(len(s_map[0])):
                    printProgressBar(i*len(s_map)+j, len(s_map)*len(s_map[0]), prefix="Iterative fitting", length=30)
                    spectrum:Spectrum = temp_s_map[i][j]

                    X = spectrum.get_X()
                    Y = spectrum.spectrum

                    N_target = 0
                    N_target_sum = 0
                    for mi in range(len(component_matrix)):
                        for mj in range(len(component_matrix[0])):
                            if mi == i and mj == j:
                                continue
                            dist = np.sqrt((i-mi)**2+(j-mj)**2)
                            N_target += component_matrix[mi,mj]/chi_matrix[mi,mj]/dist
                            N_target_sum += 1./dist/chi_matrix[mi,mj]
                    N_target = N_target/N_target_sum
                    F = N_target-component_matrix[i,j]
                    N_target = max(int(N_target),1)

                    target_gauss_index:int = np.argsort((component_matrix-N_target).flatten())[0]
                    target_gauss_index_i, target_gauss_index_j = target_gauss_index // len(component_matrix), target_gauss_index % len(component_matrix)
                    

                    _, target_gauss_props = s_map[target_gauss_index_i][target_gauss_index_j].fit_dendrogram()

                    _, dendro_props = spectrum.fit_dendrogram()
                    guess_params:List = dendro_props['params']
                    if isinstance(guess_params, np.ndarray):
                        guess_params = guess_params.tolist() 
                    if dendro_props['N'] > N_target:
                        guess_params = guess_params[3*int(np.abs(F)):]
                    elif dendro_props['N'] < N_target:
                        if target_gauss_props['N'] == N_target:
                            guess_params = target_gauss_props['params']
                        else:
                            for _ in range(abs(int(dendro_props['N']-N_target))):
                                guess_params.extend(guess_params[:3])

                    res = minimize(_chi_squared, guess_params, args=(X, Y, N_target), method='L-BFGS-B')
                    chi2 = _chi_squared(res.x, X, Y, N_target)
                    new_y_fit = _gaussian_sum(X, res.x, N_target)
                    new_props = {"params":res.x,"N":N_target,"CHI":chi2}

                    score = a1(chi2)+a2(component_matrix[i,j])+a3(0)
                    if score < scores[i,j]:
                        spectrum.fit_settings = (new_y_fit, new_props)
                    scores[i,j] = score if scores[i,j] > score else scores[i,j] 
            s_map = temp_s_map           
            
            iteration += 1
            actual_time = time.time()
            iteration_time = actual_time-iteration_time
            time_left = (actual_time-start_time)/(iteration)*(max_iteration-(iteration))
            LOGGER.print(f'Iteration {iteration}/{max_iteration} | Elapsed: {format_time(actual_time-start_time)} | Time Left: {format_time(time_left)} | Mean score: {np.mean(scores):.2}(Chi:{np.mean(chi_matrix):.2},N:{np.mean(component_matrix)})', type="SpectrumFit", level=1, color="34m")
            #print(component_matrix, chi_matrix)
            mean_scores.append(np.mean(scores))
            if (early_stop and len(mean_scores) >= 3 and 
                ((len(mean_scores)-1)-np.argmin(mean_scores) > 3 or np.abs(np.gradient(mean_scores)[-1])/mean_scores[-1] < 0.05)): #Early stopped if changes are lows
                LOGGER.print(f'Fitting early stopped.')
                break

        LOGGER.log(f"Fit done with {component_matrix[pos_X][pos_Y]} components and chi={chi_matrix[pos_X][pos_Y]:.2f}")
        return s_map[pos_X][pos_Y].fit_settings


    def fit_dendrogram(self, only_leaves:bool=False):

        X = self.get_X()
        Y = self.spectrum

        dendro = self.dendrogram()
        nodes = dendro.get_leaves()[1]
        if not(only_leaves):
            roots = dendro.get_roots()[1]
            nodes.extend(roots)
        number_components = len(nodes)
        gauss_means = [n.position[0] for n in nodes]
        gauss_amps = [n.position[1] for n in nodes]
        gauss_sigmas = []
        for i,n in enumerate(nodes):
            assert 'x_borders' in n.properties, LOGGER.error("A node don't have the x domain in her properties.")
            x_lims:Tuple[float, float] = n.properties['x_borders']
            i0 = np.abs(X - x_lims[0]).argmin()
            i1 = np.abs(X - x_lims[1]).argmin()
            x_segment = X[min(i0, i1):max(i0, i1)]
            y_segment = Y[min(i0, i1):max(i0, i1)]
            std = np.sqrt(np.sum(y_segment * (x_segment - gauss_means[i])**2) / np.sum(y_segment))
            gauss_sigmas.append(std)

        guess = []
        for i in range(number_components):
            guess.extend([gauss_amps[i], gauss_means[i], gauss_sigmas[i]])
        
        res = minimize(_chi_squared, guess, args=(X, Y, number_components), method='L-BFGS-B')

        #Clean not used gaussian components
        res.x = res.x.tolist()
        components_removed = 0
        for i in range(number_components):
            index = i-components_removed
            A, mu, sigma = res.x[3*index], res.x[3*index+1], res.x[3*index+2]
            if A < 0.01*np.max(Y) or sigma > .5*(np.max(X)-np.min(X)):
                res.x.pop(3*index)
                res.x.pop(3*index)
                res.x.pop(3*index)
                components_removed += 1
        number_components -= components_removed
        res.x = np.array(res.x)
        if components_removed > 0:
            res = minimize(_chi_squared, res.x, args=(X, Y, number_components), method='L-BFGS-B')


        chi2 = _chi_squared(res.x, X, Y, number_components)
        y_fit = _gaussian_sum(X, res.x, number_components)
        props = {"params":res.x,"N":number_components,"CHI":chi2}

        #print(chi2, res.x)

        return y_fit, props


    def fit_minimize(self, max_components:int=10, score_threshold:float=50)->Tuple[np.ndarray, Dict]:
        Y = self.spectrum
        X = self.get_X()
                
        best_result = None
        results = []
        best_score = np.inf
        best_chi2 = np.inf

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
                    best_chi2 = chi2
                    best_result = (N, res)
                if best_score < score_threshold:
                    break

            N_best, res = best_result
            y_fit = _gaussian_sum(X, res.x, N_best)
            p_res = res.x
        else:
            y_fit = X*0.
            N_best = 0.
            
        props = {"params":p_res,"N":N_best,"CHI":best_chi2}
        return y_fit, props

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