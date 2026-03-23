import os
import sys
from POLARIScore.config import CACHES_FOLDER, LOGGER, SPECTRA_FOLDER, FIGURE_FOLDER
import numpy as np
from POLARIScore.utils.utils import *
from POLARIScore.utils.physics_utils import *
from POLARIScore.objects.Raycaster import ray_mapping
from POLARIScore.objects.Spectrum import Spectrum
import json
import shutil
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.widgets import Slider
import multiprocessing as mp
from functools import partial
from typing import Literal, List, Tuple, Optional
from sklearn.decomposition import PCA
from POLARIScore.objects.Dataset import Dataset
import copy

def _output_v_function(lsr,chan,res):
    return lsr+(np.array(range(chan))-chan/2)*res
DEFAULT_OUTPUT_SETTINGS = {
    "velocity_channels": 128,
    "velocity_resolution": 1e3*0.05,
    "lsr_velocity": 0,
    "v_function": _output_v_function,
}

#Line settings example for 13CO J=U-L
_L = 0
_U = 1
DEFAULT_LINE_SETTINGS = {
    "l":_L,
    "u":_U,
    "abundance":CO13_ABUNDANCE,
    "temp_low":ROT_ENERGY(_L,CO13_ROT_CST),
    "temperature":ROT_ENERGY(_U,CO13_ROT_CST)-ROT_ENERGY(_L,CO13_ROT_CST),
    "frequency":CO13_FREQUENCY[_U-1],
    "estein_emission":CO13_A[_U-1]
}
"""Default line settings used for generate emission map, used in basic radiative transfer equations"""

DEFAULT_GLOBAL_SETTINGS = {
    "density_threshold": 300,
    "with_turbulence": True,
}
"""Default global settings when emission maps are generated"""

def _worker_get_gaussians_params(job:Dict)->Tuple[int, np.ndarray]:
    index, args = job
    max_gaussian_components = args["extra_args"]["max_gaussian_components"]
    fit_method = args["extra_args"]["fit_method"]
    spectrum = Spectrum(spectrum=args['data'])
    spectrum.host_position=(args['x'],args['y'])
    spectrum.get_X(output_settings=args["output"])
    spectrum.fit(fit_method)
    _, props= spectrum.fit_settings
    if len(props['params']) > max_gaussian_components*3:
        gaussian_params = props['params'][:max_gaussian_components*3]
    else:
        extra = [0 for _ in range(max_gaussian_components*3-len(props['params']))]
        new_params = props['params'].tolist()
        new_params.extend(extra)
        props['params'] = np.array(new_params)
        gaussian_params = props['params']
    return index, gaussian_params

def _worker_ray_radiative_transfer(cube_dimension, position, direction, last_step, extra_args):
    
    if np.linalg.norm(position) == 0:
        position = 0
    else:
        #position = int(np.floor(
        #    abs(cube_dimension*np.dot(position, direction)/np.linalg.norm(position))
        #    ))

        axis = np.argmax(np.abs(direction))
        position = int(position[axis])

        position = max(min(position, cube_dimension-1),0)
    
    if type(extra_args['TEMP']) is float:
        temperature = extra_args['TEMP']
    else:
        temperature = extra_args['TEMP'][position]

    line_settings = extra_args['line_settings']
    global_settings = extra_args['global_settings']
    output_settings = extra_args['output_settings']

    g_l = 2*line_settings["l"]+1
    g_u = 2*line_settings["u"]+1
    V = output_settings["v_function"](output_settings["lsr_velocity"],output_settings["velocity_channels"],output_settings["velocity_resolution"])

    def _get_velocity(pos):
        if 0 > pos or pos >= cube_dimension:
            return _get_velocity(position)[0], False
        v = np.array([extra_args['VX1'][pos],extra_args['VX2'][pos],extra_args['VX3'][pos]])
        return np.dot(v,direction)*1e-2, True #maybe a -1 is needed for the dot prod

    density = extra_args['RHO'][position]
    velocity, _ = _get_velocity(position)

    if not("intensity_spectrum" in last_step):
        intensity_spectrum = BLACKBODY_EMISSION((V/LIGHT_SPEED+1)*line_settings["frequency"],CMB_TEMPERATURE)
    else:
        intensity_spectrum = last_step["intensity_spectrum"]

    if density < global_settings["density_threshold"]:
        return {"intensity_spectrum": intensity_spectrum}


    sigma_doppler = 0.08*1e3*np.sqrt(temperature/20)
    sigma_turb = 0
    if global_settings["with_turbulence"]:
        vm1,fm1 = _get_velocity(int(position-1))
        vp1,fp1 = _get_velocity(int(position+1))
        sigma_turb = 0.5*(vp1-vm1)/np.sqrt(12) if fm1 and fp1 else vp1-vm1
    sigma = np.sqrt(sigma_doppler**2 + sigma_turb**2)

    #TODO Change the fct partition approximation, this is an approximation valid just for CO J=1-0
    low_density_col = 1e4*density*extra_args['cell_size']*line_settings["abundance"]*g_l*np.exp(-line_settings["temp_low"]/(temperature))/(1/3+2*temperature/5.5)
    tau0 = LIGHT_SPEED**3/(8*np.pi*line_settings["frequency"]**3)*line_settings["estein_emission"] * g_u/g_l * (1-np.exp(-line_settings["temperature"]/temperature)) * low_density_col
    tau = tau0 * GAUSSIAN(x=V-velocity,sigma=sigma)

    tau_exp = np.exp(-tau)
    intensity_spectrum = intensity_spectrum*tau_exp+BLACKBODY_EMISSION(nu=line_settings["frequency"],T=temperature)*(1-tau_exp) 

    result = {
        "intensity_spectrum": intensity_spectrum,
    }
    return result

#TODO As for training sets, make this memory less by opening the spectra just when it is needed
class SpectrumMap():
    """
    Map (matrix NxN) of spectra. Each element of the matrix contains a list of values (which can be passed easily to Spectra object).
    """
    def __init__(self, name:str,map:np.ndarray=None, load:bool=True):

        self.name:str = name

        self.map:np.ndarray = map
        
        self.line_settings:Dict = DEFAULT_LINE_SETTINGS
        self.output_settings:Dict = DEFAULT_OUTPUT_SETTINGS
        self.global_settings:Dict = DEFAULT_GLOBAL_SETTINGS

        if load:
            self.load()

    def get_spectra(self, map=None,pos:Optional[Tuple[int,int]]=None)->Union[List[List[Spectrum]],Spectrum]:
        """
        Returns a 2D list with a spectrum for each pixel.
        """
        if pos is None:
            map = self.map if map is None else map
        else:
            map = self.map[pos[0],pos[1]]

        formatted_intensity_map = []
        def make_spectrum(array,x:Optional[int]=None,y:Optional[int]=None):
            x = pos[0] if x is None else x
            y = pos[1] if y is None else y
            s_name = f'x{x}_y{y}'
            S = Spectrum(name=s_name, spectrum=array)
            S.host_map = self
            S.host_position=(x,y)
            S.get_X(self.output_settings)
            return S
        for x in range(len(map)):
            formatted_intensity_map.append([])
            if isinstance(map[x],float):
                assert pos is not None
                return make_spectrum(map)
            for y in range(len(map[x])):
                #printProgressBar(x*len(map[x])+y,len(map)*len(map[x]),prefix="Format map", length=10)
                spectrum = map[x,y]
                if isinstance(spectrum, Spectrum):
                    return map
                s_name = f'x{x}_y{y}'
                S = Spectrum(name=s_name, spectrum=spectrum)
                S.host_map = self
                S.host_position=(x,y)
                S.get_X(self.output_settings)
                formatted_intensity_map[x].append(S)
        return formatted_intensity_map

    def load(self, name=None):
        name = self.name if name is None else name
        folder = os.path.join(SPECTRA_FOLDER, name)
        if not(os.path.exists(folder)):
            LOGGER.error(f"Can't load spectrum map {self.name} because there is no folder named this way.")
            return None
    
        global_settings = {}
        if os.path.exists(os.path.join(folder,'global_settings.json')):
            with open(os.path.join(folder,'global_settings.json')) as file:
                global_settings = json.load(file)
        else:
            LOGGER.warn("No global settings json found in the spectrum map folder -> Using the default one")
        self.global_settings = self.global_settings | global_settings
        line_settings = {}
        if os.path.exists(os.path.join(folder,'line_settings.json')):
            with open(os.path.join(folder,'line_settings.json')) as file:
                line_settings = json.load(file)
        else:
            LOGGER.warn("No line settings json found in the spectrum map folder -> Using the default one")
        self.line_settings = self.line_settings | line_settings
        output_settings = {}
        if os.path.exists(os.path.join(folder,'output_settings.json')):
            with open(os.path.join(folder,'output_settings.json')) as file:
                output_settings = json.load(file)
        else:
            LOGGER.warn("No output settings json found in the spectrum map folder -> Using the default one")
        self.output_settings = self.output_settings | output_settings


        spectra_file = os.path.join(folder, "spectrum.npy")
        if not(os.path.exists(spectra_file)):
            LOGGER.error(f"Can't load spectrum map {self.name} because there is no spectrum.npy.")
            return None
        self.map = np.load(spectra_file, mmap_mode='r')

        return self
    
    def save(self, name=None, replace=True):
        name = self.name if name is None else name
        self.name = name
        if self.map is None:
            LOGGER.error("Can't save a map when there is nothing in it.")
            return None
        if not(os.path.exists(SPECTRA_FOLDER)):
            os.mkdir(SPECTRA_FOLDER)
        folder = os.path.join(SPECTRA_FOLDER, name)
        if os.path.exists(folder):
            if replace:
                LOGGER.warn(f"A previous spectrum map named similar was removed.")
                shutil.rmtree(folder)
            else:
                LOGGER.error(f"Can't save spectrum map {self.name} because there is already a spectrum map called this way and replace is set to False")
                return None
        if not(os.path.exists(folder)):
            os.mkdir(folder)

        np.save(os.path.join(folder, 'spectrum.npy'), self.map)

        with open(os.path.join(folder,'global_settings.json'), 'w') as file:
            json.dump(self.global_settings, file, indent=4)
        with open(os.path.join(folder,'line_settings.json'), 'w') as file:
            json.dump(self.line_settings, file, indent=4)
        with open(os.path.join(folder,'output_settings.json'), 'w') as file:
            #TODO serialize lambda function
            fct = self.output_settings["v_function"]
            del self.output_settings["v_function"] 
            json.dump(self.output_settings, file, indent=4)
            self.output_settings["v_function"] = fct

        LOGGER.log(f"Spectrum map {self.name} saved.")

    def getIntegratedIntensity(self):
        """
        Returns:
            map: Map with the sum of the spectra
        """
        return np.sum(self.map, axis=2)
    
    def generate_dataset(self,name:str=None,
                         what_to_compute:Dict={"gaussians":10,},
                         number:int=100, snr:Optional[Tuple[float, float]]=[7, 20]
                         ,environment:int=1
                         )->'Dataset':
        """
        Method to generate a dataset from spectrum map.

        What is computed (and so can be used to denormalize data):
        - 'x channels': in m/s
        - 'spectrum': in real intensity unit
        - 'noisy_spectrum': in real intensity unit
        - 'amplitude': maximum amplitude of the spectrum without noise

        What can be computed:
        - 'gaussians': fit the spectrum and save gaussian parameters (normalized between -1 and 1)
        - 'noisy_spectrum' if snr is not None

        Args:
            number(int, default: 100): How many spectra do we want.
            what_to_compute(dict)
            environment: int, if not 0, the data is instead of shape [environment*2+1,environment*2+1,spectra_dim] so H,W,D
            snr: Tuple of float, add white noise to achieve a random snr in the given tuple range.
        Returns:
            dataset: the new dataset.
        """
        order = ["channels", "spectrum", "snr", "amplitude"]
        if "gaussians" in what_to_compute and what_to_compute["gaussians"] is not None and what_to_compute["gaussians"] > 0:
            order.append("gaussians_amplitudes")
            order.append("gaussians_means")
            order.append("gaussians_sigmas")
        if snr is not None:
            order.append("noisy_spectrum")

        name = self.name if name is None else name

        ds = Dataset()
        ds.name = name
        ds.settings = {"order": order}

        spectra_generated = 0
        pos_explored = []
        iteration = 0
        while spectra_generated < number and iteration < number*100:
            iteration += 1
            printProgressBar(spectra_generated, number, prefix=f"Building dataset ({iteration})")
            if iteration >= number*100:
                LOGGER.warn("Failed to generated all the requested random spectras, nbr of spectra generated:"+str(spectra_generated))
                break

            x, y = np.floor(np.random.random()*(len(self.map)-environment*2)+environment), np.floor(np.random.random()*(len(self.map[0])-environment*2)+environment)
            x, y = int(x), int(y)
            if (x,y) in pos_explored:
                continue

            spectra:List[List[Spectrum]] = self.get_spectra(map=self.map[x-environment:x+environment+1,y-environment:y+environment+1:])
            
            if snr is not None:
                random_snr = np.random.random()*(snr[1]-snr[0])+snr[0]
            else:
                random_snr = 0
            
            
            clean_spectra = []
            for xi in range(len(spectra)):
                clean_spectra.append([])
                for yi in range(len(spectra)):
                    clean_spectra[xi].append(spectra[xi][yi].spectrum)
            clean_spectra = np.array(clean_spectra)

            if len(clean_spectra) == 1 and len(clean_spectra[0]) == 1:
                clean_spectra = clean_spectra[0][0]
            
            max_amplitude = np.max(clean_spectra)
            b = [np.array(spectra[0][0].get_X())
                 , clean_spectra/max_amplitude
                 , random_snr
                 , max_amplitude]

            if "gaussians_amplitudes" in order:
                try:
                    _, gaussian_parameters = _worker_get_gaussians_params(job=(1,{
                        "data": self.map[x][y],
                        "x": x,
                        "y": y,
                        "output": self.output_settings,
                        'extra_args':{'max_gaussian_components': what_to_compute["gaussians"],
                        'fit_method':'dendrogram'}
                    }))
                except:
                    continue
                if np.isnan(gaussian_parameters).any():
                    continue
                gaussian_parameters = np.array(gaussian_parameters)
                gaussian_amplitudes = gaussian_parameters[0::3]/max_amplitude
                gaussian_means = gaussian_parameters[1::3]/np.max(np.abs(b[0]))
                gaussian_sigmas = gaussian_parameters[2::3]/np.max(np.abs(b[0]))
                b.append(gaussian_amplitudes)
                b.append(gaussian_means)
                b.append(gaussian_sigmas)

            
            if random_snr > 0:
                noisy_spectra = []
                for xi in range(len(spectra)):
                    noisy_spectra.append([])
                    for yi in range(len(spectra)):
                        noisy_spectra[xi].append(spectra[xi][yi].add_noise(random_snr))
                noisy_spectra = np.array(noisy_spectra)/max_amplitude
                if len(noisy_spectra) == 1 and len(noisy_spectra[0]) == 1:
                    noisy_spectra = noisy_spectra[0][0]
                b.append(noisy_spectra)
            
            ds.save_batch(b, spectra_generated)
            del b
            pos_explored.append((x,y))
            spectra_generated += 1

        settings = {
            "order": order,
            "what_was_computed": what_to_compute,
            "spectra_number": spectra_generated
        }

        #TODO, handle the error
        ds.settings = settings
        try:
            ds.save_settings()
        except:
            del settings["what_was_computed"]
            ds.save_settings()

        LOGGER.log(f"New dataset {ds.name} saved")
        LOGGER.reset()

        return ds

    def gaussians(self, max_gaussian_components=10, fit_method:str="dendrogram"):
        """
        Apply gaussian fit on each spectrum of the cube.
        Returns a cube of the shape: W, H, max_gaussian_components*3 containing the gaussians parameters in order: A, mean, std...
        """
        #LOGGER.global_color = LOGGER._init_gc
        #LOGGER.border("Spectra-fitting", level=1)
        #LOGGER.log(f"Fit method {fit_method} on map")
            
        return self.compute(method=_worker_get_gaussians_params,used_cpu=1., stride=1, 
                            extra_args={'max_gaussian_components': max_gaussian_components, 'fit_method':fit_method})
        
    def pca(self, plot:bool=False, return_cube:bool=True):
        """
        Apply principal component analysis on the hyperspectra cube.
        Returns components, variances and scores if return_cube is False else return scores reshaped to have the same shape of initial cube. 
        """
        assert self.map is not None, LOGGER.error("Spectrum map is empty.")
        nx, ny, nv = self.map.shape
        data = self.map.reshape(nx*ny,nv)
        data_mean = np.mean(data, axis=0)
        data_centered = data# - data_mean
        pca = PCA()
        pca.fit(data_centered)

        components = pca.components_ #evectors: (ncomp, nv)
        variance = pca.explained_variance_ratio_
        scores = pca.transform(data_centered)

        if plot:
            ncomp = components.shape[0]

            comp_index = 0

            eigenvector = components[comp_index]
            eigenimage = scores[:, comp_index].reshape(nx, ny)

            fig, (ax_spec, ax_img) = plt.subplots(1, 2, figsize=(12, 5))
            plt.subplots_adjust(bottom=0.25)

            line, = ax_spec.plot(eigenvector)
            ax_spec.set_title(f"Eigenvector {comp_index}")
            ax_spec.set_xlabel("Velocity Channel")
            ax_spec.set_ylabel("Amplitude")

            im = ax_img.imshow(eigenimage, origin='lower', cmap='RdBu_r')
            ax_img.set_title(f"Projection {comp_index}")
            fig.colorbar(im, ax=ax_img, fraction=0.046, pad=0.04)

            ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
            slider = Slider(ax_slider,"Component",0,ncomp - 1,valinit=0,valstep=1)
            fig._c_slider = slider

            def update(val):
                i = int(slider.val)

                line.set_ydata(components[i])
                ax_spec.set_title(f"Eigenvector {i}")

                new_img = scores[:, i].reshape(nx, ny)
                im.set_data(new_img)
                im.set_clim(vmin=np.min(new_img), vmax=np.max(new_img))
                ax_img.set_title(f"Projection {i}")

                fig.canvas.draw_idle()

            slider.on_changed(update)

        if return_cube:            
                return scores.reshape(nx, ny, nv)
        return components, variance, scores

    def compute(self, method, save=True, used_cpu=1., stride=1, extra_args:Dict={}):
        """
        Compute "method" over the sprectra map, i.e each spectrum in the map is processed using method.

        Args:
            method(function): Method to compute, need to have only one parameter: a dict with the spectrum used as the key "data".
            save(bool, default:True): When finished, save the result as cache.
            used_cpu(float, default:1.): percent (/100) of cpu cores used, 1.=100%, 0.= 1 core/no multiprocessing.
            stride(int, default:1): compute with a step in the map of stride value, if =1 then all spectra are processed.
        Returns:
            map: shape of self.map//stride containing the result of method
        """
        LOGGER.global_color = LOGGER._init_gc
        LOGGER.border("Spectrum-Computing", level=1)
        LOGGER.log(f"Compute method {method.__name__} on map")

        stride = int(stride)

        used_cpu = max(used_cpu,1.)

        jobs = [
            (i, {
                "data": self.map[x][y],
                "x": x,
                "y": y,
                "output": self.output_settings,
                "extra_args": extra_args
            })
            for i, (y, x) in enumerate(
                [(y, x)
                for y in range(0, len(self.map[0]), stride)
                for x in range(0, len(self.map), stride)]
            )
        ]
        
        total_jobs = len(jobs)
        results = [None] * total_jobs
        completed = 0

        with mp.Pool(int(np.ceil(mp.cpu_count()*used_cpu))) as pool:
            worker_func = method
            for index, spectrum_result in pool.imap_unordered(worker_func, jobs):
                results[index] = spectrum_result
                completed += 1
                printProgressBar(completed, total_jobs, prefix="Computing", length=30)
        LOGGER.reset()
        results = np.array(results).reshape((len(self.map),len(self.map),len(results[0])))

        if save:
            if not(os.path.exists(CACHES_FOLDER)):
                os.mkdir(CACHES_FOLDER)
            try:
                path = os.path.join(CACHES_FOLDER, self.name+f"_{method.__name__}_cache.npy")
                if os.path.exists(path):
                    os.remove(path)
                    LOGGER.warn("A previous cache for this spectrum map and method was found and removed.")
                np.save(path,results)
            except:
                LOGGER.error("Can't save the result of compute operation in cache.")

        return results

    def generate(self, simulation=None, axis:Optional[int]=None, force_compute:bool=False, method:Literal["ray","vectorized"]="ray"):

        LOGGER.global_color = LOGGER._init_gc
        LOGGER.border("SPECTRUM-GENERATING", level=1)

        for key in ['VX1','VX2','VX3','RHO','TEMP']:
            assert key in simulation.data, LOGGER.error(f"Can't generate spectra -> Simulation has no {key}.")
        TEMPERATURE = simulation.data['TEMP']
        if type(simulation.data['TEMP']) is float:
            LOGGER.warn(f"Temperature is not an array -> Uniform temperature of {TEMPERATURE}")
            

        if not(self.map is None) and not(force_compute):
            LOGGER.log("Intensity is already computed, use force_compute=True to recompute the map")
            return self.map

        if simulation is None:
            if "simulation_name" in self.global_settings and not(self.global_settings["simulation_name"] is None):
                LOGGER.warn("Simulation is loaded using the settings, this can give an error if the simulation can't be opened by the easy way.")
                from .Simulation_DC import Simulation_DC
                simulation = Simulation_DC(name=self.global_settings["simulation_name"], init=True)
            else:
                LOGGER.error(f"Can't compute spectrum map because there is no simulation specified.")
                return None
            
        if axis is None:
            LOGGER.warn("Axis is not defined, setting it to 0 (face XY)")
            axis = 0

        LOGGER.log(f"Computing spectrum map for simulation {simulation.name} and for face/axis: {axis}")

        if method == "ray":
            
            results = ray_mapping(simulation, _worker_ray_radiative_transfer, axis=axis, region=[0,-1,0,-1], extra_args=
                                    {"line_settings": self.line_settings,
                                    "output_settings": self.output_settings,
                                    "global_settings": self.global_settings,
                                    "cell_size": simulation.cell_size,
                                    })
            intensity_map = []
            for i in range(len(results)):
                intensity_map.append([])
                for j in range(len(results[i])):
                    intensity_map[i].append(results[i][j]["intensity_spectrum"])

        elif method == "vectorized":
            #Dont work, physics is wrong somewhere
            is_isothermal = not isinstance(TEMPERATURE, np.ndarray)
            temperature_cube = TEMPERATURE*np.ones_like(simulation.data['RHO']) if is_isothermal else TEMPERATURE
            line_settings = self.line_settings
            global_settings = self.global_settings
            output_settings = self.output_settings
            g_l = 2*line_settings["l"]+1
            g_u = 2*line_settings["u"]+1
            V = output_settings["v_function"](output_settings["lsr_velocity"],output_settings["velocity_channels"],output_settings["velocity_resolution"])

            intensity_map = BLACKBODY_EMISSION((V/LIGHT_SPEED+1)*line_settings["frequency"],CMB_TEMPERATURE)
            rho = simulation.data["RHO"]
            vx1 = simulation.data["VX1"]
            vx2 = simulation.data["VX2"]
            vx3 = simulation.data["VX3"]
            direction = np.eye(3, dtype=int)[axis]
            for ite,i in enumerate(reversed(range(rho.shape[axis]))):
                printProgressBar(ite, rho.shape[axis], prefix="Radiative Transfer",length=30)
                density = np.take(rho, i, axis=axis)
                temperature = np.take(temperature_cube, i, axis=axis)
                v = (np.take(vx1, i, axis=axis)*direction[0] + np.take(vx2, i, axis=axis)*direction[1] + np.take(vx3, i, axis=axis)*direction[2])*1e-2
                mask = density >= global_settings["density_threshold"]
                if not np.any(mask):
                    continue
                sigma_doppler = 0.08e3 * np.sqrt(temperature / 20.0)
                sigma_turb = 0.0
                if global_settings["with_turbulence"]:
                    v_m = np.take(vx1*direction[0] + vx2*direction[1] + vx3*direction[2],max(i-1, 0),axis=axis)*1e-2
                    v_p = np.take(vx1*direction[0] + vx2*direction[1] + vx3*direction[2],min(i+1, rho.shape[axis]-1),axis=axis)*1e-2
                    sigma_turb = 0.5 * (v_p - v_m)
                sigma = np.sqrt(sigma_doppler**2 + sigma_turb**2)
            low_density_col = (1e4*density*simulation.cell_size*line_settings["abundance"]*g_l*np.exp(-line_settings["temp_low"] / temperature)/(1 / 3 + 2 * temperature / 5.5))
            tau0 = (LIGHT_SPEED**3/(8 * np.pi * line_settings["frequency"]**3)*line_settings["estein_emission"]*g_u/g_l*(1 - np.exp(-line_settings["temperature"] / temperature))*low_density_col)
            tau = tau0[..., None] * GAUSSIAN(x=V - v[..., None], sigma=sigma[..., None])
            tau_exp = np.exp(-tau)
            source = BLACKBODY_EMISSION(nu=line_settings["frequency"],T=temperature)
            intensity_map = (intensity_map * tau_exp + source[..., None] * (1 - tau_exp))

        intensity_map = np.array(intensity_map)
        V = self.output_settings["v_function"](self.output_settings["lsr_velocity"],self.output_settings["velocity_channels"],self.output_settings["velocity_resolution"])
        intensity_map = intensity_map-BLACKBODY_EMISSION(((V)/LIGHT_SPEED+1)*self.line_settings["frequency"],CMB_TEMPERATURE)
        intensity_map = CONVERT_INTENSITY_TO_KELVIN(intensity_map,self.line_settings["frequency"])


        self.map = intensity_map
        self.global_settings["shape"] = intensity_map.shape

        self.save()

        LOGGER.border("", level=1)
        LOGGER.reset()

        return intensity_map
    
    def plot_channel_map(self, simulation=None, slice=None, mean_mod=False, ax=None, norm=None, enable_slider=True):
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        intensity_map = self.map
        

        velocity_channels = self.output_settings["velocity_channels"]
        if slice is None:
            slice = int(velocity_channels/2)

        if mean_mod:
            I = np.sum(intensity_map,axis=2)
            im = ax.imshow(I/velocity_channels,extent=None if simulation is None else [simulation.axis[0][0], simulation.axis[0][1], simulation.axis[1][0],simulation.axis[1][1]] , cmap="jet", norm=LogNorm() if norm is None else norm)
        else:
            im = ax.imshow(intensity_map[:,:,slice],extent=None if simulation is None else [simulation.axis[0][0], simulation.axis[0][1], simulation.axis[1][0],simulation.axis[1][1]], cmap="viridis")
        plt.colorbar(im, label="Intensity (K)")
        if simulation is None:
            ax.set_xlabel(r"$x_1$ [pixel]")
            ax.set_ylabel(r"$x_2$ [pixel]")
        else:
            ax.set_xlabel(r"$x_1$ [pc]")
            ax.set_ylabel(r"$x_2$ [pc]")
        ax.legend()

        if not(mean_mod) and enable_slider:
            ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
            slider = Slider(ax_slider, 'Slice', 0, intensity_map.shape[2] - 1, valinit=slice, valfmt='%0.0f')
            fig._slice_slider = slider

            def update_slice(val):
                slice_idx = int(slider.val)
                im.set_data(intensity_map[:,:,slice_idx])
                fig.canvas.draw_idle()

            slider.on_changed(update_slice)

        return fig, ax

    def plot(self, fit:Optional[str]=None, simulation=None, norm=LogNorm):
        fig, ax = plt.subplots()
        intensity_map = self.map
        image = ax.imshow(self.getIntegratedIntensity(),extent=None if simulation is None else [simulation.axis[0][0], simulation.axis[0][1], simulation.axis[1][0],simulation.axis[1][1]],
                          norm=norm() if norm is not None else None)
        plt.colorbar(image, label=r"Integrated intensity [K m s$^{-1}$]")
        if simulation is None:
            ax.set_xlabel(r"$x_1$ [pixel]")
            ax.set_ylabel(r"$x_2$ [pixel]")
        else:
            ax.set_xlabel(r"$x_1$ [pc]")
            ax.set_ylabel(r"$x_2$ [pc]")

        def _convert_to_phys(x,y, invert=False):
            if simulation is not None:
                x0, x1, y0, y1 = simulation.axis[0][0], simulation.axis[0][1], simulation.axis[1][0], simulation.axis[1][1]
                img = self.getIntegratedIntensity()
                nx, ny = img.shape[1], img.shape[0]
                if not invert:
                    x = int(x/nx * (x1 - x0) + x0)
                    y = int(y/ny * (y1 - y0) + y0)
                else:
                    x = int((x-x0)/(x1-x0) * nx)
                    y = int((y-y0)/(y1-y0) * ny)
            return int(np.floor(x)),int(np.floor(y))
        fig2, ax2 = plt.subplots()
        spectrum_used = Spectrum(intensity_map[0,0])
        spectrum_used.plot(ax=ax2, channels=spectrum_used.get_X(self.output_settings))
        x0, y0 = _convert_to_phys(0,0)
        marker, = ax.plot([x0], [y0], marker='x', color='red', markersize=6, mew=2)

        def onclick(event):
            if event.inaxes == ax:
                x_click, y_click = event.xdata, event.ydata
                ax2.cla()
                x0, y0 = _convert_to_phys(x_click,y_click, invert=True)
                spectrum_used = self.get_spectra(intensity_map[x0,y0],(x0,y0))
                if fit is not None:
                    spectrum_used.fit(method=fit)
                #spectrum_used.spectrum = spectrum_used.add_noise(10)
                spectrum_used.plot(ax=ax2, channels=spectrum_used.get_X(self.output_settings), show_fit=True)
                ax2.set_title(f"Spectrum at ({round(x_click,2)}pc, {round(y_click,2)}pc)")
                marker.set_data([x_click], [y_click])
                fig.canvas.draw_idle()
                fig2.canvas.draw_idle()
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
    
def generate_spectrummap_using_orphan(name, folder=CACHES_FOLDER):
    """Method to generate SpectrumMap object and files using the deprecated version, i.e a npy file of a list of shape N x N x channels"""
    LOGGER.log("Generating spectrum map using a deprecated npy map.")
    path = os.path.join(folder,name.split(".npy")[0]+".npy")
    if not(os.path.exists(path)):
        LOGGER.error(f"Orphan spectrum map {name} is not found in folder: {folder}.")
        return 
    map = np.load(path)
    spectrum_map = SpectrumMap(name, map=map, load=False)
    spectrum_map.global_settings["shape"] = map.shape
    spectrum_map.save()
    return spectrum_map

def getSimulationSpectra(simulation, name_used:Optional[str]=None, axes:List[int]=[0,1,2]):

    name = simulation.name if name_used is None else name_used
    if "T_INDEX" in simulation.data and name_used is None:
        name=name+"_"+str(simulation.data["T_INDEX"])
    spectra = [SpectrumMap("spectrum_"+name+"_"+str(int(i+1))) for i in axes]
    for i,s in enumerate(spectra):
        if s.map is None:
            LOGGER.log(f"Spectrum for face {i} doesn't exist, generating it: ")
            s.generate(axis=i, simulation=simulation)
            s.save()
    return spectra

from .Spectrum import _method_getMoment
def _method_getMom(args):
    return _method_getMoment(args, m=0)
