import os
import sys

if __name__ == "__main__":
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(parent_dir)
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

def _output_v_function(lsr,chan,res):
    return lsr+(np.array(range(chan))-chan/2)*res
DEFAULT_OUTPUT_SETTINGS = {
    "velocity_channels": 128,
    "velocity_resolution": 1e3*0.1,
    "lsr_velocity": 0,
    "v_function": _output_v_function,
}

#Line settings example for 12CO J=U-L
_L = 0
_U = 1
DEFAULT_LINE_SETTINGS = {
    "l":_L,
    "u":_U,
    "abundance":CO_ABUNDANCE/70,
    "temp_low":ROT_ENERGY(_L,CO_ROT_CST),
    "temperature":ROT_ENERGY(_U,CO_ROT_CST)-ROT_ENERGY(_L,CO_ROT_CST),
    "frequency":CO_FREQUENCY[_U-1],
    "estein_emission":CO_A[_U-1]
}
"""Default line settings used for generate emission map, used in basic radiative transfer equations"""

DEFAULT_GLOBAL_SETTINGS = {
    "density_threshold": 300,
    "with_turbulence": True,
}
"""Default global settings when emission maps are generated"""

#Don't use it, it is use just for multiprocessing
def _unpack_and_call(worker_func, job):
    return worker_func(*job)

def _worker(method, y, row):
    return (y, [method({"x": x, "y": y, "data": val["data"], "output": val["output"]}) for x, val in enumerate(row)])

#TODO As for training sets, make this memory less by opening the spectra just when it is needed
class SpectrumMap():
    """
    Map (matrix NxN) of spectra. Each element of the matrix contains a list of values (which can be passed easily to Spectra object).
    """
    def __init__(self, name,map=None, load=True):

        self.name = name

        self.map = map
        
        self.line_settings = DEFAULT_LINE_SETTINGS
        self.output_settings = DEFAULT_OUTPUT_SETTINGS
        self.global_settings = DEFAULT_GLOBAL_SETTINGS

        if load:
            self.load()

    def format_map(self, map=None):
        map = self.map if map is None else map
        formatted_intensity_map = []
        for x in range(len(map)):
            formatted_intensity_map.append([])
            for y in range(len(map[x])):
                printProgressBar(x*len(map[x])+y,len(map)*len(map[x]),prefix="Format map", length=10)
                spectrum = map[x,y]
                if type(spectrum) is Spectrum:
                    return map
                s_name = f'x{x}_y{y}'
                S = Spectrum(name=s_name, spectrum=spectrum)
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
    
    #TODO add region args
    def compute(self, method, save=True, used_cpu=1., stride=1):
        """
        Compute "method" over the sprectra map, i.e each spectrum in the map is processed using method.

        Args:
            method(function): Method to compute, need to have only one parameter: a dict with the spectrum used as the key "data".
            save(bool, default:True): When finished, save the result as cache.
            used_cpu(float, default:1.): percent (/100) of cpu cores used, 1.=100%, 0.= 1 core/no multiprocessing.
            stride(int, default:1): compute with a step in the map of stride value, if =1 then all spectrum are processed.
        Returns:
            map: shape of self.map//stride containing the result of method
        """
        LOGGER.global_color = LOGGER._init_gc
        LOGGER.border("Spectrum-Computing", level=1)
        LOGGER.log(f"Compute method {method.__name__} on map")

        stride = int(stride)

        if used_cpu > 0.:
            used_cpu = max(used_cpu,1.)

            jobs = [
            (y,[{"data": data_point, "output": self.output_settings} for data_point in self.map[y][::stride]],)
            for y in range(0, len(self.map), stride)
]
            total_jobs = len(jobs)
            results = [None] * total_jobs
            completed = 0

            with mp.Pool(int(np.ceil(mp.cpu_count()*used_cpu))) as pool:
                worker_func = partial(_worker, method)
                for y, row_result in pool.imap_unordered(partial(_unpack_and_call, worker_func), jobs):
                    results[y//stride] = row_result
                    completed += 1
                    printProgressBar(completed, total_jobs, prefix="Computing", length=30)
        else:
            results = []
            for i,y in enumerate(range(0,len(self.map),stride)):
                results.append([])
                for j,x in enumerate(range(0,len(self.map[y]),stride)):
                    printProgressBar(len(self.map[y])*y+x, len(self.map[y])*len(self.map), prefix="Computing", length=30)
                    args = {
                        "x": x,
                        "y": y,
                        "data": self.map[y][x],
                        "output": self.output_settings
                    }
                    r = method(args)
                    results[i].append(r)
                    del args
        LOGGER.reset()

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

    def generate(self, simulation=None, axis=None, force_compute=False):

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

        def _compute_function(simulation, position, direction, last_step, spectra_object):
            if type(TEMPERATURE) is float:
                temperature = TEMPERATURE
            else:
                temperature = TEMPERATURE[position[0],position[1],position[2]]

            line_settings = spectra_object.line_settings
            global_settings = spectra_object.global_settings
            output_settings = spectra_object.output_settings

            g_l = 2*line_settings["l"]+1
            g_u = 2*line_settings["u"]+1

            V = output_settings["v_function"](output_settings["lsr_velocity"],output_settings["velocity_channels"],output_settings["velocity_resolution"])

            def _get_velocity(pos):
                if not(all(0 <= pos[i] < simulation.nres for i in range(len(pos)))):
                    return _get_velocity(position)[0], False
                v = np.array([simulation.data['VX1'][pos[0],pos[1],pos[2]],simulation.data['VX2'][pos[0],pos[1],pos[2]],simulation.data['VX3'][pos[0],pos[1],pos[2]]])
                return np.dot(v,direction)*1e-2, True

            density = simulation.data['RHO'][position[0],position[1],position[2]]
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
                vm1,fm1 = _get_velocity((position-direction).astype(int))
                vp1,fp1 = _get_velocity((position+direction).astype(int))
                sigma_turb = 0.5*(vp1-vm1) if fm1 and fp1 else vp1-vm1
            sigma = np.sqrt(sigma_doppler**2 + sigma_turb**2)

            #TODO Change the fct partition approximation, this is an approximation valid just for CO J=1-0
            low_density_col = 1e4*density*simulation.cell_size*line_settings["abundance"]*g_l*np.exp(-line_settings["temp_low"]/(temperature))/(1/3+2*temperature/5.5)
            tau0 = LIGHT_SPEED**3/(8*np.pi*line_settings["frequency"]**3)*line_settings["estein_emission"] * g_u/g_l * (1-np.exp(-line_settings["temperature"]/temperature)) * low_density_col
            tau = tau0 * GAUSSIAN(V-velocity,sigma)

            tau_exp = np.exp(-tau)
            intensity_spectrum = intensity_spectrum*tau_exp+BLACKBODY_EMISSION(nu=line_settings["frequency"],T=temperature)*(1-tau_exp) 

            result = {
                "intensity_spectrum": intensity_spectrum,
            }
            return result
        results = ray_mapping(simulation, lambda simulation,position,direction,last_step: _compute_function(simulation,position,direction,last_step,self), axis=axis, region=[0,-1,0,-1])
        
        intensity_map = []
        for i in range(len(results)):
            intensity_map.append([])
            for j in range(len(results[i])):
                intensity_map[i].append(results[i][j]["intensity_spectrum"])
        V = self.output_settings["v_function"](self.output_settings["lsr_velocity"],self.output_settings["velocity_channels"],self.output_settings["velocity_resolution"])
        intensity_map = intensity_map-BLACKBODY_EMISSION(((V)/LIGHT_SPEED+1)*self.line_settings["frequency"],CMB_TEMPERATURE)
        intensity_map = np.array(intensity_map)
        intensity_map = CONVERT_INTENSITY_TO_KELVIN(intensity_map,self.line_settings["frequency"])

        self.map = intensity_map
        self.global_settings["shape"] = intensity_map.shape

        self.save()

        LOGGER.border("", level=1)
        LOGGER.reset()

        return intensity_map
    
    def plotChannelMap(self, simulation=None, slice=None, mean_mod=False, ax=None, norm=None, enable_slider=True):
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

    def plot(self, fit=False, simulation=None):
        fig, ax = plt.subplots()
        intensity_map = self.map
        image = ax.imshow(self.getIntegratedIntensity(),extent=None if simulation is None else [simulation.axis[0][0], simulation.axis[0][1], simulation.axis[1][0],simulation.axis[1][1]])
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
            return x,y
        fig2, ax2 = plt.subplots()
        spectrum_used = Spectrum(intensity_map[0,0])
        spectrum_used.plot(ax=ax2, channels=spectrum_used.getX(self.output_settings))
        x0, y0 = _convert_to_phys(0,0)
        marker, = ax.plot([x0], [y0], marker='x', color='red', markersize=6, mew=2)

        def onclick(event):
            if event.inaxes == ax:
                x_click, y_click = event.xdata, event.ydata
                ax2.cla()
                #data, data_fit = fit_gaussians(intensity_map[x,y,:])
                #plot_fit(data,data_fit, ax=ax2)
                x0, y0 = _convert_to_phys(x_click,y_click, invert=True)
                spectrum_used = Spectrum(intensity_map[x0,y0])
                #ax2.plot(spectrum_used.getX(self.output_settings), spectrum_used.spectrum, label='Data')
                spectrum_used.plot(ax=ax2, channels=spectrum_used.getX(self.output_settings))
                if fit:
                    spectrum_used.fit(ax=ax2, X=spectrum_used.getX(self.output_settings))
                #plotSpectrum(intensity_map, ax=ax2, pos=(x, y))
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

def getSimulationSpectra(simulation, name_used=None):

    name = simulation.name if name_used is None else name_used
    spectra = [SpectrumMap("spectrum_"+name+"_"+str(int(i+1))) for i in range(3)]
    for i,s in enumerate(spectra):
        if s.map is None:
            LOGGER.log(f"Spectrum for face {i} doesn't exist, generating it: ")
            s.generate(axis=i, simulation=simulation)
            s.save()
    return spectra

from .Spectrum import _method_getMoment
def _method_getMom(args):
    return _method_getMoment(args, m=0)

if __name__ == "__main__":

    #generate_spectrummap_using_orphan("spectrum_orionMHD_lowB_0.39_512_1")
    #map = SpectrumMap(name="spectrum_highresspec_0")
    from .Simulation_DC import Simulation_DC
    sim = Simulation_DC(name="orionMHD_lowB_0.39_512", global_size=66.0948, init=True)
    map = SpectrumMap(name="spectrum_orionMHD_lowB_0.39_512_2")
    #map.plot(simulation=sim)
    #result = map.compute(, stride=1, used_cpu=1)
    #result = np.array(result)
    #plt.savefig(os.path.join(FIGURE_FOLDER,"13CO_integratedmap.jpg"))
    #plt.imshow(result)
    fig, ax=  map.plotChannelMap(simulation=sim, enable_slider=False)
    fig.savefig(os.path.join(FIGURE_FOLDER,"13CO_channelmap_0.jpg"))
    plt.show()