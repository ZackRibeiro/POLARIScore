import os
import sys
from POLARIScore.utils.utils import *
from POLARIScore.config import *
from POLARIScore.utils.physics_utils import PC_TO_CM, power_spectrum_2d, power_spectrum_3d
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, PowerNorm
import json
import inspect
from POLARIScore.utils.batch_utils import compute_img_score
from POLARIScore.utils.observation_utils import find_context
from astropy.io import fits
from astropy import units as u
import numpy as np
from POLARIScore.objects.SpectrumMap import getSimulationSpectra
from POLARIScore.objects.Dataset import Dataset
from typing import Dict,List,Tuple,Callable,Union, Literal, Optional
from matplotlib.widgets import Slider
from scipy.ndimage import zoom
from scipy.optimize import curve_fit
import matplotlib.cm as cm
import matplotlib.axes
import glob
from POLARIScore.utils.sim_utils import init_idefix, init_ramses
from POLARIScore.objects.SpectrumMap import SpectrumMap
import json
from scipy.ndimage import distance_transform_edt

class Simulation_DC():
    """
    DataCube Simulation is a sim where all the cells have the same size. 
    Easier to manipulate than AMR simulation.
    """
    def __init__(self, name:str, global_size:float=1, init:bool=True):
        """
        DataCube Simulation is a sim where all the cells have the same size. 
        Easier to manipulate than AMR simulation, i.e the sim tree.

        Args:
            name (str): folder name where the simulation is stored
            global_size (float): size of the not cropped simulation in parsec
            init (bool, default:True): open files and load data, else need to call init() after. (For example after modifying the self.folder) 
        """
        self.name:str = name
        """Simulation name, name of the folder where the sim is in"""
        self.global_size:float = global_size
        """Real spatial size of the global simulation in parsec"""
        self.folder:str = os.path.join(os.path.dirname(os.path.abspath(__file__)),"../data/sims/"+name+"/")
        """Path to the folder where the simulation is stored"""
        self.data:Dict[str,np.ndarray] = {}
        """Data, example: density is stored in RHO"""

        self.cores:List[Dict] = None
        """List of dense cores in simulation"""


        self.header:Dict = {}
        """Dict of sim settings"""
        self.nres:int = None
        """Resolution of the simulation (pixels*pixels), i.e shape of the matrix"""
        self.relative_size:float =1.
        """Relative size of the simulation to the global simulation"""
        self.center:Tuple[float,float,float] = [.5, .5, .5]
        """Center of the simulation to the global simulation"""
        self.cell_size:float = None
        """Simulation cell size in cm"""
        self.size:float = self.global_size
        """Real spatial size of the simulation in parsec"""
        self.bbox:Tuple[Tuple[float,float],Tuple[float,float],Tuple[float,float]] = ([0, self.size],[0, self.size],[0, self.size])
        """Simulation faces surface in parsec"""

        self.volumic_density = [None,None,None]
        self.volumic_density_method = [None,None,None]
        """Cache, e.g for computed densities, ndarray are 2D tensors"""
        self.cache: Dict = {}

        if init:
            self.init()

    def init(self, **kwargs):
        """Try to auto init depending on the files in the simulation folder"""



        if len(glob.glob(os.path.join(self.folder,"*.vtk"))) > 0:
            init_idefix(self, **kwargs)
        elif len(glob.glob(os.path.join(self.folder,"*.fits"))) > 0:
            init_ramses(self, **kwargs)
        else:
            LOGGER.error(f"Can't initialize simulation {self.name}, verify if the folder exists and if there is data files in it.")
            raise FileNotFoundError()
        
        for k in self.data.keys():
            d = self.data[k]
            if not(isinstance(d, np.ndarray)):
                continue
            if isinstance(d,np.memmap):
                continue
            if d.size <= 1e4:
                #Not a large file so we skip
                continue
            mem_path = os.path.join(self.folder,"data_"+k+".mem")
            if os.path.exists(mem_path):
                self.data[k] = np.memmap(mem_path, dtype='float32', mode='r', shape=self.data[k].shape)
                continue
            fp = np.memmap(mem_path, dtype='float32', mode='w+', shape=self.data[k].shape)
            fp[:] = self.data[k][:]
            self.data[k] = fp
            

    def project_data(self, key:Union[str,np.ndarray], i,j ,axis):
        """Return 1D Vector with data on an axis.
        Args:
            key: data key or np.ndarray cube
            i,j: 2D position on the face
            axis: face
        Returns:
            1D Vector with data on the axis
        """
        cube = key if isinstance(key, np.ndarray) else self.data[key]
        if isinstance(cube, float):
            return cube
        
        if i < 0 or i >= cube.shape[0]:
            old_i = i
            i = min(max(i,0),cube.shape[0]-1)
            LOGGER.warn(f"Position i is out of bounds: {old_i}, replacing it with {i}")
        if j < 0 or j >= cube.shape[0]:
            old_j = j
            j = min(max(j,0),cube.shape[0]-1)
            LOGGER.warn(f"Position j is out of bounds: {old_j}, replacing it with {j}")

        if axis == 0:
            ray_values = cube[:, i, j].copy()
        elif axis == 1:
            ray_values = cube[i, :, j].copy()
        elif axis == 2:
            ray_values = cube[i, j, :].copy()
        else:
            raise ValueError("Axis must be 0, 1, or 2")

        return ray_values
    
    def compute_velocity_decomposition(self, density_weighted:bool=True, axis=0, 
                                       bins:int=128, bin_min:Optional[float]=None, bin_max:Optional[float]=None)->"SpectrumMap":
        LOGGER.log(f"Computing velocity decomposition on axis {axis}.")
        direction = np.array([0, 0, 0])
        direction[axis] = -1
        bin_min = min(np.min(self.data['VX1']),np.min(self.data['VX2']),np.min(self.data['VX3'])) if bin_min is None else bin_min
        bin_max = max(np.max(self.data['VX1']),np.max(self.data['VX2']),np.max(self.data['VX3'])) if bin_max is None else bin_max
        bins = np.linspace(bin_min,bin_max,bins+1)
        result = []
        ite = 0
        for i in range(self.nres):
            result.append([])
            for j in range(self.nres):
                ite += 1
                printProgressBar(ite, total=self.nres**2, length=30, prefix="Computing velocity decomposition")
                rho = self.project_data('RHO', i=i, j=j, axis=axis)
                vx1 = self.project_data('VX1', i=i, j=j, axis=axis)
                vx2 = self.project_data('VX2', i=i, j=j, axis=axis)
                vx3 = self.project_data('VX3', i=i, j=j, axis=axis)
                vel = np.array([vx1, vx2, vx3]).transpose().dot(direction)
                hist, _ = np.histogram(vel,bins=bins,weights=rho if density_weighted else None)
                result[i].append(hist)
        result =  np.array(result)
        smap = SpectrumMap(self.name+f"_vel_decomposition_ax_{axis}", map=result ,load=False)
        smap.output_settings = {
            "velocity_channels": len(bins)-1,
            "velocity_resolution": (bin_max-bin_min)/(len(bins)-1),
            "lsr_velocity": 0,
            "v_function": lambda _,chan,res: (bin_min+np.array(range(chan))*res)/1e2
            }
        smap.global_settings = {}
        smap.line_settings = {}
        return smap
    
    def format_key_to_spectrum_map(self, key="RHO" , axis=0)->"SpectrumMap":
        assert key in self.data, LOGGER.error(f"No key {key} found in simulation data.")
        channels_number = len(self.data[key])

        data = np.moveaxis(self.data[key], axis, -1)

        smap = SpectrumMap(self.name+f"_rho_ax_{axis}", map=data ,load=False)
        smap.output_settings = {
            "velocity_channels": channels_number,
            "velocity_resolution": self.size/channels_number,
            "lsr_velocity": 0,
            "v_function": lambda _,chan,res: np.array(range(chan))*res,
            "velocity_name": "Distance",
            "velocity_unit": "pc",
            "intensity_unit": r"$\mathrm{cm}^{-3}$",
            "intensity_name": key,
            }
        smap.global_settings = {}
        smap.line_settings = {}
        return smap
    
        
    def load_cores(self, path:str='catalog_search_results.json', alpha_vir_max:Optional[float]=2.)->List[Dict]:
        if self.cores is not None:
            return self.cores
        with open(os.path.join(self.folder, path)) as file:
            data = json.load(file)
            file.close()
            cores = []
            for index in data[list(data.keys())[0]].keys():
                c = {'index':index}
                for key in data.keys():
                    key_name:str = key
                    if 'pos_n_max_' in key_name:
                        key_name = key_name.replace('pos_n_max','pos')
                    c[key_name] = data[key][index]


                keys_to_invert = []#[('pos_x','pos_z'),('vel_x','vel_z'),('B_x','B_z')]
                for key in keys_to_invert:
                    temp = c[key[0]]
                    c[key[0]] = c[key[1]]
                    c[key[1]] = temp

                keys_to_flip = []#['pos_y','pos_z']
                for key in keys_to_flip:
                    c[key] = self.global_size - c[key]

                keys_to_rename = [('rho_mean','average_n')]
                keys_to_duplicate = [('size','radius_pc','radius')]
                for key in keys_to_rename:
                    c[key[1]] = c[key[0]]
                    del c[key[0]]
                for key in keys_to_duplicate:
                    for key_2 in key[1:]:
                        c[key_2] = c[key[0]]
                #c['radius_pc'] = c['radius_pc']/2


                if alpha_vir_max is not None and 'alpha_vir' in data.keys():
                    if c['alpha_vir'] > alpha_vir_max:
                        continue

                cores.append(c)

            self.cores = cores
            LOGGER.log(f"{len(self.cores)} cores loaded in simulation {self.name}")
            return self.cores
        
    def get_cores(self, axis: int, box: Optional[Tuple[float, float, float, float]]=None, just_center=False, flip_y=True) -> Tuple[List[Dict], List[Tuple[float, float]]]:
        """box is x_min, x_max, y_min, y_max or can add z_min, z_max"""
        self.cores = self.load_cores()
        if box is None:
            box = (self.bbox[0][0],self.bbox[0][1],self.bbox[1][0],self.bbox[1][1],self.bbox[2][0],self.bbox[2][1])

        key_1 = 'pos_'
        key_2 = 'pos_'
        key_3 = 'pos_'
        invert_x = False
        invert_y = flip_y
        invert_z = False

        if axis == 0:
            key_1 += "x"
            key_2 += "y"
            key_3 += "z"
        elif axis == 1:
            key_1 += "x"
            key_2 += "z"
            key_3 += "y"
        elif axis == 2:
            key_1 += "y"
            key_2 += "z"
            key_3 += "x"

        x_c = np.array([c[key_1] for c in self.cores])
        y_c = np.array([c[key_2] for c in self.cores])
        z_c = np.array([c[key_3] for c in self.cores])
        sizes = np.array([c["size"] for c in self.cores])


        if invert_x:
            x_c = self.global_size - x_c
        if invert_y:
            y_c = self.global_size - y_c
        if invert_z:
            z_c = self.global_size - z_c


        x_min_core,x_max_core = x_c,x_c
        y_min_core,y_max_core = y_c,y_c
        z_min_core,z_max_core = z_c,z_c
        if not(just_center):
            x_min_core = x_c - sizes
            x_max_core = x_c + sizes
            y_min_core = y_c - sizes
            y_max_core = y_c + sizes
            z_min_core = z_c - sizes
            z_max_core = z_c + sizes

        flags = (
            (x_max_core >= box[0]) &
            (x_min_core <= box[1]) &
            (y_max_core >= box[2]) &
            (y_min_core <= box[3])
        )
        if len(box) > 4:
            flags = flags & (z_max_core >= box[4]) & (z_min_core <= box[5])

        indexes = np.where(flags)[0]


        cores_in_area = [self.cores[i] for i in indexes]
        return cores_in_area, (x_c[indexes], y_c[indexes], z_c[indexes])
    def get_cores_multiplicity(self, include_scale:bool=True, include_resolution:bool=False, offset:float=0.,add_flag_to_core:bool=True)->Tuple[float, float, float]:
        "Gives core multiplicity per line of sight (value between 0 and 1.), if 1. then all line of sight showing dense cores have at least 2 dense cores"
        cores = self.load_cores()
        len_cores = len(self.cores)

        multiplicities = np.array([0,0,0])
        for i in range(len_cores):
            c = cores[i]
            x,y,z = c['pos_x'], c['pos_y'], c['pos_z']
            s = c['size']
            flags_m = [False, False, False]
            for j in range(len_cores):
                c2 = cores[j]
                x2, y2, z2 = c2['pos_x'], c2['pos_y'], c2['pos_z']
                if x2 == x and y == y2 and z2 == z:
                    continue
                s2 = c2['size']
                dists = np.array([np.sqrt((y-y2)**2 + (z-z2)**2), np.sqrt((x-x2)**2 + (z-z2)**2), np.sqrt((y-y2)**2 + (x-x2)**2)])
                flags = [False, False, False]
                if include_scale:
                    flags = np.logical_or(flags, dists < s/2+s2/2+offset)
                if include_resolution:
                    flags = np.logical_or(flags, dists <= self.relative_size*self.global_size/self.nres)

                for k, f in enumerate(flags):
                    if f and not(flags_m[k]):
                        if add_flag_to_core:
                            if not('confused' in c):
                                c['confused'] = [False,False,False]
                            c['confused'][k] = True
                        multiplicities[k] += 1
                        flags_m[k] = True
                if all(flags_m):
                    break
        return multiplicities/len_cores

    def get_core_volumes(self, indexes:Union[int, List[int]],plot:bool=False, density_threshold=4e5):

        indexes = indexes if isinstance(indexes, (np.ndarray, list, tuple)) else [indexes]

        cores = []
        for i,c in enumerate(self.load_cores()):
            if i in indexes:
                print(c['n_max'])
                cores.append(c)
        
        volumes = []
        for c in cores:
            pos = np.array([c['pos_x'],c['pos_y'],c['pos_z']])
            pos = np.astype(np.floor(((pos-self.bbox[0][0]) / (self.relative_size*self.global_size))*self.nres),int)
            volumes.append(contour_3d(self.data['RHO'], pos=pos,threshold=density_threshold))

        if plot:
            coords = np.array(volumes[0])
            min_coords = coords.min(axis=0)
            coords = coords - min_coords
            max_coords = coords.max(axis=0) + 1
            voxels = np.zeros(max_coords, dtype=bool)
            voxels[coords[:, 0], coords[:, 1], coords[:, 2]] = True
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')

            ax.voxels(voxels, edgecolor='k')

        return volumes
            
    def get_core_distance_map(self, axis:int=0, method:Literal["normal","sorted_peak"]="normal", plot:bool=False):

        method = method.lower()
        cores, _ = self.get_cores(axis=0)

        distance_cube = None
        if method == "normal":
            grid = np.ones_like(self.data['RHO'], dtype=np.uint8)

            for c in cores:
                ix = int((c['pos_x'] - self.bbox[0][0]) / (self.bbox[0][1] - self.bbox[0][0]) * self.nres)
                iy = int((c['pos_y'] - self.bbox[1][0]) / (self.bbox[1][1] - self.bbox[1][0]) * self.nres)
                iz = int((c['pos_z'] - self.bbox[2][0]) / (self.bbox[2][1] - self.bbox[2][0]) * self.nres)

                grid[ix, iy, iz] = 0

            factor = 2
            sx, sy, sz = grid.shape
            grid = grid.reshape(sx//factor, factor,
                                sy//factor, factor,
                                sz//factor, factor)
            grid = grid.min(axis=(1, 3, 5))

            LOGGER.log("Computing distance cube of cores...")
            distance_cube = distance_transform_edt(grid).astype(np.float32)
            distance_cube = 1.-distance_cube/(self.nres/factor)
            distance_cube = distance_cube
        elif method == "sorted_peak":
            pass
         
        if plot and axis in [0,1,2] and distance_cube is not None:
            fig, ax = plt.subplots()
            plt.subplots_adjust(bottom=0.2)

            artists = {'im': None, 'cores': None}

            Nx, Ny = distance_cube.shape[1], distance_cube.shape[2]
            x = np.arange(Ny)
            y = np.arange(Nx)
            X, Y = np.meshgrid(x, y)

            def _plotData(slice=0):

                global im

                data = distance_cube[slice,::-1,:]
                if axis == 1:
                    data = distance_cube[::-1,slice,:]
                elif axis == 2:
                    data = distance_cube[::-1,:,slice]

                if artists["im"] is None:
                    artists["im"] = ax.imshow(data, origin="lower", cmap="jet", extent=[self.bbox[0][0], self.bbox[0][1], self.bbox[1][0],self.bbox[1][1]], norm=PowerNorm(gamma=2, vmin=np.min(distance_cube), vmax=np.max(distance_cube)))
                else:
                    artists["im"].set_data(data)

                if self.cores is not None:
                    _, cores_in_pos = self.get_cores(axis=axis, box=[self.bbox[0][0], self.bbox[0][1], self.bbox[1][0],self.bbox[1][1], (slice-1)/(self.nres/factor)*self.global_size*self.relative_size+self.bbox[0][0],(slice+1)/(self.nres/factor)*self.global_size*self.relative_size+self.bbox[0][0]])
                    if artists["cores"] is None:
                        artists["cores"] = ax.scatter(cores_in_pos[0], cores_in_pos[1], marker="+", color="white")
                    else:
                        artists["cores"].set_offsets(np.c_[cores_in_pos[0], cores_in_pos[1]])

                ax.set_title(f"Slice {slice}")
                fig.canvas.draw_idle()

            _plotData(slice=0)
            plt.colorbar(artists["im"], ax=ax, label="Distance")

            ax_slider = fig.add_axes([0.2, 0.05, 0.6, 0.03])
            slider = Slider(ax_slider, 'Slice', 0, distance_cube.shape[axis]-1, valinit=0, valfmt='%0.0f')
            fig._slice_slider = slider

            def update_slice(val):
                slice_idx = int(val)
                _plotData(slice=slice_idx)

            fig._slice_slider.on_changed(update_slice)
        return distance_cube

    def load_fit(self, key:str, path:str, unit:float=1.)->bool:
        """
        Load data stored as fits
        Args:
            key: dict key in self.data
            path: path to the fit file
            unit: multiply data by this factor
        Returns:
            is_loaded:bool
        """
        path = os.path.join(self.folder, path)
        if not(".fit" in path):
            path += ".fits"
        if not(os.path.exists(path)):
            LOGGER.error(f"Data {key} not loaded in simulation {self.name} -> File not found")
            return False
        simfile = fits.open(path)
        if key in self.data:
            LOGGER.warn(f"Sim Data had already a key {key} -> has been replaced")
        self.data[key] = simfile[0].data*unit
        simfile.close()
        if self.data[key] is None:
            LOGGER.warn(f"Data {key} not loaded in simulation {self.name} -> file empty")
            return False
        return True

    def from_index_to_scale(self,index:int)->float:
        """Return the size in cm"""
        return index*self.cell_size
    
    def _set_cache(self, key:str, value, force:bool=False):
        if key in self.cache and not(force):
            return self.cache[key]
        self.cache[key] = value
        return value
    
    def _compute_v_density(self, method:Callable=compute_mass_weighted_density, axis:int=0, force:bool=False)->np.ndarray:
        """
        Compute volume density of an axis if not already computed or force param is set to true.
        Args:
            method: Method used to compute the volume density.
            axis (int): Axis
            force (bool): If true, then even if the volume density was already computed on this face, this will be computed again.
        Returns:
            2D matrix (ndarray) 
        """
        if self.volumic_density_method[axis] is None or self.volumic_density_method[axis] != method.__name__ or self.volumic_density[axis] is None or force:
            LOGGER.log(f"Computing {method.__name__} for face {axis}, for {self.name}")
            self.volumic_density_method[axis] = method.__name__
            self.volumic_density[axis] = method(self.data['RHO'], axis=axis)
        return self.volumic_density[axis]
    
    def get_region_datacube(self, key:str, axis:int, bbox:Tuple[float,float,float,float], res:int):
        assert key in self.data, LOGGER.error(f"Can't fetch region because there is no {key} in data.")
        
        x0,x1,y0,y1 = bbox

        i_x0 = convert_pc_to_index(x0, self.nres,self.size,start=self.bbox[0][0],flip=False)
        i_x1 = convert_pc_to_index(x1, self.nres,self.size,start=self.bbox[0][0],flip=False)
        i_y0 = convert_pc_to_index(y0, self.nres,self.size,start=self.bbox[1][0],flip=False)
        i_y1 = convert_pc_to_index(y1, self.nres,self.size,start=self.bbox[1][0],flip=False)
        
        if axis == 0:
            data = self.data[key][:, i_y0:i_y1, i_x0:i_x1]
            data = np.moveaxis(data, 0, -1)
        elif axis == 1:
            data = self.data[key][i_y0:i_y1, :, i_x0:i_x1]
            data = np.moveaxis(data, 1, -1)
        elif axis == 2:
            data = self.data[key][i_y0:i_y1, i_x0:i_x1, :]

        if data.shape[0] > res:
            factors = []
            for si, shape in enumerate(data.shape):
                if si < 2:
                    factors.append(res/shape)
                    continue
                factors.append(1.)
            data = zoom(data, factors, order=3)

        return data

    def generate_dataset(self,name:str=None,
                         what_to_compute:Dict={"cospectra":False,"density":False,"vdens":compute_mass_weighted_density,"cores":False,"density_methods":None}
                       ,number:int=8,size:Union[float,Tuple[float,float]]=0.,img_size:int=128,random_rotate:bool=True,limit_area:Tuple=(None,None,None),
                       nearest_size_factor:float=0.75,axes:Union[int,List[int]]=[0,1,2])->bool:
        """
        Util method to generate a dataset from simulation.

        What can be computed:
        - 'vdens': compute average volume density along axes
        - 'co_spectra': compute the co spectra
        - 'density': keep the density cube in the dataset
        - 'cores': generate a catalog of dense cores contained in the region
        - 'density_methods': Dict of methods with keys=names and values=callable with each method taking the region cube of density (third axis is line of sight) and returns 2D maps.

        Args:
            number(int, default: 8): How many pairs of images do we want.
            size(float, default: 0): Size in parsec for the areas, if 0 it takes the lowest size possible else it is downsampled. Can be an interval.
            img_size(int, default: 128):  Size of the img/matrix, if 0 it will take the size rounded (for example 128).
            random_rotate(bool, default: True): Randomly rotate 0°,90°,180°,270° for each region.
            limit_area(list): In which region of the simulation we'll pick the areas: ([for face1],[for face2],[for face3]) -> ([x_min,x_max,y_min,y_max],...) for each face.
            nearest_size_factor(float, default:0.75): If the new area picked is too close to an old area of a factor nearest_size_factor*area_size then we'll choose another area.
            axes(list of ints or int): What faces of the simulation datacube will be used for generate the batch (e.g you may want to use 2 faces for training data and 1 face for validation data).
            what_to_compute(dict)
        Returns:
            flag: if dataset was correctly generated.
        """
        LOGGER.border("DATASET-GENERATING", color="36m")
        LOGGER.global_color = "36m"

        axes = axes if type(axes) is list else [axes]
        LOGGER.log(f"Trying to generate {number} images using simulation {self.name} on faces {axes}.")

        flag_cospectra = False
        if "cospectra" in what_to_compute and what_to_compute["cospectra"] is not None:
            if isinstance(what_to_compute["cospectra"], bool):
                flag_cospectra = what_to_compute["cospectra"]
            else:
                flag_cospectra = True

        if flag_cospectra:
            co_spectra = getSimulationSpectra(self)
            if isinstance(what_to_compute["cospectra"], str):
                if what_to_compute["cospectra"] == "pca":
                    co_spectra = [smap.pca(return_cube=True) for smap in co_spectra]
                elif what_to_compute["cospectra"] == "gaussians":
                    co_spectra = [smap.gaussians() for smap in co_spectra]
            else:
                co_spectra = [smap.map for smap in co_spectra]
        flag_number_density = what_to_compute["density"] if "density" in what_to_compute else False
        flag_density_methods = what_to_compute["density_methods"] is not None and isinstance(what_to_compute["density_methods"], dict)
        flag_physize = True
        flag_cores = "cores" in what_to_compute and what_to_compute["cores"]

        order = ["cdens"]
        order.append("vdens")
        if flag_cospectra:
            order.append("cospectra")
        if flag_density_methods:
            for dens_method_name in what_to_compute["density_methods"].keys():
                order.append("density_"+dens_method_name)
        if flag_number_density:
            order.append("density")
        if flag_physize:
            order.append("physize")
        if flag_cores:
            order.append("cores")

        name = self.name if name is None else name

        ds = Dataset()
        ds.name = name
        ds.settings = {"order": order}
        ds.data = {"physical_size": []}

        scores = []
        img_generated = 0
        areas_explored = [[],[],[]]
        iteration = 0
        while img_generated < number and iteration < number*100 :
            iteration += 1
            printProgressBar(img_generated, number+1, prefix=iteration, length=20)
            if iteration >= number*100:
                LOGGER.warn("Failed to generated all the requested random batches, nbr of imgs generated:"+str(img_generated))
                break


            face = axes[int(np.floor(np.random.random()*len(axes)))]
            if flag_cospectra:
                co_spec = co_spectra[face]  
            
            limits = limit_area[face]
            if limits is None:
                limits =  []
                for i in range(len(self.bbox)):
                    if i != face:
                        limits.append(self.bbox[i])
                limits = np.array(limits).flatten()
            center = np.array([limits[0]+(limits[1]-limits[0])*np.random.random(),limits[2]+(limits[3]-limits[2])*np.random.random()])
            c_x, c_y = center
            
            s_pc = size
            if type(s_pc) is list:
                s_pc = np.min(size) + np.random.random()*(np.max(size)-np.min(size))
            if s_pc <= 0:
                s_pc = self.from_index_to_scale(img_size)/PC_TO_CM

            flag = False
            for point in areas_explored[face]:
                if np.linalg.norm(center-point) < nearest_size_factor * s_pc:
                    flag = True
                    break
            if flag:
                continue

            start_x = c_x - s_pc / 2
            start_y = c_y - s_pc / 2
            end_x = c_x + s_pc / 2
            end_y = c_y + s_pc / 2

            if(start_x <= limits[0] or start_y <= limits[2] or end_x >= limits[1] or end_y >= limits[3]):
                continue

            densities = self.get_region_datacube('RHO', axis=face, bbox=[start_x, end_x, start_y, end_y],res=img_size)
            c_dens = compute_column_density(densities, (self.bbox[face][1]-self.bbox[face][0])/densities.shape[-1], axis=-1)
            v_dens = what_to_compute["vdens"](densities, axis=-1)

            def _process_img(img, k):
                p_img = img
                #downsample
                if p_img.shape[0] > img_size:
                    factors = []
                    for si, shape in enumerate(p_img.shape):
                        if si < 2:
                            factors.append(img_size/shape)
                            continue
                        factors.append(1.)
                    p_img = zoom(p_img, factors, order=3)

                # Randomly choose a rotation (0, 90, 180, or 270 degrees)
                p_img = np.rot90(p_img, k, axes=(0,1))
                return p_img
            
            def rotate_index(y, x, shape, k):
                H, W = shape[:2]

                if k == 0:
                    return y, x
                elif k == 1:
                    return W - 1 - x, y
                elif k == 2: 
                    return H - 1 - y, W - 1 - x
                elif k == 3: 
                    return x, H - 1 - y

            k = np.random.choice([0, 1, 2, 3]) if random_rotate else 0
            b = [_process_img(c_dens,k)]
            b.append(_process_img(v_dens,k))

            score = compute_img_score(b[0],b[1])
            #if(np.random.random() > RANDOM_BATCH_SCORE_fct(score[0])):
            #    continue

            if flag_cospectra:
                b.append(_process_img(co_spec,k))

            if flag_density_methods:
                for dens_method_name in what_to_compute["density_methods"]:
                    dens_method = what_to_compute["density_methods"][dens_method_name]
                    d_result:np.ndarray = dens_method(densities)
                    d_result = _process_img(d_result, k)
                    b.append(d_result)
            if flag_number_density:
                densities = _process_img(densities, k)
                b.append(densities)

            if flag_physize:
                b.append(np.array([s_pc]))

            if flag_cores:
                cores, cores_pos = self.get_cores(axis=face, box=([start_x,end_x,start_y,end_y]),just_center=True,flip_y=False)
                #if len(cores) <= 0:
                #    continue
                catalog_cores = {}
                for i,c in enumerate(cores):
                    core = {}
                    core['index'] = c['index']
                    c_pos = np.array([cores_pos[0][i],cores_pos[1][i],cores_pos[2][i]])
                    cx_idx = convert_pc_to_index(c_pos[0], img_size, s_pc, start=start_x)
                    cy_idx = convert_pc_to_index(c_pos[1], img_size, s_pc, start=start_y)
                    c_pos[2] = convert_pc_to_index(c_pos[2], densities.shape[-1], self.bbox[face][1]-self.bbox[face][0], start=self.bbox[face][0])

                    c_pos[1], c_pos[0] = rotate_index(cy_idx, cx_idx, (img_size, img_size), k)
                    c_pos = c_pos.astype(int)
                    core['position_x'] = c_pos[0]
                    core['position_y'] = c_pos[1]
                    core['position_z'] = c_pos[2]
                    core['position_z_pc'] = cores_pos[2][i]
                    core['size'] = c['size'] * self.nres/self.size
                    core['mass'] = c['mass']
                    if face == 0:
                        c_vel_los = c['vel_x']
                    elif face == 1:
                        c_vel_los = c['vel_y']
                    else:
                        c_vel_los = c['vel_z']
                    core['velocity'] = c_vel_los
                    catalog_cores[i] = core
                b.append(catalog_cores)

            ds.data['physical_size'].append(size)
            ds.save_batch(b, img_generated)
            del b
            scores.append(score)
            areas_explored[face].append(center)
            img_generated += 1

        print("")

        #Random permutation
        '''
        random_idx = np.random.permutation(len(imgs))
        r_imgs = []
        for r_id in random_idx:
            r_imgs.append(imgs[r_id])
        imgs = r_imgs
        r_scores = []
        for r_id in random_idx:
            r_scores.append(scores[r_id])
        imgs = r_imgs'
        '''
        scores = [r[0] for r in scores]
        
        settings = {
            "SIM_name":self.name,
            "order": order,
            "what_was_computed": what_to_compute,
            "img_number": img_generated,
            "img_size": s_pc,
            "areas_explored":areas_explored,
            "scores": scores,
            "scores_fct": inspect.getsourcelines(RANDOM_BATCH_SCORE_fct)[0][0],
            "scores_offset": str(RANDOM_BATCH_SCORE_offset),
            "number_goal": number,
            "iteration": iteration,
            "random_rotate": random_rotate,
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
    
    def plot_slice(self, axis:int=0, slice:int=0, N_arrows:int=50, show_velocity:bool=True, enable_slider:bool=True, data:Optional[np.ndarray] = None, vector_size:float=1000):

            if data is None:
                assert 'RHO' in self.data, LOGGER.error(f"There is no density stored in data. Keys actually stored in data: {self.data.keys()}")
                data_rho = self.data['RHO']
            else:
                data_rho = data
            assert slice < data_rho.shape[axis], LOGGER.error(f"Slice index ({str(slice)}) can't be higher than data matrix size ({data_rho.shape[axis]}).")
            
            if not(axis in [0,1,2]):
                LOGGER.warn(f"Slice plot: Axis {axis} is not valid -> take the default axis: 0")

            fig, ax = plt.subplots()
            plt.subplots_adjust(bottom=0.2)

            velocity = [None, None, None]
            if show_velocity and "VX1" in self.data:
                velocity = [self.data["VX1"]/1e4, self.data["VX2"]/1e4, self.data["VX3"]/1e4]

            artists = {'im': None, 'qui': None, 'cores': None}

            Nx, Ny = data_rho.shape[1], data_rho.shape[2]
            x = np.arange(Ny)
            y = np.arange(Nx)
            X, Y = np.meshgrid(x, y)

            def _plotData(slice=slice):

                global im, qui

                density = data_rho[slice,:,:]
                if axis == 1:
                    density = data_rho[:,slice,:]
                elif axis == 2:
                    density = data_rho[:,:,slice]

                if not(any([val is None for val in velocity])):
                    Ux = velocity[0][slice,:,:]
                    Uy = velocity[1][slice,:,:]
                    if axis == 1:
                        Ux = velocity[0][:,slice,:]
                        Uy = velocity[2][:,slice,:]
                    elif axis == 2:
                        Ux = velocity[1][:,:,slice]
                        Uy = velocity[2][:,:,slice]

                    step_x = max(Ny // N_arrows, 1)
                    step_y = max(Nx // N_arrows, 1)

                    X_sub = X[::step_y, ::step_x]/data_rho.shape[0]*(self.bbox[0][1]-self.bbox[0][0])+self.bbox[0][0]
                    Y_sub = Y[::step_y, ::step_x]/data_rho.shape[0]*(self.bbox[0][1]-self.bbox[0][0])+self.bbox[0][0]
                    Ux_sub = Ux[::step_y, ::step_x]
                    Uy_sub = Uy[::step_y, ::step_x]

                if artists["im"] is None:
                    artists["im"] = ax.imshow(density, cmap="jet", extent=[self.bbox[0][0], self.bbox[0][1], self.bbox[1][0],self.bbox[1][1]], norm=LogNorm(vmin=np.min(self.data['RHO']),vmax=np.max(self.data['RHO'])))
                else:
                    artists["im"].set_data(density)

                if self.cores is not None:
                    cores_in, cores_in_pos = self.get_cores(axis=axis, box=[self.bbox[0][0], self.bbox[0][1], self.bbox[1][0],self.bbox[1][1], (slice-1)/self.nres*self.global_size*self.relative_size+self.bbox[0][0],(slice+1)/self.nres*self.global_size*self.relative_size+self.bbox[0][0]])
                    if artists["cores"] is None:
                        artists["cores"] = ax.scatter(cores_in_pos[0], cores_in_pos[1], marker="+", color="white")
                    else:
                        artists["cores"].set_offsets(np.c_[cores_in_pos[0], cores_in_pos[1]])

                if not(velocity[0] is None):
                    if artists["qui"] is None:
                        artists["qui"] = ax.quiver(X_sub, self.global_size-Y_sub, Ux_sub, Uy_sub, color="white", scale=vector_size)
                    else:
                        artists["qui"].set_UVC(Ux_sub, Uy_sub)

                ax.set_title(f"Slice {slice}")
                fig.canvas.draw_idle()

            _plotData(slice=slice)
            plt.colorbar(artists["im"], ax=ax, label="Density")

            if enable_slider:
                ax_slider = fig.add_axes([0.2, 0.05, 0.6, 0.03])
                slider = Slider(ax_slider, 'Slice', 0, data_rho.shape[axis]-1, valinit=slice, valfmt='%0.0f')
                fig._slice_slider = slider

                def update_slice(val):
                    slice_idx = int(val)
                    _plotData(slice=slice_idx)

                fig._slice_slider.on_changed(update_slice)

    def plot(self,method:Callable=compute_column_density,fig=None,axis:Union[List[int],int]=[0,1,2],plot_pdf:bool=False,color_bar:bool=True,derivate:int=0,norm=None,label=None):
        """
        Plot simulations faces with probabiliy density function

        Args:
            method(function): Method to compute the data (2d tensor)
            axis(list or int): axis or axes
            plot_pdf(bool): if True plot the probability density function
            color_bar(bool): if True, plot the colorbar
            derivate(int): Derivate the data n times where n is derivate param.
        Returns:
            Tuple(fig, axes)
        """

        assert 'RHO' in self.data, LOGGER.error(f"There is no density stored in data. Keys actually stored in data: {self.data.keys()}")

        axis = axis if type(axis) is list else [axis]
        axis = np.array(axis)
        axis = axis[np.argsort(axis)]

        if fig is not None:
            color_bar = False

        densities = []
        for ax in axis:
            if len(inspect.signature(method).parameters) == 3:
                d = method(self.data['RHO'], self.cell_size, axis=ax)
            else:
                d = method(self.data['RHO'], axis=ax)
            d = compute_derivative(d, order=derivate)
            d = np.abs(d)
            densities.append(d)  

        if fig is None:
            fig = plt.figure(figsize=(4 * len(axis), 6 if plot_pdf else 3.5))

        nrows = 2 if plot_pdf else 1
        ncols = len(axis)

        axes = fig.subplots(nrows, ncols)

        if ncols == 1:
            axes = [axes]
        if not plot_pdf:
            axes = [axes]

        def _plot(column, data, ax):
            plt_ax:matplotlib.axes.Axes = axes[0][column]
            cd = plt_ax.imshow(data, extent=[self.bbox[0][0], self.bbox[0][1], self.bbox[1][0],self.bbox[1][1]], cmap="jet", norm=LogNorm() if norm is None else norm)
            
            if self.cores is not None:
                cores_in, cores_in_pos = self.get_cores(axis=ax, box=[self.bbox[0][0], self.bbox[0][1], self.bbox[1][0],self.bbox[1][1]])
                plt_ax.scatter(cores_in_pos[0], cores_in_pos[1], marker="+", color="white")
            
            if plot_pdf:
                pdf = compute_pdf(data/np.mean(data))
                pdf[1] = pdf[1]/np.sum(pdf[1])
                plt_ax_2:matplotlib.axes.Axes = axes[1][column]
                plt_ax_2.plot([(pdf[1][i+1]+pdf[1][i])/2 for i in range(len(pdf[1])-1)],pdf[0],color='black')
                plt_ax_2.set_xlabel("s")
                plt_ax_2.set_ylabel("p")
                plt_ax_2.set_title("PDF")
                plt_ax_2.set_yscale("log")
                plt_ax_2.grid()
            return cd

        for i, ai in enumerate(axis):
            cd = _plot(i,densities[i], ai)
            if ai == 0:
                axes[0][i].set_title("Top-Down View (XY Projection)")
                axes[0][i].set_xlabel("X (pc)")
                axes[0][i].set_ylabel("Y (pc)")
            elif ai == 1:
                axes[0][i].set_title("Side View (XZ Projection)")
                axes[0][i].set_xlabel("X (pc)")
                axes[0][i].set_ylabel("Z (pc)")
            elif ai == 2:
                axes[0][i].set_title("Front View (YZ Projection)")
                axes[0][i].set_xlabel("Y (pc)")
                axes[0][i].set_ylabel("Z (pc)")

        if color_bar:
            cbar = plt.colorbar(cd, ax=axes[0], orientation="vertical", fraction=0.02, pad=0.02)
            cbar.set_label(r"$N_H$ ($cm^{-2}$)" if label is None else label)

        return fig, axes

    def plot_correlation(self,ax=None, method:Callable=compute_mass_weighted_density, axis:int=-1, force_compute:bool=False, lines:List[int]=[0,1,2], colorbar=True):

        """
        Plot correlation between the column density and the volumic density

        Args:
            method(function): Method to compute volumic density
            axis(int): which face of the sim, if -1 all faces are taken
            force_compute(bool): if True, the column density and volume density will be computed even if cache is available.
        Returns:
            Tuple(fig, ax)
        """
        
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        if axis >= 0:
            column_density = compute_column_density(self.data['RHO'], self.cell_size, axis=axis).flatten()
            volume_density = self._compute_v_density(method=method, axis=axis,force=force_compute).flatten()
        else:
            column_density = np.array([compute_column_density(self.data['RHO'], self.cell_size, axis=i) for i in range(3)]).flatten()
            volume_density = np.array([self._compute_v_density(method=method, axis=i,force=force_compute) for i in range(3)]).flatten()
            
        logx = np.log10(column_density)
        logy = np.log10(volume_density)

        xbins = np.logspace(logx.min(), logx.max(), 256)
        ybins = np.logspace(logy.min(), logy.max(), 256)


        _, _,_,hist = ax.hist2d(column_density, volume_density, bins=(xbins,ybins), norm=LogNorm(), cmap=cm.viridis)

        if colorbar:
            plt.colorbar(hist, ax=ax, label="counts")
        ax.set_xlabel(r"$N_H$ $(\text{cm}^{-2})$")
        ax.set_ylabel(r"$<n_H>_m$ $(c\text{m}^{-3})$")

        ax.set_xscale("log")
        ax.set_yscale("log")

        ax = plt.gca()

        plot_lines(ax, x=column_density, y=volume_density, lines=lines, logspace=True)

        #ax.grid(True)
        ax.set_axisbelow(True)
        #fig.tight_layout()
        return fig, ax
    
    def plot_pdf_2D(self):
        fig, axes = plt.subplot_mosaic(
        [
            ["A", "A", "O"],
            ["B", "B", "C"],
            ["B", "B", "C"]
        ],
        figsize=(7, 7)
        )

        #axes["O"].remove()

        self.plot_pdf(ax=axes["A"], what="cdens", color="black", legend=False, offset_method='none', scatter=False, bins=40)
        self.plot_pdf(ax=axes["C"], what="vdens", color="black", legend=False, offset_method='none', scatter=False, swap_axes=True, bins=40)
        axes["A"].set_ylabel(r"$P_{N_H}$")
        axes["C"].set_xlabel(r"$P_{<n_H>_m}$")
        axes["C"].set_ylabel("")
        self.plot_correlation(ax=axes["B"], colorbar=False)
        self.fit_correlation(ax=axes["B"], show_gaussians=False, legend=False)
        self.fit_correlation(ax=axes["O"], legend=False, labels=False, show_data=False)

        return fig, axes

    def fit_correlation(self, ax=None, number=4, method=compute_mass_weighted_density, show_gaussians=True, 
                        legend=True, show_data=True, labels=True,
                        )->Callable[[Union[np.ndarray, float]], Union[np.ndarray, float]]:
        
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
  
        column_density = np.array([compute_column_density(self.data['RHO'], self.cell_size, axis=i) for i in range(3)]).flatten()
        volume_density = np.array([self._compute_v_density(method=method, axis=i,force=True) for i in range(3)]).flatten()

        sorted_indexes = np.argsort(column_density)
        column_density = column_density[sorted_indexes]
        volume_density = volume_density[sorted_indexes]
        binned_x, binned_y = bin_mean(column_density, volume_density, dx=0.1,min_per_bin=3)
        binned_x = np.log10(binned_x)
        binned_y = np.log10(binned_y)

        def _gaussian(x, sigma, mean, amplitude):
            return amplitude * np.exp(-(x - mean)**2 / (2 * sigma**2))

        def fit_function(x, *params):
            sigmas = params[0:number]
            means  = params[number:2*number]
            amps   = params[2*number:3*number]

            y = np.zeros_like(x)
            for i in range(number):
                y += _gaussian(x, sigmas[i], means[i], amps[i])
            return y

        sigma_guess = [0.5] * number
        mean_guess  = np.linspace(binned_x.min(), binned_x.max(), number)
        amp_guess   = [binned_y.max()/number] * number
        p0 = np.concatenate([sigma_guess, mean_guess, amp_guess])

        X = np.linspace(np.min(binned_x), np.max(binned_x), 100)

        popt, pcov = curve_fit(fit_function, binned_x, binned_y, p0=p0, maxfev=20000)

        if show_gaussians:
            for i in range(number):
                ax.plot(10**X, 10**_gaussian(X, popt[i], popt[number+i], popt[2*number+i]),color="green", label=rf"$G_{i}$: $\sigma$={popt[i]:.2f} $\mu$={popt[number+i]:.2f} Amp={popt[2*number+i]:.2f}")
                LOGGER.log(f"Gaussian {i}: std={popt[i]:.2e} mean={popt[number+i]:.2e} amp={popt[2*number+i]:.2e}")

        ax.plot(10**X, 10**fit_function(X, *popt), color="red", linestyle="--", label=r"Fit: $\sum^4_i \mathrm{G_i}$")
        if show_data:
            ax.plot(10**binned_x, 10**binned_y, marker="+", color="black", label="Data")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(visible=True)

        if labels:
            ax.set_xlabel(r"$N_H$ ($\mathrm{cm}^{-2}$)")
            ax.set_ylabel(r"$<n_H>_m$ ($\mathrm{cm}^{-3}$)")

        if legend:
            ax.legend()

        return lambda X: fit_function(X, *popt)
    
    def get_rms_velocity(self):
        for key in ['VX1', 'VX2', 'VX3']:
            assert key in self.data, LOGGER.error(f"No {key} in simulation data.")
        return np.sqrt(self.data['VX1']**2+self.data['VX2']**2+self.data['VX3']**2)  

    def plot_power_spectrum(self, ax:Optional["matplotlib.axes.Axes"]=None
                            , what_to_plot:Literal['column_density','density','rms_velocity']="column_density" , bins:int=30, label:Optional[str]=None
                            , color:Optional[str]=None, normalize:bool=False, linestyle:str="-", energy:bool=False):
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        data = None
        if what_to_plot in self.data:
            data = self.data[what_to_plot]
        elif what_to_plot == "rms_velocity":
            label = r"v rms" if label is None else label
            data = self.get_rms_velocity()
        elif what_to_plot == "density":
            label = r"n_H" if label is None else label
            data = self.data['RHO']
        else:
            label = r"$N_H$" if label is None else label
            data = compute_column_density(self.data['RHO'], self.cell_size, axis=0)

        pixel_size = self.cell_size/PC_TO_CM

        LOGGER.log("Computing power spectrum...")
        ylabel = r"$P(k)$"
        if energy:
            ylabel = r"$E(k)$"
        if len(data.shape) == 2:
            k, Pk = power_spectrum_2d(data, px_size=pixel_size, bins=bins)
            if energy:
                Pk = Pk*2*np.pi*k
        else:
            assert len(data.shape) == 3, LOGGER.error("If data is not 2D then it needs to be 3D")
            k, Pk = power_spectrum_3d(data, px_size=pixel_size, bins=bins)
            if energy:
                Pk = Pk*4*np.pi*k*k


        if normalize:
            Pk = Pk / np.max(Pk)
        ax.plot(k,Pk, color=color, label=label, linestyle=linestyle)

        #cut_index = (np.where(k > 0.0)[0][0],np.where(k > 10)[0][0])
        #dPk = np.gradient(Pk[cut_index[0]:cut_index[-1]])
        #ddPk = np.gradient(dPk)
        #sorted_indexes = np.argsort(np.abs(ddPk))
        #sonic_index = sorted_indexes[-1]
        #k_sonic = k_coldens[cut_index[0]:cut_index[-1]][sonic_index]
        #ax.vlines(k_sonic, ax.get_ylim()[0], ax.get_ylim()[1], color="red")
        #ax.text(k_sonic - 0.3,0.5,rf'$k={k_sonic:.2}={1/k_sonic:.2}$',
        #rotation=90,va='center',ha='left',color='red',fontsize=11, transform=ax.get_xaxis_transform())

        ax.set_xscale("log")
        ax.set_yscale("log")

        ax.set_xlabel(r"$k\ \mathrm{[pc^{-1}]}$")
        if normalize:
            ylabel += " (normalized)"
        ax.set_ylabel(ylabel)
        ax.grid(visible=True)

        ax.legend()

        return fig, ax

    def plot_pdf(self,ax=None,bins: int = 20,offset_method: Literal["mean", "max", "none"] = "none",what: Literal["cdens", "vdens", "rho", "vel"] = "rho",
                 color=None, legend=True, scatter=True, label:Optional[str]=None,drawstyle:Optional[str]="steps-mid",
                 vdens_method:Optional[Callable]=compute_mass_weighted_density, swap_axes=False):
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        is_log = True
        if what.lower() in ("cdens", "column_density"):
            data = np.array([compute_column_density(self.data['RHO'], self.cell_size, axis=i) for i in range(3)]).flatten()
            label = r"$<N_H>$" if label is None else label
        elif what.lower() in ("vdens", "average_density"):
            data = np.array([self._compute_v_density(method=vdens_method, axis=i, force=True) for i in range(3)]).flatten()
            label = r"$<n_H>_m$" if label is None else label
        elif what.lower() in ("vel","velocity"):
            data = np.array([*self.data['VX1'].flatten(),*self.data['VX2'].flatten(),*self.data['VX3'].flatten()])
            label = r"$v$" if label is None else label
            is_log = False
        elif what.lower() in ("rho", "density"):
            data = self.data['RHO']
            label = r"$n_H$" if label is None else label

        mask = (~np.isnan(data))
        if is_log:
            mask = mask & (data > 0)


        def _normalize_x(hist, bin_centers):
            if offset_method == "mean":
                return (bin_centers - bin_centers[np.argmin(np.abs(hist - np.mean(hist)))]) / \
                    (np.max(bin_centers) - np.min(bin_centers))
            elif offset_method == "max":
                return (bin_centers - bin_centers[np.argmax(hist)]) / \
                    (np.max(bin_centers) - np.min(bin_centers))
            else:
                return bin_centers
        data = data[mask]
        data = np.log10(data) if is_log else data
        bin_edges = np.linspace(np.min(data), np.max(data), bins + 1)
        hist, _ = np.histogram(data, bins=bin_edges, density=False)
        hist_stats_error = np.sqrt(hist) / hist
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        bin_centers = _normalize_x(hist, bin_centers)
        if is_log:
            bin_centers = 10**bin_centers
        ax.plot(hist if swap_axes else bin_centers, bin_centers if swap_axes else hist, drawstyle=drawstyle, marker="+" if scatter else None, color=color, label=label)
        if not(swap_axes):
            ax.errorbar(bin_centers, hist, yerr=hist_stats_error * hist, fmt='none', color="black")

        set_xlabel = ax.set_ylabel if swap_axes else ax.set_xlabel
        set_ylabel = ax.set_xlabel if swap_axes else ax.set_ylabel

        if offset_method == "mean":
            set_xlabel(r"($x-\mu) / (max(x)-min(x))$")
        elif offset_method == "max":
            set_xlabel(r"($x-max(x)) / (max(x)-min(x))$")
        else:
            set_xlabel(label)

        set_ylabel("pdf")

        if is_log:
            ax.set_xscale("log")
            ax.set_yscale("log")
        ax.grid(visible=True)

        if legend:
            ax.legend()

        return fig, ax


def mergeSimu(sim_array:List[Simulation_DC],keys=['RHO'])->Simulation_DC:
    """
    Merge simulations into one.
    TODO: add temp and velocity
    """
    assert all(sim.nres == sim_array[0].nres for sim in sim_array),  LOGGER.error("Resolution mismatch among simulations.")
    LOGGER.log(f"merge {len(sim_array)} simulations")
    datacube_size = sim_array[0].nres
    sim_len = int(len(sim_array)**(1/3))
    merged_simulation = np.zeros((datacube_size*sim_len, datacube_size*sim_len, datacube_size*sim_len))

    centers = np.array([sim.center for sim in sim_array])
    maxs = np.max(centers+sim_array[0].relative_size/2, axis=0)
    mins = np.min(centers-sim_array[0].relative_size/2, axis=0)

    for i,sim in enumerate(sim_array):
        printProgressBar(i, len(sim_array), prefix="Merging:")
        for k in keys:
            sim_data = sim.data[k].transpose()
            x_center, y_center, z_center = (centers[i] - mins)/(maxs-mins)
            x_offset = int((x_center) * datacube_size*sim_len -datacube_size/2)
            y_offset = int((y_center) * datacube_size*sim_len -datacube_size/2)
            z_offset = int((z_center) * datacube_size*sim_len -datacube_size/2)
            
            if 0 <= x_offset < datacube_size*sim_len and 0 <= y_offset < datacube_size*sim_len and 0 <= z_offset < datacube_size*sim_len:
                merged_simulation[x_offset:x_offset + datacube_size, 
                                y_offset:y_offset + datacube_size, 
                                z_offset:z_offset + datacube_size] = sim_data
            else:
                LOGGER.error(f"Simulation center ({x_center}, {y_center}, {z_center}) out of bounds for merged datacube.")
                return
    LOGGER.log("Simulations merged")

    host = sim_array[0]
    host.data = {}
    host.data['RHO'] = merged_simulation.transpose()
    host.nres = datacube_size*sim_len
    host.relative_size = host.relative_size*sim_len
    host.center = np.array([0.5,0.5,0.5])
    host.cell_size = (host.global_size*host.relative_size/host.nres) * u.parsec
    host.cell_size = host.cell_size.to(u.cm)
    host.size = host.global_size*host.relative_size
    host.bbox = ([host.center[0]*host.global_size-host.size/2,host.center[0]*host.global_size+host.size/2],[host.center[1]*host.global_size-host.size/2,host.center[1]*host.global_size+host.size/2],[host.center[2]*host.global_size-host.size/2,host.center[2]*host.global_size+host.size/2])  
    
    return host

import glob
def openSimulation(name_root:str, global_size:float, use_cache:bool=True,cache_name="sim_memory",keys=['RHO','VX1','VX2','VX3','TEMP'])->Simulation_DC:
    """
    Open a datacube simulation
    TODO: add temp and velocity
    Args:
        name_root(str): name of the simulation folder
        global_size(float): physical size of the global simulation (not the datacube) like the datacube can be 5pc long but the simulation was runned with a grid 66pc long.
        use_cache(bool): Use cache
        cache_name(str): name of the cache file .npy containing the simulation.
    Returns:
        Simulation
    """
    files =glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)),"../data/sims/"+name_root+"*"))
    LOGGER.log(f"Opening {len(files)} simulations")
    names = [f.split("/")[-1] for f in files]
    sims = []
    if use_cache and any([os.path.exists(CACHES_FOLDER+cache_name+"_"+k) for k in keys]):
        LOGGER.log("Merged using cached data")
        sim = Simulation_DC(names[0], global_size, init=True)
        sim_len = int(len(names)**(1/3))
        for k in keys:
            if os.path.exists(CACHES_FOLDER+cache_name+"_"+k):
                sim.data[k] = np.memmap(CACHES_FOLDER+cache_name+"_"+k, dtype='float32', mode='r', shape=(sim.data[k].shape[0]*sim_len,sim.data[k].shape[1]*sim_len,sim.data[k].shape[2]*sim_len))

        sim.nres = sim.nres*sim_len
        sim.relative_size = sim.relative_size*sim_len
        sim.center = np.array([0.5,0.5,0.5])
        sim.cell_size = (sim.global_size*sim.relative_size/sim.nres) * u.parsec
        sim.cell_size = sim.cell_size.to(u.cm).value
        sim.size = sim.global_size*sim.relative_size
        sim.bbox = ([sim.center[0]*sim.global_size-sim.size/2,sim.center[0]*sim.global_size+sim.size/2],[sim.center[1]*sim.global_size-sim.size/2,sim.center[1]*sim.global_size+sim.size/2],[sim.center[2]*sim.global_size-sim.size/2,sim.center[2]*sim.global_size+sim.size/2])  
        return sim

    for n in names:
        sims.append(Simulation_DC(n, global_size, init=True))
    sim = mergeSimu(sims)
    del sims

    for k in keys:
        if not(k in sim.data):
            continue
        fp = np.memmap(CACHES_FOLDER+cache_name+"_"+k, dtype='float32', mode='w+', shape=sim.data[k].shape)
        fp[:] = sim.data[k][:]
        sim.data[k] = fp


    return sim