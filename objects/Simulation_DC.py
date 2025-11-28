import os
import sys
from POLARIScore.utils.utils import *
from POLARIScore.config import *
from POLARIScore.utils.physics_utils import PC_TO_CM
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import json
import inspect
from POLARIScore.utils.batch_utils import compute_img_score
from POLARIScore.utils.observation_utils import find_context
from astropy.io import fits
from astropy import units as u
import numpy as np
from POLARIScore.objects.SpectrumMap import getSimulationSpectra
from POLARIScore.objects.Dataset import Dataset
from typing import Dict,List,Tuple,Callable,Union, Literal
from matplotlib.widgets import Slider
from scipy.ndimage import zoom
from scipy.optimize import curve_fit
import matplotlib.cm as cm


class Simulation_DC():
    """
    DataCube Simulation is a sim where all the cells have the same size. 
    Easier to manipulate than AMR simulation, i.e the sim tree.
    """
    def __init__(self, name:str, global_size:float, init:bool=True):
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
        self.file:str = os.path.join(self.folder,SIM_DATA_NAME)
        """Path to the simulation data"""
        self.data:np.ndarray = None
        """Raw simulation density data"""
        self.data_temp:np.ndarray = None
        """Raw simulation temperature data"""
        self.data_vel:Tuple[np.ndarray,np.ndarray,np.ndarray] = [None,None,None]
        """Raw simulation velocity data (tuple of 3 datacube for xvel, yvel, zvel)"""

        self.header:Dict = None
        """Dict of sim settings"""
        self.nres:int = None
        """Resolution of the simulation (pixels*pixels), i.e shape of the matrix"""
        self.relative_size:float =None
        """Relative size of the simulation to the global simulation"""
        self.center:Tuple[float,float,float] = None
        """Center of the simulation to the global simulation"""
        self.cell_size:float = None
        """Simulation cell size in cm"""
        self.size:float = None
        """Real spatial size of the simulation in parsec"""
        self.axis:Tuple[Tuple[float,float],Tuple[float,float],Tuple[float,float]] = None
        """Simulation faces surface in parsec"""

        """Cache for computed densities, ndarray are 2D tensors"""
        self.column_density:Tuple[np.ndarray,np.ndarray,np.ndarray] = [None,None,None]
        self.column_density_method:Tuple[np.ndarray,np.ndarray,np.ndarray] = [None,None,None]
        self.volumic_density:Tuple[np.ndarray,np.ndarray,np.ndarray] = [None,None,None]
        self.volumic_density_method:Tuple[np.ndarray,np.ndarray,np.ndarray] = [None,None,None]

        if init:
            self.init()

    def loadTemperature(self)->bool:
        """
        Load Temperature data from files

        Returns:
            isLoaded:bool
        """
        path = os.path.join(self.folder,SIM_DATA_NAME.split(".fits")[0]+"_temp.fits")
        if not(os.path.exists(path)):
            LOGGER.warn(f"Temperature not loaded in {self.name}, file not found")
            return False
        simfile = fits.open(path)
        self.data_temp = simfile[0].data
        simfile.close()
        if self.data_temp is None:
            LOGGER.warn(f"Temperature not loaded in {self.name}, file empty")
            return False
        return True
    
    def loadVelocity(self)->bool:
        """
        Load velocity data from files

        Returns:
            isLoaded:bool
        """
        path_x = os.path.join(self.folder,SIM_DATA_NAME.split(".fits")[0]+"_velx.fits")
        path_y = os.path.join(self.folder,SIM_DATA_NAME.split(".fits")[0]+"_vely.fits")
        path_z = os.path.join(self.folder,SIM_DATA_NAME.split(".fits")[0]+"_velz.fits")

        if not(os.path.exists(path_x)):
            LOGGER.warn(f"Velocity not loaded in {self.name}, file for x component not found")
            return False
        if not(os.path.exists(path_y)):
            LOGGER.warn(f"Velocity not loaded in {self.name}, file for y component not found")
            return False
        if not(os.path.exists(path_z)):
            LOGGER.warn(f"Velocity not loaded in {self.name}, file for z component not found")
            return False
        
        simfile = fits.open(path_x)
        self.data_vel[0] = simfile[0].data
        simfile.close()
        simfile = fits.open(path_y)
        self.data_vel[1] = simfile[0].data
        simfile.close()
        simfile = fits.open(path_z)
        self.data_vel[2] = simfile[0].data
        simfile.close()

        return True

    def init(self, loadTemp:bool=False, loadVel:bool=False):
        """
        Load files and data in self variables

        Args:
            loadTemp (bool): try to load temperature ?
            loadVel (bool): try to load velocity ?
        """

        LOGGER.log(f"Loading simulation {self.name}")

        simfile = fits.open(self.file)
        self.data = simfile[0].data
        simfile.close()

        self.column_density = [None,None,None]
        self.column_density_method = [None,None,None]
        self.volumic_density = [None,None,None]
        self.volumic_density_method = [None,None,None]
        
        if loadTemp:
            LOGGER.log(f"Loading temperature of simulation {self.name}")
            self.loadTemperature()
        if loadVel:
            LOGGER.log(f"Loading velocity of simulation {self.name}")
            self.loadVelocity()

        if os.path.exists(os.path.join(self.folder,"processing_config.json")):
            with open(os.path.join(self.folder,"processing_config.json"), "r") as file:
                self.header = json.load(file)
            self.nres = self.header["run_parameters"]["nres"] if "nres" in self.header["run_parameters"] else self.header["run_parameters"]["nxyz"]
            self.relative_size = self.header["run_parameters"]["size"]
            self.center = np.array([self.header["run_parameters"]["xcenter"],self.header["run_parameters"]["ycenter"],self.header["run_parameters"]["zcenter"]])
            self.cell_size = (self.global_size*self.relative_size/self.nres) * u.parsec
            self.cell_size = self.cell_size.to(u.cm).value
            self.size = self.global_size*self.relative_size
            self.axis = ([self.center[0]*self.global_size-self.size/2,self.center[0]*self.global_size+self.size/2],[self.center[1]*self.global_size-self.size/2,self.center[1]*self.global_size+self.size/2],[self.center[2]*self.global_size-self.size/2,self.center[2]*self.global_size+self.size/2])    
        LOGGER.log(f"Loading finished for simulation {self.name}")

    def from_index_to_scale(self,index:int)->float:
        """Return the size in cm"""
        return index*self.cell_size

    def _compute_c_density(self, method:Callable=compute_column_density, axis:int=0, force:bool=False)->np.ndarray:
        """
        Compute column density of an axis if not already computed or force param is set to true.
        Args:
            method: Method used to compute the column density.
            axis (int): Axis
            force (bool): If true, then even if the column density was already computed on this face, this will be computed again.
        Returns:
            2D matrix (ndarray) 
        """
        if self.column_density_method[axis] is None or self.column_density_method[axis] != method.__name__ or self.column_density[axis] is None or force:
            LOGGER.log(f"Computing {method.__name__} for face {axis}, for {self.name}")
            self.column_density_method[axis] = method.__name__
            self.column_density[axis] = method(self.data, self.cell_size, axis=axis)
        return self.column_density[axis]
    
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
            self.volumic_density[axis] = method(self.data, axis=axis)
        return self.volumic_density[axis]

    def generate_batch(self,name:str=None,method:Callable=compute_mass_weighted_density,what_to_compute:Dict={"cospectra":False,"density":False,"context":10.},number:int=8,size:Union[float,Tuple[float,float]]=0.,img_size:int=128,random_rotate:bool=True,limit_area:Tuple=(None,None,None),nearest_size_factor:float=0.75,axis:Union[int,List[int]]=[0,1,2])->bool:
        """
        Generate a batch, i.e pairs of images (2D matrix) like [(col_dens_1, vol_dens_1),(col_dens_2, vol_dens_2)]
        using this simulation. This will take randoms positions images in simulation.

        Args:
            method(function): Method to compute the volumic density, like do we take the volume weighted mean ? Or the mass weighted mean ? Or even the max density along the l.o.s ?
            number(int, default: 8): How many pairs of images do we want.
            size(float, default: 0): Size in parsec for the areas, if 0 it takes the lowest size possible else it is downsampled. Can be an interval.
            img_size(int, default: 128):  Size of the img/matrix, if 0 it will take the size rounded (for example 128).
            random_rotate(bool, default: True): Randomly rotate 0°,90°,180°,270° for each region.
            limit_area(list): In which region of the simulation we'll pick the areas: ([for face1],[for face2],[for face3]) -> ([x_min,x_max,y_min,y_max],...) for each face.
            nearest_size_factor(float, default:0.75): If the new area picked is too close to an old area of a factor nearest_size_factor*area_size then we'll choose another area.
            axis(list of ints or int): What faces of the simulation datacube will be used for generate the batch (e.g you may want to use 2 faces for training data and 1 face for validation data).
            what_to_compute(dict): keys descriptions (values are bools):<br /> 'co_spectra': compute the co spectra<br /> 'density': keep the density cube in the dataset<br /> 'context': generate a downsampled global region (default is all the sim face) with a channel for a crop mask: 1 if the random region contains the pos else 0.
        Returns:
            flag: if dataset was correctly generated.
        """

        LOGGER.border("BATCH-GENERATING")


        axis = axis if type(axis) is list else [axis]
        LOGGER.log(f"Generating {number} images using simulation {self.name} on faces {axis}.")



        column_density = [self._compute_c_density(axis=0),self._compute_c_density(axis=1),self._compute_c_density(axis=2)]
        volume_density = [self._compute_v_density(method, axis=0),self._compute_v_density(method, axis=1),self._compute_v_density(method, axis=2)]

        flag_cospectra = what_to_compute["cospectra"] if "cospectra" in what_to_compute else False
        if flag_cospectra:
            co_spectra = getSimulationSpectra(self)
        flag_number_density = what_to_compute["density"] if "density" in what_to_compute else False
        flag_context = (what_to_compute["context"] is not None and what_to_compute["context"]) if "context" in what_to_compute else False
        flag_physize = True

        order = ["cdens","vdens"]
        if flag_cospectra:
            order.append("cospectra")
        if flag_number_density:
            order.append("density")
        if flag_context:
            order.append("cdens_context")
        if flag_physize:
            order.append("physize")

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
            print(f'{iteration}', end = "\r")
            if iteration >= number*100:
                LOGGER.warn("Failed to generated all the requested random batches, nbr of imgs generated:"+str(img_generated))
                break

            face = axis[int(np.floor(np.random.random()*len(axis)))]
            c_dens = column_density[face]
            v_dens = volume_density[face]
            if flag_cospectra:
                co_spec = co_spectra[face]  
            
            limits = limit_area[face]
            if limits is None:
                limits = [0,self.global_size,0,self.global_size]
            center = np.array([limits[0]+(limits[1]-limits[0])*np.random.random(),limits[2]+(limits[3]-limits[2])*np.random.random()])
            c_x, c_y = center
            c_x = convert_pc_to_index(c_x, self.nres,self.size,start=self.axis[0][0])
            c_y = convert_pc_to_index(c_y, self.nres,self.size,start=self.axis[0][0])
            if(c_x < 0 or c_y < 0):
                continue
            
            s = img_size
            i_size = size
            if type(i_size) is list:
                i_size = np.min(size) + np.random.random()*(np.max(size)-np.min(size))
            s = max(convert_pc_to_index(i_size, self.nres, self.size, clip=False)+1, img_size)
            i_size = self.from_index_to_scale(s)/PC_TO_CM

            #Verify if the region is already covered by a previous generated image
            flag = False
            for point in areas_explored[face]:
                if np.linalg.norm(center-point) < nearest_size_factor * i_size:
                    flag = True
                    break
            if flag:
                continue

            start_x = c_x - s // 2
            start_y = c_y - s // 2
            end_x = c_x + s // 2 + s%2
            end_y = c_y + s // 2 + s%2

            if(start_x < 0 or start_y < 0 or end_x >= self.nres or end_y >= self.nres):
                continue

            def _process_img(img, k, skip_crop=False):
                p_img = img
                if not(skip_crop):
                    p_img = p_img[start_x:end_x, start_y:end_y]
                #downsample
                #Verify this downsample method 
                if p_img.shape[0] > img_size:
                    factors = []
                    for si, shape in enumerate(p_img.shape):
                        if si < 2:
                            factors.append(img_size/shape)
                            continue
                        factors.append(1.)
                    p_img = zoom(p_img, factors, order=3)

                # Randomly choose a rotation (0, 90, 180, or 270 degrees)
                if random_rotate:
                    p_img = np.rot90(p_img, k, axes=(0,1))
                return p_img

            k = np.random.choice([0, 1, 2, 3])
            b = [_process_img(c_dens,k),_process_img(v_dens,k)]

            score = compute_img_score(b[0],b[1])
            if(np.random.random() > RANDOM_BATCH_SCORE_fct(score[0])):
                continue

            if flag_cospectra:
                b.append(_process_img(co_spec,k))

            if flag_number_density:
                if face == 0:
                    densities = self.data[:, start_x:end_x, start_y:end_y]
                elif face == 1:
                    densities = self.data[start_x:end_x, :, start_y:end_y]
                elif face == 2:
                    densities = self.data[start_x:end_x, start_y:end_y, :]

                if densities.shape[0] == self.nres:
                    densities = np.moveaxis(densities, 0, -1)
                elif densities.shape[1] == self.nres:
                    densities = np.moveaxis(densities, 1, -1)

                densities = _process_img(densities, k, skip_crop=True)

                b.append(densities)

            if flag_context:
                assert column_density[face].shape[0]//s, LOGGER.error("Datacube dimension need to be divisible by size asked to generate context.")
                context_size = what_to_compute["context"] if type(what_to_compute["context"]) is float or type(what_to_compute["context"]) is int else size
                if(context_size < i_size*2):
                    context_size = i_size*2
                context_size_idx = convert_pc_to_index(context_size, self.nres,self.size, clip=False)+1
                context_cdens = column_density[face].copy()
                context_x1,context_y1,context_x2,context_y2 = find_context(canvas=context_cdens, region=(start_x,start_y,end_x,end_y), context_size=context_size_idx)
                #crop to context
                context_cdens = context_cdens[context_x1:context_x2,context_y1:context_y2]
                context_cropmask = np.zeros_like(context_cdens)
                context_cropmask[start_x-context_x1:end_x-context_x1, start_y-context_y1:end_y-context_y1] = 1.
                context_mat = np.zeros((2,img_size,img_size))
                context_mat[0,:,:] = _process_img(context_cdens,k,skip_crop=True)
                context_mat[1,:,:] = _process_img(context_cropmask,k,skip_crop=True)
                b.append(context_mat)

            if flag_physize:
                b.append(np.array([size]))

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
            "method": method.__name__,
            "order": order,
            "what_was_computed": what_to_compute,
            "img_number": img_generated,
            "img_size": size,
            "areas_explored":areas_explored,
            "scores": scores,
            "scores_fct": inspect.getsourcelines(RANDOM_BATCH_SCORE_fct)[0][0],
            "scores_offset": str(RANDOM_BATCH_SCORE_offset),
            "number_goal": number,
            "iteration": iteration,
            "random_rotate": random_rotate,
        }
        ds.settings = settings
        ds.save_settings()

        LOGGER.log(f"New dataset {ds.name} saved")
    
        return ds
    
    def plotSlice(self, axis:int=0, slice:int=256, N_arrows:int=20, show_velocity:bool=True, enable_slider:bool=True):

            assert slice < self.data.shape[axis], LOGGER.error(f"Slice index ({str(slice)}) can't be higher than data matrix size ({self.data.shape[axis]}).")
            
            if not(axis in [0,1,2]):
                LOGGER.warn(f"Slice plot: Axis {axis} is not valid -> take the default axis: 0")

            fig, ax = plt.subplots()
            plt.subplots_adjust(bottom=0.2)

            velocity = self.data_vel if show_velocity else [None,None,None]

            artists = {'im': None, 'qui': None}

            Nx, Ny = self.data.shape[1], self.data.shape[2]
            x = np.arange(Ny)
            y = np.arange(Nx)
            X, Y = np.meshgrid(x, y)

            def _plotData(slice=slice):

                global im, qui

                density = self.data[slice,:,:]
                if axis == 1:
                    density = self.data[:,slice,:]
                elif axis == 2:
                    density = self.data[:,:,slice]

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

                    X_sub = X[::step_y, ::step_x]
                    Y_sub = Y[::step_y, ::step_x]
                    Ux_sub = Ux[::step_y, ::step_x]
                    Uy_sub = Uy[::step_y, ::step_x]

                if artists["im"] is None:
                    artists["im"] = ax.imshow(density, origin="lower", cmap="jet", extent=[0, Ny, 0, Nx], norm=LogNorm())
                else:
                    artists["im"].set_data(density)

                if not(velocity[0] is None):
                    if artists["qui"] is None:
                        artists["qui"] = ax.quiver(X_sub, Y_sub, Ux_sub, Uy_sub, color="white", scale=200)
                    else:
                        artists["qui"].set_UVC(Ux_sub, Uy_sub)

                ax.set_title(f"Slice {slice}")
                fig.canvas.draw_idle()

            _plotData(slice=slice)
            plt.colorbar(artists["im"], ax=ax, label="Density")

            if enable_slider:
                ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
                slider = Slider(ax_slider, 'Slice', 0, self.data.shape[0]-1, valinit=slice, valfmt='%0.0f')

                def update_slice(val):
                    slice_idx = int(slider.val)
                    _plotData(slice=slice_idx)

                slider.on_changed(update_slice)

    def plot(self,method:Callable=compute_column_density,axis:Union[List[int],int]=[0],plot_pdf:bool=False,color_bar:bool=True,derivate:int=0):
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

        axis = axis if type(axis) is list else [axis]
        axis = np.array(axis)
        axis = axis[np.argsort(axis)]

        densities = []
        for ax in axis:
            d = method(self.data, self.cell_size, axis=ax)
            d = compute_derivative(d, order=derivate)
            d = np.abs(d)
            densities.append(d)  

        fig, axes = plt.subplots(2 if plot_pdf else 1, len(axis), figsize=(4*len(axis), 6 if plot_pdf else 3.5))
        if len(axis) <= 1:
            axes = [axes]
        if not(plot_pdf):
            axes = [axes]

        def _plot(column, data):
            print(np.mean(data))
            cd = axes[0][column].imshow(data, extent=[self.axis[0][0], self.axis[0][1], self.axis[1][0],self.axis[1][1]], cmap="jet", norm=LogNorm())
            if plot_pdf:
                pdf = compute_pdf(data)
                axes[1][column].plot([(pdf[1][i+1]+pdf[1][i])/2 for i in range(len(pdf[1])-1)],pdf[0])
                axes[1][column].scatter([(pdf[1][i+1]+pdf[1][i])/2 for i in range(len(pdf[1])-1)],pdf[0])
                axes[1][column].set_xlabel("s")
                axes[1][column].set_ylabel("p")
                axes[1][column].set_title("PDF")
            return cd

        for i, ai in enumerate(axis):
            cd = _plot(i,densities[i])
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
            cbar.set_label(r"$N_H$ ($cm^{-2}$)")

        return fig, axes

    def plot_correlation(self,ax=None, method:Callable=compute_mass_weighted_density, axis:int=-1, force_compute:bool=False, lines:List[int]=[0,1,2], colorbar=True)->Tuple:

        """
        Plot correlation between the column density and the volumic density

        Args:
            method(function): Method to compute volumic density
            axis(int): which face of the sim, if -1 all faces are taken
            contour_levels(int): If instead of using color map, a contour map is used (for value > 0, levels of the contour map = this var)
            force_compute(bool): if True, the column density and volume density will be computed even if cache is available.
        Returns:
            Tuple(fig, ax)
        """
        
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        if axis >= 0:
            column_density = self._compute_c_density(axis=axis,force=force_compute).flatten()
            volume_density = self._compute_v_density(method=method, axis=axis,force=force_compute).flatten()
        else:
            column_density = np.array([self._compute_c_density(axis=0,force=force_compute),self._compute_c_density(axis=1,force=force_compute),self._compute_c_density(axis=2,force=force_compute)]).flatten()
            volume_density = np.array([self._compute_v_density(method=method, axis=0,force=force_compute),self._compute_v_density(method=method, axis=1,force=force_compute),self._compute_v_density(method=method, axis=2,force=force_compute)]).flatten()

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

        plot_lines(column_density, volume_density, ax, lines=lines, logspace=True)

        ax.grid(True)
        ax.set_axisbelow(True)
        fig.tight_layout()
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
        self.plot_pdf(ax=axes["C"], what="vdens", color="black", legend=False, offset_method='none', scatter=False, swap_axis=True, bins=40)
        axes["A"].set_ylabel(r"$P_{N_H}$")
        axes["C"].set_xlabel(r"$P_{<n_H>_m}$")
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
  
        column_density = np.array([self._compute_c_density(axis=i,force=True) for i in range(3)]).flatten()
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

    def plot_pdf(self,ax=None,bins: int = 20,vdens_method=compute_mass_weighted_density,offset_method: Literal["mean", "max", "none"] = "mean",what: Literal["both", "cdens", "vdens"] = "both",
                 color=None, legend=True, scatter=True, swap_axis=False):
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        if what in ("both", "cdens"):
            cdens = np.array([
                self._compute_c_density(axis=0),
                self._compute_c_density(axis=1),
                self._compute_c_density(axis=2)
            ]).flatten()
        else:
            cdens = None

        if what in ("both", "vdens"):
            vdens = np.array([
                self._compute_v_density(method=vdens_method, axis=0, force=True),
                self._compute_v_density(method=vdens_method, axis=1, force=True),
                self._compute_v_density(method=vdens_method, axis=2, force=True)
            ]).flatten()
        else:
            vdens = None

        if what == "both":
            mask = (~np.isnan(vdens)) & (vdens > 0) & (~np.isnan(cdens)) & (cdens > 0)
        elif what == "cdens":
            mask = (~np.isnan(cdens)) & (cdens > 0)
        else:
            mask = (~np.isnan(vdens)) & (vdens > 0)

        def _normalize_x(hist, bin_centers):
            if offset_method == "mean":
                return (bin_centers - bin_centers[np.argmin(np.abs(hist - np.mean(hist)))]) / \
                    (np.max(bin_centers) - np.min(bin_centers))
            elif offset_method == "max":
                return (bin_centers - bin_centers[np.argmax(hist)]) / \
                    (np.max(bin_centers) - np.min(bin_centers))
            else:
                return bin_centers

        if what in ("both", "cdens"):
            log10_coldens = np.log10(cdens[mask])
            hist_cd_raw, _ = np.histogram(log10_coldens, bins=bins+1, density=False)
            hist_cd_stats_error = np.sqrt(hist_cd_raw) / hist_cd_raw
            hist_cd, bin_edges_cd = np.histogram(log10_coldens, bins=bins+1, density=True)
            bin_centers_cd = 0.5 * (bin_edges_cd[1:] + bin_edges_cd[:-1])
            bin_centers_cd = _normalize_x(hist_cd, bin_centers_cd)
            ax.plot(hist_cd if swap_axis else 10**bin_centers_cd, 10**bin_centers_cd if swap_axis else hist_cd, drawstyle="steps-mid", marker="o" if scatter else None, color="blue" if color is None else color, label=r"$N_H$ [$cm^{-2}$]")
            if not(swap_axis):
                ax.errorbar(10**bin_centers_cd, hist_cd, yerr=hist_cd_stats_error * hist_cd, fmt='none', color="black")

        if what in ("both", "vdens"):
            log10_voldens = np.log10(vdens[mask])
            bin_edges_pr = np.linspace(np.min(log10_voldens), np.max(log10_voldens), bins + 1)
            hist_pr_raw, _ = np.histogram(log10_voldens, bins=bin_edges_pr, density=False)
            hist_pred_stats_error = np.sqrt(hist_pr_raw) / hist_pr_raw
            hist_pr, bin_edges_pr = np.histogram(log10_voldens, bins=bin_edges_pr, density=True)
            bin_centers_pr = 0.5 * (bin_edges_pr[1:] + bin_edges_pr[:-1])
            bin_centers_pr = _normalize_x(hist_pr, bin_centers_pr)
            ax.plot(hist_pr if swap_axis else 10**bin_centers_pr, 10**bin_centers_pr if swap_axis else hist_pr, drawstyle="steps-mid", marker="o" if scatter else None, color="red" if color is None else color, label=r"$<n_H>_m$ [$cm^{-3}$]")
            if not(swap_axis):
                ax.errorbar(10**bin_centers_pr, hist_pr, yerr=hist_pred_stats_error * hist_pr, fmt='none', color="black")

        if offset_method == "mean":
            ax.set_xlabel(r"($x-\mu) / (max(x)-min(x))$")
        elif offset_method == "max":
            ax.set_xlabel(r"($x-max(x)) / (max(x)-min(x))$")
        
        if swap_axis:
            ax.set_xlabel("density")
            if what == "cdens":
                ax.set_ylim(np.min(cdens), np.max(cdens))
                ax.set_xlim(np.min(hist_cd), np.max(hist_cd))
            elif what == "vdens":
                ax.set_ylim(np.min(vdens), np.max(vdens))
                ax.set_xlim(np.min(hist_pr), np.max(hist_pr))
        else:
            ax.set_ylabel("density")
            if what == "cdens":
                ax.set_xlim(np.min(cdens), np.max(cdens))
                ax.set_ylim(np.min(hist_cd), np.max(hist_cd))
            elif what == "vdens":
                ax.set_xlim(np.min(vdens), np.max(vdens))
                ax.set_ylim(np.min(hist_pr), np.max(hist_pr))

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(visible=True)

        if legend:
            ax.legend()

        return fig, ax


def mergeSimu(sim_array:List[Simulation_DC])->Simulation_DC:
    """
    Merge simulations into one.
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
        sim_data = sim.data.transpose()
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
    host.data = merged_simulation.transpose()
    host.nres = datacube_size*sim_len
    host.relative_size = host.relative_size*sim_len
    host.center = np.array([0.5,0.5,0.5])
    host.cell_size = (host.global_size*host.relative_size/host.nres) * u.parsec
    host.cell_size = host.cell_size.to(u.cm)
    host.size = host.global_size*host.relative_size
    host.axis = ([host.center[0]*host.global_size-host.size/2,host.center[0]*host.global_size+host.size/2],[host.center[1]*host.global_size-host.size/2,host.center[1]*host.global_size+host.size/2],[host.center[2]*host.global_size-host.size/2,host.center[2]*host.global_size+host.size/2])  
    
    return host

import glob
def openSimulation(name_root:str, global_size:float, use_cache:bool=True,cache_name="sim_memory")->Simulation_DC:
    """
    Open a datacube simulation
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
    if use_cache and os.path.exists(CACHES_FOLDER+cache_name):
        LOGGER.log("Merged using cached data")
        sim = Simulation_DC(names[0], global_size, init=True)
        sim_len = int(len(names)**(1/3))
        sim.data = np.memmap(CACHES_FOLDER+cache_name, dtype='float32', mode='r', shape=(sim.data.shape[0]*sim_len,sim.data.shape[1]*sim_len,sim.data.shape[2]*sim_len))
        sim.nres = sim.nres*sim_len
        sim.relative_size = sim.relative_size*sim_len
        sim.center = np.array([0.5,0.5,0.5])
        sim.cell_size = (sim.global_size*sim.relative_size/sim.nres) * u.parsec
        sim.cell_size = sim.cell_size.to(u.cm).value
        sim.size = sim.global_size*sim.relative_size
        sim.axis = ([sim.center[0]*sim.global_size-sim.size/2,sim.center[0]*sim.global_size+sim.size/2],[sim.center[1]*sim.global_size-sim.size/2,sim.center[1]*sim.global_size+sim.size/2],[sim.center[2]*sim.global_size-sim.size/2,sim.center[2]*sim.global_size+sim.size/2])  
        return sim

    for n in names:
        sims.append(Simulation_DC(n, global_size, init=True))
    sim = mergeSimu(sims)
    del sims
    fp = np.memmap(CACHES_FOLDER+cache_name, dtype='float32', mode='w+', shape=sim.data.shape)
    fp[:] = sim.data[:]
    sim.data = fp
    return sim

if __name__ == "__main__":
    sim = Simulation_DC(name="orionMHD_lowB_0.39_512", global_size=66.0948, init=True)
    #fct = sim.fit_correlation()
    sim.plot_pdf_2D()
    #sim.plot(plot_pdf=False, axis=[0,1,2], color_bar=True)
    #sim.plot(method=compute_mass_weighted_density, axis=[0,1,2], plot_pdf=False)
    #sim.init(loadTemp=True, loadVel=True)
    #sim = openSimulation("orionMHDt2_lowB_multi", global_size=66.0948, cache_name="sim_memory_2")
    #sim.plotSlice(axis=2, enable_slider=True)
    #sim.generate_batch(name="highres_sim2_32px_val",number=10000, img_size=32,what_to_compute={"cospectra":False, "density":False,"context":None},axis=[2])
    #sim.plot(derivate=2, axis=0)
    #plt.figure()
    #sim.plot_correlation(method=compute_mass_weighted_density, contour_levels=3)

    plt.show()