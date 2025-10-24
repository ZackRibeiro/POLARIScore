import os
import sys
from astropy.io import fits
from astropy.wcs import WCS

from POLARIScore.utils.physics_utils import PC_TO_CM, plot_lognorm, plot_imf_chabrier, dcmf_func, CONVERT_NH_TO_EXTINCTION
from POLARIScore.config import *
import matplotlib.pyplot as plt 
import numpy as np
from POLARIScore.utils.utils import *
from matplotlib.colors import LogNorm
import torch
import torch.nn.functional as F
from astropy.coordinates import SkyCoord, Angle
from astropy.wcs.utils import pixel_to_skycoord, skycoord_to_pixel
import astropy.units as u
import re
from POLARIScore.networks.Trainer import Trainer
from POLARIScore.objects.Dataset import getDataset
from typing import Dict, List, Tuple, Union
from scipy.stats import lognorm
from scipy.optimize import curve_fit
from POLARIScore.scripts.plotORIONsimDCMF import plot_sim_dcmf
from POLARIScore.utils.batch_utils import compute_smoothness
from POLARIScore.utils.physics_utils import CONVERT_massn_TO_n_coldens

def _crop(wcs, lims):
    ra_min, ra_max, dec_min, dec_max = lims
    corner_coords = SkyCoord([ra_min, ra_max, ra_max, ra_min], 
                            [dec_min, dec_min, dec_max, dec_max], 
                            unit="deg", frame="fk5")
    x_pix, y_pix = skycoord_to_pixel(corner_coords, wcs)
    x_min, x_max = int(np.min(x_pix)), int(np.max(x_pix))
    y_min, y_max = int(np.min(y_pix)), int(np.max(y_pix))
    return (x_min,x_max,y_min,y_max)

class Observation():
    """Observation object contains multiple tools to read observations and apply models on them."""
    def __init__(self,name:str,file_name:str):

        self.name:str = name
        self.folder:str = os.path.join(OBSERVATIONS_FOLDER, name)
        """Path to the folder where the observation is stored"""

        file_name = file_name.split(".fits")[0]+".fits"
        self.file:str = os.path.join(self.folder,file_name)
        """Path to the observation data"""
        self.data:np.ndarray = None
        self.prediction:np.ndarray = None
        self.wcs: 'WCS' = None
        self.cores: List[Dict] = None
        """Cores [{...core1_properties}]"""

        self.init()
    
    def init(self):
        file = fits.open(self.file)
        f = file[0]
        self.data = np.clip(f.data*2.,a_min=0.,a_max=None) #Obs are in N_H2, models are trained on N_H
        self.wcs = WCS(f.header)
        file.close()

    def predict(self, model_trainer:'Trainer', patch_size:Tuple[int,int]=(128, 128), nan_value:float=-1.0, overlap:float=0.5, downsample_factor:float=1., apply_baseline:bool=False)->np.ndarray:
        """
        Predict a quantity by applying a model to an observation.
        Args:
            model_trainer (Trainer): Model wrapped in a Trainer object.
            patch_size (tuple[int, int]): Shape of the 2D patches on which the model will be applied. The observation will be divided into patches of this shape.
            nan_value (float): Value used to replace NaNs in the observation.
            overlap (float): Fraction of overlap between consecutive patches.
            downsample_factor (float): Factor by which the observation is downsampled.
            baseline (bool): Whether to apply baseline correction to the model.
        Returns:
            predicted_observation
        """

        input_matrix = self.data
        nan_mask = np.isnan(input_matrix) | (input_matrix <= 0)
        if nan_value < 0:
            nan_value = float(np.nanmin(self.data[self.data>0]))
        input_matrix[nan_mask] = nan_value
        input_tensor = torch.tensor(input_matrix.astype(np.float32))
        downsampled_tensor = F.interpolate(input_tensor.unsqueeze(0).unsqueeze(0), 
                                       scale_factor=1.0/downsample_factor, 
                                       mode='bilinear', align_corners=True).squeeze(0).squeeze(0)
        
        downsampled_nan_mask = F.interpolate(torch.tensor(nan_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0),
                                               scale_factor=1.0 / downsample_factor,mode='nearest'
                                               ).squeeze(0).squeeze(0).numpy().astype(bool)

        height, width = downsampled_tensor.shape
        patch_height, patch_width = patch_size
        stride_height = int(patch_height * (1 - overlap))
        stride_width = int(patch_width * (1 - overlap))

        output_tensor = torch.zeros_like(downsampled_tensor)
        count_tensor = torch.zeros_like(downsampled_tensor)

        i_range = range(0, height - patch_height + 1, stride_height)
        j_range = range(0, width - patch_width + 1, stride_width)

        for i0,i in enumerate(i_range):
            for j0,j in enumerate(j_range):
                printProgressBar(i0*len(j_range)+j0,len(i_range)*len(j_range),prefix="Obs Pred")
                patch = downsampled_tensor[i:i+patch_height, j:j+patch_width].cpu().detach().numpy()
                valid_patch_mask = downsampled_nan_mask[i:i + patch_height, j:j + patch_width]

                if np.any(valid_patch_mask):
                    continue
                
                #Work only for 1 output: col density
                output_patch = model_trainer.predict_tensor(patch)[0]
                if apply_baseline:
                    output_patch = model_trainer.apply_baseline(output_patch, log=False)
                
                output_tensor[i:i+patch_height, j:j+patch_width] += torch.from_numpy(output_patch)
                count_tensor[i:i+patch_height, j:j+patch_width] += 1

        print("")
        output_tensor = output_tensor / count_tensor

        upsampled_output = F.interpolate(output_tensor.unsqueeze(0).unsqueeze(0), 
                                     size=(input_matrix.shape[0], input_matrix.shape[1]), 
                                     mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
        output_matrix = upsampled_output.numpy()
        output_matrix[nan_mask] = np.nan

        self.prediction = output_matrix

        return output_matrix
    
    def find_scale(self, pc: float, px_size: int, distance_pc: float) -> float:
        """
        Compute the downsampling scale factor so that a region of `px_size` pixels 
        corresponds to `pc` parsecs in width.
        """

        try:
            pixscale_deg = np.abs(self.wcs.pixel_scale_matrix.diagonal()).mean()
        except AttributeError:
            cdelt = self.wcs.wcs.cdelt
            pixscale_deg = np.mean(np.abs(cdelt))

        pixscale_rad = np.deg2rad(pixscale_deg)
        pc_per_pix = distance_pc * pixscale_rad
        current_width_pc = px_size * pc_per_pix
        scale = pc/current_width_pc

        return scale

    def get_cores(self, force_compute:bool=False)->Union[List[Dict], None]:
        """
        Get cores from files "observed_core_catalog.txt" and "derived_core_catalog.txt"
        Args:
            force_compute (bool): If False, the function will return the cached version of cores if available.
        Returns:
            cores
        """

        if not(self.cores is None) and not(force_compute):
            return self.cores

        observed_cores_path = os.path.join(self.folder, "observed_core_catalog.txt")
        if(not(os.path.exists(observed_cores_path))):
            return
        with open(observed_cores_path, "r", encoding="utf-8") as file:
            observed_lines = file.readlines()
        derived_cores_path = os.path.join(self.folder, "derived_core_catalog.txt")
        if(not(os.path.exists(observed_cores_path))):
            return
        with open(derived_cores_path, "r", encoding="utf-8") as file:
            derived_lines = file.readlines()

        border_car = ["!","|"]

        def _search_for_property(name, lines):
            for i, line in enumerate(lines):
                if not(line[0] in border_car):
                    continue
                if not(name.lower() in line.lower()):
                    continue
                match = re.search(r"\((\d+)\)", line)
                if match:
                    return int(match.group(1))
                else:
                    LOGGER.error("Can't get cores properties, there is no int in () for property {name} in cores files.")
                    return None
            LOGGER.error("Can't get cores properties, there is no match for property {name} in cores files.")
            return None


        observed_cores = []
        for i, line in enumerate(observed_lines):
            if not(line[0] in border_car):
                properties = line.strip().split()
                if len(properties) < 3:
                    continue
                offset_index = 4
                try:
                    ra_str = f"{properties[2]} {properties[3]} {properties[4]}"
                    dec_str = f"{properties[5]} {properties[6]} {properties[7]}"
                    coord = SkyCoord(ra_str, dec_str, unit=(u.hourangle, u.deg)) 
                except ValueError:
                    ra_str = f"{properties[2]}"
                    dec_str = f"{properties[3]}"
                    offset_index = 0
                    coord = SkyCoord(ra_str, dec_str, unit=(u.hourangle, u.deg)) 
                pc = {
                    "name": properties[1],
                    "peak_ncol": float(properties[54+offset_index])*1e21*2,
                    "radius": float(properties[58+offset_index]),
                    "ra": coord.ra.deg,
                    "dec": coord.dec.deg
                }
                observed_cores.append(pc)
        derived_cores = []
        for i, line in enumerate(derived_lines):
            if not(line[0] in border_car):
                properties = line.strip().split()
                if len(properties) < 3:
                    continue
                pc = {
                    "name": properties[1],
                    "peak_n": float(properties[13+offset_index])*1e4*2,
                    "average_n": float(properties[14+offset_index])*1e4*2,
                    "mass": float(properties[6+offset_index]),
                    "radius_pc": float(properties[5+offset_index]),
                    "bonnorebert": float(properties[16+offset_index]),
                    "comment": properties[18+offset_index] if len(properties) > (18+offset_index) else "" 
                }

                try:
                    pc["type"] = int(properties[17+offset_index])
                except:
                    pc["type"] = properties[17+offset_index]

                derived_cores.append(pc)
        
        cores = []
        for obs_dict in observed_cores:
            for der_dict in derived_cores:
                if obs_dict["name"] != der_dict["name"]:
                    continue
                if (type(der_dict["type"]) is int and der_dict["type"] != 2) or (type(der_dict["type"]) is str and der_dict["type"] in ["starless","protostellar"]):
                    continue
                if "tentative" in der_dict["comment"] or "SED" in der_dict["comment"] or "N region" in der_dict["comment"]:
                    continue
                if der_dict["bonnorebert"] > 2:
                    continue
                core = {**obs_dict, **der_dict}
                cores.append(core)
                break

        self.cores = cores

        return cores

    def get_predicted_density_at_cores(self, column_density:bool=False)->List[float]:
        """
        By default, returns the predicted densities at cores position. If column_density is set to True, then it returns the column density instead.
        """
        if self.cores is None:
            self.get_cores()
        if self.cores is None:
            LOGGER.error("No cores found")
            return None
        
        densities = self.prediction
        if column_density:
            densities = self.data

        if densities is None:
            LOGGER.error("No predicted density found")
            return None

        coords = SkyCoord(
            [core["ra"] for core in self.cores],
            [core["dec"] for core in self.cores],
            unit="deg"
        )
        x_pix, y_pix = skycoord_to_pixel(coords, self.wcs)

        values = []
        for x, y in zip(x_pix, y_pix):
            x_int, y_int = int(round(x)), int(round(y))
            if (0 <= y_int < densities.shape[0]) and (0 <= x_int < densities.shape[1]):
                values.append(densities[y_int, x_int])
            else:
                values.append(np.nan)

        for core, val in zip(self.cores, values):
            core["data_value"] = val

        return values
        
    def _get_cores_predicted_values(self, region:Union[Tuple[float,float,float,float],None]=None, return_ncol:bool=False, return_indexes:bool=False):
        """
        Args:
            region: [ra_max, ra_min, dec_min, dec_max]
            return_ncol: Return column density
            return_indexes: Return indexes
        """
        predicted_densities = np.array(self.get_predicted_density_at_cores())
        derived_densities =  np.array([c["average_n"] for c in self.get_cores()])
        global_indexes = np.array(range(predicted_densities.shape[0]))
        mask = (~np.isnan(predicted_densities)) & (predicted_densities > 0) & (derived_densities > 0)
        if region is not None:
            ra_max, ra_min, dec_min, dec_max = region
            ra = np.array([c["ra"] for c in self.get_cores()])
            dec = np.array([c["dec"] for c in self.get_cores()])
            region_mask = (ra >= ra_min) & (ra <= ra_max) & (dec >= dec_min) & (dec <= dec_max)
            mask = mask & region_mask
        predicted_densities = predicted_densities[mask]
        column_densities = np.array(self.get_predicted_density_at_cores(column_density=True))[mask]
        predicted_densities = CONVERT_massn_TO_n_coldens(column_densities,10,predicted_densities,np.array([c["radius_pc"] for c in self.get_cores()])[mask],is_density=False)
        derived_densities = derived_densities[mask]
        global_indexes = global_indexes[mask]
        predicted_densities = np.log10(predicted_densities)
        derived_densities = np.log10(derived_densities)
        if return_ncol:
            column_densities = np.log10(column_densities)
            sorted_indexes = np.argsort(column_densities)
            global_indexes = global_indexes[sorted_indexes]
            if return_indexes:
                return (predicted_densities[sorted_indexes], derived_densities[sorted_indexes], column_densities[sorted_indexes], global_indexes)
            return (predicted_densities[sorted_indexes], derived_densities[sorted_indexes], column_densities[sorted_indexes])
        if return_indexes:
            return (predicted_densities, derived_densities, global_indexes)
        return (predicted_densities, derived_densities)

    #-------PLOT-------

    def plot(self, data:np.ndarray=None, norm=None, plot_cores:Union[bool,Tuple[Union[float,None],Union[float,None]]]=False, crop:Union[Tuple[float,float,float,float],None]=None, force_vol:bool=False, force_col:bool=False):
        """
        Plot observation.
        Args:
            data: by default column densities of the observation, but can be predicted densities...
            norm: matplotlib norm
            plot_cores: can be a bool or a Tuple of float. If this is a tuple, this sets the column densities limit where a core will be drawn
            crop: [ra_min, ra_max, dec_min, dec_max]
            force_vol: Force volume density labels.
            force_col: Force column density labels.
        """

        if(plot_cores is not None):
            plot_cores_lims = [0, 1e23]
            if(type(plot_cores) is not bool):
                if((type(plot_cores) is tuple or type(plot_cores) is list) and len(plot_cores) >= 2):
                    plot_cores_lims = plot_cores
                plot_cores = True
                
        
        fig = plt.figure(figsize=(10,10))
        ax = plt.subplot(projection=self.wcs)
        data = self.data if data is None else data
        flag_vol_density = False
        label = r"$N_H(cm^{-2})$"
        norm = norm if not(norm is None) else LogNorm()
        if not(force_col) and (np.nanpercentile(data,50) < 1e10 or force_vol):
            flag_vol_density = True
            label=r"$n_H(cm^{-3})$"
        im = ax.imshow(data, cmap="rainbow", norm=norm)
        overlay = ax.get_coords_overlay('fk5')
        overlay.grid(color='black', ls='dotted')
        overlay[0].set_axislabel('Right Ascension (J2000)')
        #overlay[1].set_axislabel('Declination (J2000)')
        plt.colorbar(im, label=label)
        fig.tight_layout()

        if plot_cores:
            self.plot_cores(ax, norm=norm, vol_density=flag_vol_density, lims=plot_cores_lims)

        if not(crop is None):
            x_min, x_max, y_min, y_max = _crop(self.wcs, crop)
            ax.set_xlim((x_min, x_max))
            ax.set_ylim((y_min, y_max))

        return fig, ax

    def plot_cores(self,ax,cores:Union[List[Dict],None]=None,norm=None,vol_density:bool=False,show_text:bool=False, lims:Tuple[Union[None,float],Union[None,float]]=[None,None]):
        """
        Plot the cores as dots on a Matplotlib Axes.
        Args:
            ax (matplotlib.axes.Axes): the matplotlib ax
            cores : cores
            norm (matplotlib.colors.Normalize): matplotlib norm
            vol_density (bool): If True, treat the provided quantities as volume densities.
            show_text (bool): If True, annotate each dot with its value next to it.
            lims: A core is drawn only if his column density is in lims
        """
        if cores is None:
            cores = self.cores
        if cores is None:
            cores = self.get_cores()
            if cores is None:
                LOGGER.warn("Can't get the dense cores")
                return
        if lims[0] is not None or lims[1] is not None:
            resulted_cores = []
            for i,c in enumerate(cores):
                flag = True
                if lims[0] is not None and c["peak_ncol"] < lims[0]:
                    flag = False
                if flag and lims[1] is not None and c["peak_ncol"] > lims[1]:
                    flag = False
                if flag:
                    resulted_cores.append(c)
            cores = resulted_cores
        LOGGER.log(f"Plot {len(cores)} cores.")
            
        ra = [c["ra"] for c in cores]
        dec = [c["dec"] for c in cores]
        if vol_density:
            values = np.array([c["peak_n"] for c in cores])
        else:
            values = np.array([c["peak_ncol"] for c in cores])

        world_coords = SkyCoord(ra, dec, unit="deg", frame="fk5")
        x_pix, y_pix = skycoord_to_pixel(world_coords, ax.wcs)

        radius = np.array([c["radius"] if c["radius"] > 0. else 1. for c in cores]) / 3600

        pixel_scale = np.mean(np.abs(ax.wcs.pixel_scale_matrix.diagonal()))

        if norm is None:
            colors = 'none'
        else:
            colors = plt.cm.rainbow(norm(values))


        ax.scatter(x_pix, y_pix, s=radius/pixel_scale, facecolors=colors, edgecolors="black")
        if show_text:
            for i,c in enumerate(cores):
                ax.text(x_pix[i], y_pix[i], f"${values[i]:.2e}$", color='black')

        return ax
    
    def plot_validity_with_model(self, dataset_name:str="batch_highres", ax=None, c_x:Callable[[np.ndarray],float]=np.min, c_y:Callable[[np.ndarray], float]=np.max, logspace=True, patch_size:Tuple[int,int]=(128, 128), nan_value:float=None, overlap:float=0.5):

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure


        input_matrix = self.data
        input_tensor = input_matrix.astype(np.float32)
        if nan_value is not None:
            nan_mask = np.isnan(input_matrix)
            if nan_value < 0:
                nan_value = float(np.nanmin(self.data))
            input_tensor[nan_mask] = nan_value

        height, width = input_tensor.shape
        patch_height, patch_width = patch_size
        stride_height = int(patch_height * (1 - overlap))
        stride_width = int(patch_width * (1 - overlap))

        i_range = range(0, height - patch_height + 1, stride_height)
        j_range = range(0, width - patch_width + 1, stride_width)

        qx_obs = []
        qy_obs = []

        for i0,i in enumerate(i_range):
            for j0,j in enumerate(j_range):
                patch = input_tensor[i:i+patch_height, j:j+patch_width].flatten()
                if(np.isinf(patch).any() or np.isnan(patch).any()):
                    continue  
                qx_obs.append(c_x(patch))
                qy_obs.append(c_y(patch))

        ds = getDataset(dataset_name)
        qx_batch = ds.compute_over(c_x)
        qy_batch = ds.compute_over(c_y)

        if logspace:
            qx_batch = np.log10(qx_batch)
            qy_batch = np.log10(qy_batch)
            qx_obs = np.log10(qx_obs)
            qy_obs = np.log10(qy_obs)


        ax.scatter(qx_obs, qy_obs, label="Observation", marker="+")
        ax.scatter(qx_batch, qy_batch, label="Dataset", marker="+")

        #ax.set_xlabel(r"$\sigma_{log_{10}(N)}$")
        #ax.set_ylabel(r"$log_{10}(<N_{col}>)$")

        ax.legend()

        return fig, ax

    def plot_cores_hist(self, ax=None, region:Union[Tuple[float,float,float,float],None]=None):
        """
        Args:
            ax: matplotlib axis
            region: [ra_max, ra_min, dec_min, dec_max]
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        predicted_densities, derived_densities = self._get_cores_predicted_values(region=region)

        ax.hist(predicted_densities, bins=10, alpha=0.5, label="Predicted Densities")
        ax.hist(derived_densities, bins=10, alpha=0.5, label="Derived Densities")
        
        ax.set_xlabel(r"$\log_{10}(n_H) [cm^{-3}]$")

        ax.legend()

        return fig, ax
    
    def plot_cores_hist2d(self, ax=None, region:Union[Tuple[float,float,float,float],None]=None):
        """
        Args:
            ax: matplotlib axis
            region: [ra_max, ra_min, dec_min, dec_max]
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        predicted_densities, derived_densities = self._get_cores_predicted_values(region=region)

        _,_,_, hist= ax.hist2d(predicted_densities, derived_densities, bins=(10,10), norm=LogNorm())
        plt.colorbar(hist, ax=ax, label="counts")

        ax.set_xlabel("Predicted")
        ax.set_ylabel("Derived")

        return fig, ax

    def plot_dcmf(self, ax=None, bins:int=10, ext_lims:Tuple[Union[float,None],Union[float,None]]=[None,None], logM:bool=True, fit=True):
        """
        Plot the dense core mass function
        Args:
            ax: matplotlib axis
            bins: number of bins in the DCMF
            ext_lims: Choose only the dense cores between the two extinctions Av given.
            logM: plot dN/dlogM or dN/dM.
            fit: try to fit the dcmf by a lognormal with power law.
        """

        if ext_lims[0] is None:
            ext_lims[0] = -1000
        if ext_lims[1] is None:
            ext_lims[1] = 1000.

        derived_cores = self.get_cores()
        predicted_densities = np.array(self.get_predicted_density_at_cores(), dtype=np.float64)
        column_densities = np.array(self.get_predicted_density_at_cores(column_density=True), dtype=np.float64)
        derived_densities =  np.array([c["average_n"] for c in self.get_cores()], dtype=np.float64)
        mask = (~np.isnan(predicted_densities)) & (predicted_densities > 0) & (derived_densities > 0)
        column_densities = column_densities[mask]

        extinctions = []
        derived_radius = []
        for i in range(len(derived_cores)):
            extinctions.append(CONVERT_NH_TO_EXTINCTION(derived_cores[i]['peak_ncol']))
            derived_radius.append(derived_cores[i]['radius_pc'])
        extinctions = np.array(extinctions)[mask]
        derived_radius = np.array(derived_radius, dtype=np.float64)[mask]
        predicted_densities = predicted_densities[mask]
        predicted_densities = CONVERT_massn_TO_n_coldens(column_densities,10,predicted_densities,derived_radius, is_density=False)
        derived_densities = derived_densities[mask]

        global_mask = (extinctions >= ext_lims[0]) & (extinctions <= ext_lims[1])

        def _get_dcmf(masses:np.ndarray):
            m = masses[global_mask]
            m = np.log10(m)
            hist, bin_edges = np.histogram(m, bins=bins)
            bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
            dcmf = hist
            if logM:
                dcmf = dcmf/ (bin_edges[1:] - bin_edges[:-1])
            return dcmf, bin_centers
        
        def _compute_mass(radius,densities):
            m_H = 1.67e-24  # g
            mu = 1.4        # mean molecular weight for H (not H2)
            pc_to_cm = PC_TO_CM
            Msun = 1.989e33 # g
            rslt_masses = []
            for n, r_pc in zip(densities, radius):
                r_cm = r_pc * pc_to_cm
                volume = (4/3) * np.pi * (r_cm**3)
                mass = mu * m_H *n* volume
                mass_Msun = mass / Msun
                rslt_masses.append(mass_Msun)
            rslt_masses = np.array(rslt_masses, dtype=np.float64)
            return rslt_masses
        
        _dcmf_function = lambda M,amp,mu,sigma,alpha,cutoff: dcmf_func(M,amp,mu,sigma,alpha,cutoff, logM=logM)

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure()

        derived_mass = _compute_mass(derived_radius,derived_densities)
        derived_dcmf, derived_bin_centers = _get_dcmf(derived_mass)
        derived_bin_centers = 10**derived_bin_centers

        ax.plot(derived_bin_centers, derived_dcmf, drawstyle="steps-mid", color="blue", label=f"{self.name} (Könyves et al, 2020)")
        ax.scatter(derived_bin_centers, derived_dcmf, color="blue")
        if fit:
            popt, _ = curve_fit(_dcmf_function, derived_bin_centers, derived_dcmf,
                            p0=[np.max(derived_dcmf), 1.5, np.std(np.log(derived_mass)), 2.3, 1.6])
            amp_fit, mu_fit, sigma_fit, alpha_fit ,cutoff_fit = popt
            LOGGER.log(f"Best DCMF fit for estimated cores: amp={amp_fit:.2e}, mu={mu_fit:.3f}, sigma={sigma_fit:.3f}, alpha={alpha_fit}, cutoff={cutoff_fit}")
            func = lambda X: _dcmf_function(X, popt[0], popt[1], popt[2], popt[3], popt[4])
            plot_function(func, ax=ax, scatter=False, logspace=True, lims= (0.01, 100), color="blue", linestyle="--")

        LOGGER.log(f"DCMF with {len(derived_radius[global_mask])} cores.")

        if(self.prediction is not None):            
            predicted_masses = np.array(_compute_mass(derived_radius, predicted_densities))
            predicted_dcmf, predicted_bin_centers = _get_dcmf(predicted_masses)
            predicted_bin_centers = 10**predicted_bin_centers
            ax.plot(predicted_bin_centers, predicted_dcmf, drawstyle="steps-mid", color="red", label=f"{self.name} (Neural Network)")
            ax.scatter(predicted_bin_centers, predicted_dcmf, color="red")

            if fit:
                popt, _ = curve_fit(_dcmf_function, predicted_bin_centers[predicted_bin_centers>0.], predicted_dcmf[predicted_bin_centers>0.],
                            p0=[np.max(predicted_dcmf), 0.22, np.std(np.log(predicted_masses)), 2.3, 1.6])
                amp_fit, mu_fit, sigma_fit, alpha_fit ,cutoff_fit = popt
                LOGGER.log(f"Best DCMF fit for predicted cores: amp={amp_fit:.2e}, mu={mu_fit:.3f}, sigma={sigma_fit:.3f}, alpha={alpha_fit}, cutoff={cutoff_fit}")
                func = lambda X: _dcmf_function(X, popt[0], popt[1], popt[2], popt[3], popt[4])
                plot_function(func, ax=ax, scatter=False, logspace=True, lims= (0.01, 100), color="red", linestyle="--")

        plot_sim_dcmf(ax, factor=0.035, logM=logM)
        plot_imf_chabrier(ax, logM=logM)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"Mass [$M_\odot$]")
        ax.set_ylabel(r"$dN/d\log M$" if logM else r"$dN/dM$")

        ax.set_xlim([0.01, 100])
        ax.set_ylim([0.8,600])

        plt.legend()

        return fig, ax

    def plot_cores_space(self, ax=None, region:Union[Tuple[float,float,float,float],None]=None):
        """
        Args:
            ax: matplotlib axis
            region: [ra_max, ra_min, dec_min, dec_max]
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        predicted_densities, derived_densities, column_densities = self._get_cores_predicted_values(region=region, return_ncol=True)

        bins_number = 10
        contour_levels = 10

        def _plot(c1, c2, color="black", label="", linestyles="solid"):
            hist, xedges, yedges = np.histogram2d(c1, c2, bins=(bins_number, bins_number))
            hist = np.where(hist == 0, np.nan, hist)

            xcenters = 0.5 * (xedges[:-1] + xedges[1:])
            ycenters = 0.5 * (yedges[:-1] + yedges[1:])
            X, Y = np.meshgrid(xcenters, ycenters)

            contour = ax.contour(X, Y, hist.T, levels=contour_levels, colors=color, linestyles=linestyles, label=label)
            #ax.clabel(contour, fmt=lambda x: r"$10^{{{:.0f}}}$".format(np.log10(x)), inline=True, fontsize=8)

        _plot(column_densities, predicted_densities, linestyles="dashed", label="predicted")
        _plot(column_densities, derived_densities, label="derived")

        #ax.legend()

        merged_list = [d for d in derived_densities]
        merged_list.extend([c for c in predicted_densities])
        plot_lines(column_densities, merged_list, ax)

        #ax.grid()

        return fig, ax
    
    def plot_cores_error(self, ax=None, region:Union[Tuple[float,float,float,float],None]=None, alpha:float=1., mov_average:int=5, show_errors:bool=True):
        """
        Args:
            ax: matplotlib axis
            region: [ra_max, ra_min, dec_min, dec_max]
            alpha (float): opacity
            mov_average (bool): data is downsampled/smoothed using moving average method.
            show_errors (bool): show std deviation induced by the moving average.
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        predicted_densities, derived_densities, column_densities = self._get_cores_predicted_values(region=region, return_ncol=True)
        ax.grid()

        residuals = predicted_densities-derived_densities
        if mov_average > 1:
            residuals, residuals_std = moving_average(residuals, n=mov_average, return_std=True) 
            column_densities = moving_average(column_densities, n=mov_average)
        line, = ax.plot(column_densities,residuals, marker="+", alpha=alpha, label=self.name)
        if mov_average > 1 and show_errors:
            ax.fill_between(column_densities,residuals-residuals_std,residuals+residuals_std, color=line.get_color(), alpha=0.2)
        
        ax.set_xlabel(r"$\log_{10}(N_{col})$")
        ax.set_ylabel(r"$\log_{10}(n_{pred})-\log_{10}(n_{estimated})$")

        ax.legend()

        return fig, ax

    #-------SAVE-------

    def serialize_cores(self, region:Union[Tuple[float,float,float,float],None]=None)->str:
        """Serialize the core properties into a file named 'cores.txt' within the observation folder."""
        cores = self.get_cores()
        if cores is None:
            LOGGER.error("Cant serialize, there is no cores.")
            return
        predicted_densities, derived_densities, column_densities, indexes = self._get_cores_predicted_values(region=region, return_ncol=True, return_indexes=True)
        returned_cores = []
        for i0,i in enumerate(indexes):
            returned_cores.append({
                "name": cores[i]["name"],
                "ra": cores[i]["ra"],
                "dec": cores[i]["dec"],
                "n_estimated": derived_densities[i0],
                "n_predicted": predicted_densities[i0],
                "column_density": column_densities[i0],
            })
        string = dictsToString(returned_cores)
        with open(os.path.join(self.folder, "cores.txt"), "w") as file:
            file.write(string)
        LOGGER.log(f"Cores serialized for obs {self.name}, see the cores.txt file.")
        return string

    def save(self,replace:bool=True):
        """
        Args:
            replace (bool): if set to False, if there is an existing file then this function does nothing. 
        """
        if self.prediction is None:
            LOGGER.error(f"Can't save cache for prediction on {self.name} because there has no prediction on this observation, use .predict(model)")
            return
        if not(os.path.exists(CACHES_FOLDER)):
            os.mkdir(CACHES_FOLDER)
        path = os.path.join(CACHES_FOLDER,self.name+".npy")
        if os.path.exists(path):
            if not(replace):
                LOGGER.error(f"Can't save cache for prediction on {self.name} because there is already a cache and replace is set to False")
                return
            os.remove(path)
        LOGGER.log(f"Observation prediction {self.name} saved")
        np.save(path,self.prediction)

    def load(self)->np.ndarray:
        path = os.path.join(CACHES_FOLDER,self.name.split(".npy")[0]+".npy")
        if not(os.path.exists(path)):
            return
        self.prediction = np.load(path) 
        return self.prediction
    
def script_data_and_figures(name,crop=None,suffix=None,save_fig=False,plot_cores=True,normcol=[None,None],normvol=[None,None], show=True):
    obs = Observation(name, "column_density_map")
    name = name.replace("_","")
    fig, ax = obs.plot(norm=LogNorm(vmin=normcol[0], vmax= normcol[1]),plot_cores=plot_cores,crop=crop, force_col=True)
    suff = '_'+suffix if not(suffix is None) else ""
    if save_fig:
        fig.savefig(FIGURE_FOLDER+f"obs_{name.lower()}_columndensity{suff}.jpg")

    from networks.Trainer import load_trainer
    obs.load()
    if obs.prediction is None:
        trainer = load_trainer("UNet")
        obs.predict(trainer,patch_size=(128,128), overlap=0.5, downsample_factor=4, nan_value=-1., apply_baseline=False)
        obs.save()
    fig, ax = obs.plot(obs.prediction,plot_cores=plot_cores,norm=LogNorm(vmin=normvol[0], vmax=normvol[1]),crop=crop, force_vol=True)
    if save_fig:
        fig.savefig(FIGURE_FOLDER+f"obs_{name.lower()}_volumedensity{suff}.jpg")
    
    """from POLARIScore.utils.batch_utils import plot_batch_correlation
    fig, ax = plot_batch_correlation([(obs.data,obs.prediction)],show_yx=False)
    ax.set_xlabel(r"Column density ($log_{10}(cm^{-2})$)")
    ax.set_ylabel(r"Mass-weighted density ($log_{10}(cm^{-3})$)")
    fig.tight_layout()
    if save_fig:
        fig.savefig(FIGURE_FOLDER+f"obs_{name.lower()}_correlation.jpg")
    """
        
    print(f"Max: {np.nanmax(obs.prediction)}, percentiles(10%,50%,90%,95%): {np.nanpercentile(obs.prediction,[10,50,90,95])}")

    if show:
        plt.show()

if __name__ == "__main__":

    #Orion A cropped_region = [Angle("5h36m20s").deg, Angle("5h33m30s").deg, Angle("-6d03m").deg, Angle("-4d55").deg]

    # Orion B cropped_regions
    #cropped_region = [Angle("5h49m").deg, Angle("5h45").deg, Angle("-0d19m").deg, Angle("0d53m").deg]
    #script_data_and_figures("OrionB", suffix="NGC20712068_cores", normcol=[1e21,None], normvol=[0.5e1,1e5], save_fig=True, crop=cropped_region, show=True, plot_cores=True)
    #cropped_region = [Angle("5h42m56s").deg, Angle("5h40m28s").deg, Angle("-2d32m").deg, Angle("-1d28m").deg]
    #script_data_and_figures("OrionB", suffix="NGC20232024_cores", normcol=[1e21,None], normvol=[0.5e1,1e5], save_fig=True, crop=cropped_region, show=True, plot_cores=True)

    #script_data_and_figures("Taurus_L1495", normcol=[0.5e21,3e22], normvol=[1e1,2.5e4], save_fig=True, plot_cores=False, show=True)

    obs = Observation("OrionB","column_density_map")
    obs.load()
    #fig, axes = obs.plot(norm=LogNorm(vmin=1e21, vmax= 1e24),plot_cores=(None, 10**22.2), force_col=True)
    #fig.savefig(FIGURE_FOLDER+"OrionB_Test.jpg")
    obs.plot_dcmf(bins=10)
    #obs.plot_cores_hist()
    #fig, ax = obs.plot_validity_with_model("batch_highres_2", patch_size=(512,512), c_x=compute_smoothness, c_y=lambda x: np.log10(np.mean(x)), logspace=False)
    #ax.set_xlim([0,0.25])
    #ax.set_ylim([20,24])
    #obs.plot_cores_error(mov_average=15)
    plt.show()