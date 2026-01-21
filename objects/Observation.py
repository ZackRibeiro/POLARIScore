import os
import sys
from astropy.io import fits
from astropy.wcs import WCS
from matplotlib import axes

from POLARIScore.utils.physics_utils import PC_TO_CM, plot_lognorm, plot_imf_chabrier, dcmf_func, CONVERT_NH_TO_EXTINCTION
from POLARIScore.config import *
import matplotlib.pyplot as plt 
import numpy as np
from POLARIScore.utils.utils import *
from POLARIScore.utils.observation_utils import get_clumps
from matplotlib.colors import LogNorm, NoNorm, CenteredNorm
import torch
import torch.nn.functional as F
from astropy.coordinates import SkyCoord, Angle
from astropy.wcs.utils import pixel_to_skycoord, skycoord_to_pixel
import astropy.units as u
import re
from POLARIScore.networks.Trainer import Trainer
from POLARIScore.objects.Dataset import getDataset
from typing import Dict, List, Optional, Tuple, Union, Literal
from scipy.stats import lognorm
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from POLARIScore.scripts.plotORIONsimDCMF import plot_sim_dcmf
from POLARIScore.utils.batch_utils import compute_smoothness
from POLARIScore.utils.physics_utils import CONVERT_massn_TO_n_coldens
from POLARIScore.objects.DenseCore import DenseCore
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm

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
    def __init__(self,name:str,file_name:str,distance:float=400.):

        self.name:str = name
        self.folder:str = os.path.join(OBSERVATIONS_FOLDER, name)
        """Path to the folder where the observation is stored"""

        self.distance = distance
        """Distance (in parsec) to the observation"""

        file_name = file_name.split(".fits")[0]+".fits"
        self.file:str = os.path.join(self.folder,file_name)
        """Path to the observation data"""
        self.data:np.ndarray = None
        self.prediction:np.ndarray = None
        self.prediction_error:np.ndarray = None
        self.wcs: 'WCS' = None
        self.cores: List[Dict] = None
        """Cores [{...core1_properties}]"""
        self.catalog_name = "Könyves et al, 2020"

        self.skeleton = None
        """Skeleton map"""

        self.init()
    
    def init(self):
        file = fits.open(self.file)
        f = file[0]
        self.data = np.clip(f.data*2.,a_min=0.,a_max=None) #Obs are in N_H2, models are trained on N_H
        self.wcs = WCS(f.header)
        file.close()

    def predict_using_function(self,function: Callable,nan_value=1e19,chunk_size=10000):
        data = self.data
        n = data.shape[0]

        prediction = np.empty_like(data, dtype=float)
        #self.data = np.memmap("file.npy", dtype="float64", mode="r")
        if nan_value < 0:
            nan_value = float(np.nanmin(data[data > 0]))

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)

            chunk = data[start:end].copy()
            nan_mask = np.isnan(chunk) | (chunk <= 0)
            chunk[nan_mask] = nan_value

            chunk = 10 ** (function(np.log10(chunk)))
            prediction[start:end] = chunk

        self.prediction = prediction
        return prediction    

    def predict(self, model_trainer:'Trainer', patch_size:Tuple[int,int]=(128, 128), nan_value:float=-1.0, overlap:float=0.5, downsample_factor:float=1., apply_baseline:bool=True)->np.ndarray:
        """
        Predict a quantity by applying a neural network to an observation.
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
                output_patch = model_trainer.predict_tensor(patch, input_names="cdens", output_names="vdens")[0]
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
    
    def apply_filter(self, method:Literal["gaussian"]="gaussian", factor=1., original_beam=18.2, replace=True):
        """
        Apply filter, e.g convolve by a gaussian beam with a width of original_beam*factor.
        """

        try:
            pixscale_deg = np.abs(self.wcs.pixel_scale_matrix.diagonal()).mean()
        except AttributeError:
            cdelt = self.wcs.wcs.cdelt
            pixscale_deg = np.mean(np.abs(cdelt))

        sigma_pixels = (original_beam*factor / (2*np.sqrt(2*np.log(2)))) / (pixscale_deg*3600)
        
        if method == "gaussian":
            smoothed = gaussian_filter(self.data, sigma=sigma_pixels)
            if replace:
                self.data = smoothed
            return smoothed
        else:
            raise NotImplementedError()

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

    def get_cores(self, force_compute:bool=False, use_deconvolved_values:bool=False)->Union[List['DenseCore'], None]:
        """
        Get cores from files "observed_core_catalog.txt" and "derived_core_catalog.txt" (works with Herschel Gound Belt Survey)
        Args:
            force_compute (bool): If False, the function will return the cached version of cores if available.
            use_deconvolved_values (bool): If True, use deconvolved radius and density.
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
                    "average_n": float(properties[(15 if use_deconvolved_values else 14)+offset_index])*1e4*2,
                    "mass": float(properties[6+offset_index]),
                    "radius_pc": float(properties[(4 if use_deconvolved_values else 5)+offset_index]),
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

        cores = [DenseCore(self, c) for c in cores]
        self.cores = cores
        return cores

    def get_skeleton(self, force_compute:bool=False, file_name="skeleton_map"):
        if self.skeleton is not None and not(force_compute):
            return self.skeleton
        try:
            file = fits.open(os.path.join(self.folder,file_name+".fits"))
        except:
            LOGGER.error(f"Can't load skeleton map on the observation {self.name} > No file named {file_name}.")
            return
        f = file[0]
        skeleton = f.data
        skeleton[skeleton > 0] = 1
        self.skeleton = skeleton
        file.close()

        return self.skeleton

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
            [core.data["ra"] for core in self.cores],
            [core.data["dec"] for core in self.cores],
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

        return values
        
    def _get_cores_predicted_values(self, region:Union[Tuple[float,float,float,float],None]=None, return_ncol:bool=False, return_indexes:bool=False, correction:bool=True):
        """
        Returns values in LOG10
        Deprecated, use DenseCore object instead
        Args:
            region: [ra_max, ra_min, dec_min, dec_max]
            return_ncol: Return column density
            return_indexes: Return indexes
            correction: if True apply density correction
        """
        predicted_densities = np.array(self.get_predicted_density_at_cores())
        derived_densities =  np.array([c.data["average_n"] for c in self.get_cores()])
        global_indexes = np.array(range(predicted_densities.shape[0]))
        mask = (~np.isnan(predicted_densities)) & (predicted_densities > 0) & (derived_densities > 0)
        if region is not None:
            ra_max, ra_min, dec_min, dec_max = region
            ra = np.array([c.data["ra"] for c in self.get_cores()])
            dec = np.array([c.data["dec"] for c in self.get_cores()])
            region_mask = (ra >= ra_min) & (ra <= ra_max) & (dec >= dec_min) & (dec <= dec_max)
            mask = mask & region_mask
        predicted_densities = predicted_densities[mask]
        column_densities = np.array(self.get_predicted_density_at_cores(column_density=True))[mask]
        if correction:
            predicted_densities = CONVERT_massn_TO_n_coldens(column_densities,10,predicted_densities,np.array([c.data["radius_pc"] for c in self.get_cores()])[mask],is_density=False)
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

    def pc_to_pixels(self, pc):
        """Convert a physical scale (pc) into pixel scale."""
        try:
            pixscale_deg = np.abs(self.wcs.pixel_scale_matrix.diagonal()).mean()
        except AttributeError:
            pixscale_deg = np.mean(np.abs(self.wcs.wcs.cdelt))
        pixscale_rad = np.deg2rad(pixscale_deg)
        pc_per_pix = self.distance * pixscale_rad
        return pc / pc_per_pix

    def rectify_error_baseline(self, revert:bool=False)->np.ndarray:
        """
        Allows to align the validation error means on 0. So it removes the error means on the predicted map. <br />
        **Be careful** when you use it because maybe you already applied residuals fitting when applying the neural network on observation (if apply_baseline was True).
        Args:
            revert: If true go backward, i.e add the mean to the predicted map.
        """
        assert self.prediction is not None, LOGGER.error("Can't rectify baseline of prediction if there is no prediction loaded in the observation.")
        assert self.prediction_error is not None, LOGGER.error("Can't rectify baseline of prediction if there is no baseline (error) loaded in the observation.")
        
        log10_pred = np.log10(self.prediction)

        bin_centers, _, _, means = self.prediction_error            
        interp_mean = np.interp(log10_pred, bin_centers, means)

        return 10**(log10_pred+interp_mean) if revert else 10**(log10_pred-interp_mean)

    #-------PLOT-------

    def plot(self, data:np.ndarray=None, norm=LogNorm(), ax:'axes.Axes'=None,
             plot_cores:Union[bool,Tuple[Union[float,None],Union[float,None]]]=False, cores_color:Optional[str]=None,
             plot_skeleton:bool= False,
             crop:Union[Tuple[float,float,float,float],None]=None, force_vol:bool=False, force_col:bool=False,
             cbar:bool=True, sbar:float=5., show_ax_labels:bool=True, toplabel:Optional[str]=None,
             sbar_transparent:bool=False, cmap:Optional[str]=None, clabel:Optional[str]=None):
        """
        Plot observation.
        Args:
            data: by default column densities of the observation, but can be predicted densities...
            norm: matplotlib norm
            plot_cores: can be a bool or a Tuple of float. If this is a tuple, this sets the column densities limit where a core will be drawn
            plot_skeleton: plot skeleton map (filaments) above the data map.
            crop: [ra_min, ra_max, dec_min, dec_max]
            force_vol: Force volume density labels.
            force_col: Force column density labels.
            cbar: Plot the color bar
            clabel: Label of the color map
            sbar: Plot a scale bar with size being sbar
            sbar_transparent: If True no opaque box behind the scale
            show_ax_labels: Plot ticks
            toplabel: Add text at the top left of the region.
        """

        if plot_cores is not None:
            plot_cores_lims = [0, 1e23]
            if(type(plot_cores) is not bool):
                if((type(plot_cores) is tuple or type(plot_cores) is list) and len(plot_cores) >= 2):
                    plot_cores_lims = plot_cores
                plot_cores = True

        data = self.data if data is None else data

        if crop is not None:
            x_min, x_max, y_min, y_max = _crop(self.wcs, crop)

            x_min, x_max = int(x_min), int(x_max)
            y_min, y_max = int(y_min), int(y_max)

            data = data[y_min:y_max, x_min:x_max]
            wcs = self.wcs.slice((slice(y_min, y_max), slice(x_min, x_max)))
        else:
            wcs = self.wcs

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(projection=wcs)
        else:
            fig = ax.figure


        flag_vol_density = False
        label = r"$N_H(cm^{-2})$" if clabel is None else clabel
        if not(force_col) and (np.nanpercentile(data,50) < 1e10 or force_vol):
            flag_vol_density = True
            label=r"$<n_H>_m(cm^{-3})$"  if clabel is None else clabel
        im = ax.imshow(data, cmap="rainbow" if cmap is None and len(data.shape) == 2 else cmap, norm=norm if len(data.shape) == 2 else None)
        overlay = ax.get_coords_overlay('fk5')
        if show_ax_labels:
            overlay.grid(color='black', ls='dotted')
            overlay[0].set_axislabel('ra')
            overlay[1].set_axislabel('dec')
        else:
            for coord in overlay:
                coord.set_ticks_visible(False)
                coord.set_ticklabel_visible(False)
                coord.set_axislabel('')
            for coord in ax.coords:
                coord.set_ticks_visible(False)
                coord.set_ticklabel_visible(False)
                coord.set_axislabel('')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        if cbar and len(data.shape) == 2:
            plt.colorbar(im, label=label)

        if plot_skeleton:
            self.plot_skeleton(ax)

        if plot_cores:
            self.plot_cores(ax, norm=norm, vol_density=flag_vol_density, lims=plot_cores_lims, color=cores_color, opacity=0.7)

        if sbar > 0.:
            scale_bar_px = self.pc_to_pixels(sbar)
            fontprops = fm.FontProperties(size=9)

            scalebar = AnchoredSizeBar(
                ax.transData,
                scale_bar_px,
                f"{sbar:.0f} pc",
                loc="lower right",
                pad=0.4,
                color="black",
                frameon=not(sbar_transparent),
                size_vertical=.0,
                fontproperties=fontprops,
            )

            ax.add_artist(scalebar)

        if toplabel is not None:
            ax.text(0.02, 0.98,toplabel,transform=ax.transAxes,
            ha="left",va="top",fontsize=10,color="black", bbox=dict(facecolor="white",edgecolor="black", boxstyle="round,pad=0.2",alpha=1.) if not(sbar_transparent) else None)

        return fig, ax
    
    def plot_correlation(self, ax:Optional["axes.Axes"]=None, bins_number=128, show_yx:bool=True, lines:List[float]=[0,1,2]):
            if ax is None:
                fig, ax = plt.subplots()
            else:
                fig = ax.figure


            mask = (~np.isnan(self.prediction)) & (self.prediction > 0) & (~np.isnan(self.data)) & (self.data > 0)
            data = np.log10(self.data[mask]).flatten()
            pred = np.log10(self.prediction[mask]).flatten()

            _, _, _,hist = ax.hist2d(data, pred, bins=(bins_number,bins_number), norm=LogNorm())
            if show_yx:
                yx = np.linspace(np.min(data), np.max(data), 10)
                p = ax.plot(yx,yx,linestyle="--",color="red",label=r"$y=x$")

            plt.colorbar(hist, ax=ax, label="counts")
            ax = plt.gca()
            ax.set_xlim([20, 24])
            ax.set_ylim([1, 8])

            plot_lines(ax,x=data, y=pred, lines=lines)

            ax.grid(True)
            ax.set_axisbelow(True)

            ax.set_xlabel(r"measured $N_H$ ($cm^{-2}$)")
            ax.set_ylabel(r"predicted $<n_H>_m$ ($cm^{-3}$)")

            fig.tight_layout()

            return fig, ax

    def plot_density_distributions(self, ax:Optional["axes.Axes"]=None, bins:int=20, monte_carlo:int=10, offset_method:Literal["mean","max","wout_ncol"]="mean", 
                                   color:Optional[str]="red", label:Optional[str]=None, marker:Optional[str]=None, draw_style:Optional[str]="steps-mid"):
        """
        Plot PDF (histogram) of column density and predicted volume density.
        Args:
            ax: matplotlib axis
            bins: number of bins in the PDFs
            monte_carlo (int): if prediction error available, will compute histogram error by sampling N times from error assuming it's gaussian
            offset_method (str): Method used to 'normalize' x, i.e to align column density and volume density. If offset_method='wout_ncol' then no normalization is made because column density will not be shown.
            color (str): Color for the volume density PDF.
            label (str): Label for the volume density PDF.
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        mask = (~np.isnan(self.prediction)) & (self.prediction > 0) & (~np.isnan(self.data)) & (self.data > 0)

        log10_coldens = np.log10(self.data[mask])
        log10_pred = np.log10(self.prediction[mask]).flatten()

        hist_cd, _ = np.histogram(log10_coldens, bins=bins+1, density=False)
        hist_cd_stats_error = np.sqrt(hist_cd)/hist_cd
        hist_cd, bin_edges_cd = np.histogram(log10_coldens, bins=bins+1, density=True)
        bin_centers_cd = 0.5 * (bin_edges_cd[1:] + bin_edges_cd[:-1])

        bin_edges_pr = np.linspace(np.min(log10_pred), np.max(log10_pred), bins + 1)
        hist_pr, _ = np.histogram(log10_pred, bins=bin_edges_pr, density=False)
        hist_pred_stats_error = np.sqrt(hist_pr)/hist_pr
        hist_pr, bin_edges_pr = np.histogram(log10_pred, bins=bin_edges_pr, density=True)
        bin_centers_pr = 0.5 * (bin_edges_pr[1:] + bin_edges_pr[:-1])

        def _normalize_x(hist, bin_centers):
            if offset_method == "mean":
                return (bin_centers - bin_centers[np.argmin(np.abs(hist-np.mean(hist)))]) / (np.max(bin_centers) - np.min(bin_centers))
            elif offset_method == "wout_ncol":
                return bin_centers
            else:
                return (bin_centers - bin_centers[np.argmax(hist)]) / (np.max(bin_centers) - np.min(bin_centers))
        
        bin_centers_cd = _normalize_x(hist_cd, bin_centers_cd)
        bin_centers_pr = _normalize_x(hist_pr, bin_centers_pr)

        if offset_method != "wout_ncol":
            ax.plot(10**bin_centers_cd, hist_cd, drawstyle=draw_style, marker=marker, color="black", label=r"$N_H$ [$cm^{-2}$]")
            ax.errorbar(10**bin_centers_cd, hist_cd, yerr=hist_cd_stats_error*hist_cd, fmt='none', color="black")
        ax.plot(10**bin_centers_pr, hist_pr, drawstyle=draw_style, marker=marker, color=color, label=r"$<n_H>_m$ [$cm^{-3}$]" if label is None else label)
        ax.errorbar(10**bin_centers_pr, hist_pr, yerr=hist_pred_stats_error*hist_pr, fmt='none', color=color)

        if self.prediction_error is not None and monte_carlo > 0:
            try:
                bin_centers, q1, q2, _ = self.prediction_error
            except:
                LOGGER.error("Density error is not in the good format in DenseCore -> Can't sample a random density given the error.")
            q1_interp = np.interp(log10_pred, bin_centers, q1)
            q2_interp = np.interp(log10_pred, bin_centers, q2)
            gauss_sigma = (q2_interp-q1_interp)/(2*1.64485)

            all_pred_hists = []
            for mc in range(monte_carlo):
                printProgressBar(mc, monte_carlo, prefix="MC-Dist", length=20)
                random_predicted_densities = np.random.normal(loc=log10_pred, scale=gauss_sigma)
                rd_hist, _ = np.histogram(random_predicted_densities, bins=bin_edges_pr, density=True)
                all_pred_hists.append(rd_hist)
            print("")
            all_pred_hists = np.array(all_pred_hists)
            hist_std = np.std(all_pred_hists, axis=0)
            x_step, y_lower_step, y_upper_step = step_fill(10**bin_centers_pr, hist_pr - 2*hist_std,
                hist_pr + 2*hist_std, log_bins=True, offset=1.0)
            ax.fill_between(x_step, y_lower_step, y_upper_step, color=color, alpha=0.3)
            
        if offset_method == "mean":
            ax.set_xlabel(r"($x-\mu) / (max(x)-min(x))$")
        elif offset_method == "wout_ncol":
            ax.set_xlabel(r"$<n>_m$ [$cm^{-3}$]")
        else:
            ax.set_xlabel(r"($x-max(x)) / (max(x)-min(x))$")
        ax.set_ylabel("density")

        plot_lines(ax=ax, lines= [0, -1, -2], logspace=False)

        ax.set_xscale("log")
        ax.set_yscale("log")

        ax.grid(visible=True)

        ax.legend()
        
        return fig, ax

    def plot_cores(self,ax:"axes.Axes",cores:Union[List[Dict],None]=None,norm=None,vol_density:bool=False,
                   show_text:bool=False, lims:Tuple[Union[None,float],Union[None,float]]=[None,None],
                   color:Optional[str]=None, opacity:float=1.):
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
            cores = [c.data for c in self.get_cores()]
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


        ax.scatter(x_pix, y_pix, s=radius/pixel_scale, facecolors=colors if color is None else color, alpha=opacity, edgecolors="black")
        if show_text:
            for i,c in enumerate(cores):
                ax.text(x_pix[i], y_pix[i], f"${values[i]:.2e}$", color='black')

        return ax
    
    def plot_skeleton(self, ax:"axes.Axes", skeleton:Optional[np.ndarray]=None):
        if skeleton is None:
            skeleton = self.get_skeleton(force_compute=False)
        if skeleton is None:
            LOGGER.warn("Can't plot skeleton above the map > No skeleton found.")
            return
        #skeleton_masked = np.ma.masked_where(self.skeleton == 0, self.skeleton)
        
        ax.contour(self.skeleton, levels=[0.5], colors="black", linewidths=1.2)

        return ax

    def plot_validity_with_model(self, dataset_name:str="batch_highres", ax:Optional["axes.Axes"]=None, c_x:Callable[[np.ndarray],float]=np.min, c_y:Callable[[np.ndarray], float]=np.max, logspace=True, patch_size:Tuple[int,int]=(128, 128), nan_value:float=None, overlap:float=0.5):

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

    def plot_cores_hist(self, ax:Optional["axes.Axes"]=None, region:Union[Tuple[float,float,float,float],None]=None, linestyle:bool=False, drawstyle:Optional[str]="steps-mid", bins:int=15, plot_catalog:bool=True, label:Optional[str]=None
                        , correction:bool=True):
        """
        Args:
            ax: matplotlib axis
            region: [ra_max, ra_min, dec_min, dec_max]
            linestyle (bool): If True use linestyles instead of colors
            drawstyle: Drawstyle argument of plt.plot function
            bins (int): number of bins 
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        predicted_densities, derived_densities = self._get_cores_predicted_values(region=region, correction=correction)

        def _get_hist(densities:np.ndarray):
            log_min, log_max = (3, 6.5)
            bin_edges = np.linspace(log_min, log_max,  bins)
            hist, bin_edges = np.histogram(densities, bins=bin_edges)
            bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
            return hist, 10**bin_centers
        
        hist_pr, bins_pr = _get_hist(predicted_densities)
        hist_dr, bins_dr = _get_hist(derived_densities)

        ax.plot(bins_pr, hist_pr, drawstyle=drawstyle, marker="+", linestyle="--" if linestyle else "-", color="black" if linestyle else None, label=f"{self.name} (Neural network)" if label is None else label)
        if plot_catalog:
            ax.plot(bins_dr, hist_dr, drawstyle=drawstyle, marker="+", linestyle="-" if linestyle else "-", color="black" if linestyle else None, label=f"{self.name} ({self.catalog_name})")
        
        ax.set_xlabel(r"$n_H (cm^{-3})$")
        ax.set_ylabel("Number of cores per log bin")
        ax.set_xscale("log")
        ax.set_yscale("log")

        ax.legend()

        return fig, ax
    
    def plot_cores_hist2d(self, ax:Optional["axes.Axes"]=None, region:Union[Tuple[float,float,float,float],None]=None):
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

    def plot_cores_baseline(self, ax:Optional["axes.Axes"]=None, suffixes:Optional[Union[List[str],str]]=None, derived_cores:bool=False, density_correction:bool=True,
                             x_coldens:bool=False, invert_xy:bool=False, mov_average:int=0, fit:bool=False, cmap_color=True, forced_label=None):
        """
        Plot dense cores baseline with x axis depending on args. By default this plot the predicted mass-weighted average density of cores in function of their id.
        Args:
            suffixes: If load multiple predictions caches, e.g _cinn to load OrionB_cinn.npy as prediction if name of observation is OrionB.
            derived_cores: If true, also add the derived catalog cores on the plot.
            density_correction: Transform the mass-weighted average density to core volume average density using two mediums assumptions.
            x_coldens: If true, use column density as x axis, so this will plot n(N).
            invert_xy: If true, this will invert x and y axes. For example if x_coldens is True then it will plot N(n).
            mov_average: Apply moving average on the points.
            fit: Try to fit the function using a modified power law (used for N(n) and n(N)). If this is true, the points that will be shown on the plot are the log binned average points, same points used for the fit.
        """

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        if suffixes is not None:
            suffixes = suffixes if type(suffixes) is list else [suffixes]
        else:
            suffixes = [""]

        colors = FIGURE_CMAP(np.linspace(FIGURE_CMAP_MIN, FIGURE_CMAP_MAX, len(suffixes)))

        def _make_baseline(densities, label, color):
            if x_coldens:
                col_densities = np.array([c.get_center_density(column_density=True) for c in cores])
            x = np.array(range(len(densities)))

            mask = (~np.isnan(densities)) & (~np.isinf(densities)) & (densities > 0.)
            densities = densities[mask]
            x = x[mask]
            if x_coldens:
                col_densities = col_densities[mask]
                sorted_indexes = np.argsort(col_densities)
                x = col_densities[sorted_indexes]
                densities = densities[sorted_indexes]

            if invert_xy:
                temp = x
                x = densities
                densities = temp
            if fit and x_coldens:
                def _fit_function(x, a, b):
                    return x**a*np.exp(b)
                def _fit_function_inv(x, a, b, alpha):
                    return np.log(a*x**np.abs(alpha)+np.abs(b))

                binned_x, binned_y = bin_mean(x, densities, dx=0.1,min_per_bin=3)

                try:
                    if invert_xy:
                        popt, _ = curve_fit(_fit_function_inv, binned_x, np.log(binned_y), p0=[(np.max(densities)-np.min(densities))/(np.max(x)-np.min(x)), np.min(densities), 0.5],
                                            bounds=([0, 1e21, 0],[np.inf, 1e22, 1]))
                        fit_a, fit_b, fit_alpha = popt
                        func = lambda X: np.exp(_fit_function_inv(X, fit_a, fit_b, fit_alpha))
                        plot_function(func, ax=ax, scatter=False, logspace=True, lims= (np.min(x), np.max(x)), color=color, linestyle="--")
                        c_mu = 2.3 #maybe 1.4
                        c_mh = 1.67e-24
                        c_cs = fit_a* ((c_mu*c_mh*6.674e-8)/np.pi)**(0.5)
                        c_kb = 1.38e-16 
                        c_T = c_cs**2*(c_mu*c_mh/c_kb)
                        #label = rf"{label}: T={c_T:.3}K, $\sum_{{i\neq c}} n_i l_i$={fit_b:.3} cm$^{{-2}}$, $\alpha$={fit_alpha:.3}"
                        label = rf"{label}: T={c_T:.3}K, $\alpha$={fit_alpha:.3}"
                    else:
                        popt, _ = curve_fit(_fit_function, binned_x, binned_y, p0=[1., 0.])
                        fit_a, fit_b = popt
                        func = lambda X: _fit_function(X, fit_a, fit_b)
                        plot_function(func, ax=ax, scatter=False, logspace=True, lims= (np.min(x), np.max(x)), color=color, linestyle="--")
                        label = label + f": {fit_a:.3}, {fit_b:.3}"
                except:
                    LOGGER.warn("Dense cores baseline fit failed")
            
            if mov_average > 0:
                x = moving_average(x, n=mov_average)
                densities = moving_average(densities, n=mov_average)

            #ax.plot(x, densities, lw=1., color=color)
            ax.scatter(binned_x if fit else x, binned_y if fit else densities, marker="+", color=color, label=label)

        for i,s in enumerate(suffixes):
            if s != "" or self.prediction is None:
                self.load(suffix=s)
            cores = self.get_cores()
            if cores is None:
                LOGGER.warn(f"No cores found in {s}.")
                continue
            densities = np.array([c.get_center_density(correction=density_correction) for c in cores])
            _make_baseline(densities, s.replace("_","") if forced_label is None else forced_label, colors[i] if cmap_color else None)
        if derived_cores:
            _make_baseline(np.array([c.data['average_n'] for c in cores]), self.catalog_name, "red")


        ax.set_yscale("log")
        ax.set_xlabel("Core index")
        ax.set_ylabel("$n_H$ [cm^-3]" if density_correction else "$<n>_m$ [cm^-3]")
        

        if x_coldens:
            ax.set_xscale("log")
            ax.set_xlabel("$N_H$ [cm^-2]")
            if invert_xy:
                ax.set_xlabel("$n_H$ [cm^-3]" if density_correction else "$<n>_m$ [cm^-3]")
                ax.set_ylabel("$N_H$ [cm^-2]")
        
        ax.grid(which='both', axis='x')


        ax.legend()

        return fig, ax

    def plot_fractal_dim(self, ax:Optional["axes.Axes"]=None, suffixes:Optional[List[str]]=None, distance:Optional[float]=None, thresholds:List[float]=[0.85], colors:Optional[List[str]]=None):
        """
        Plot Perimeter vs Area of clumps identified in predicted volume density map(or in column density if suffixes='!COLUMN_DENSITY'). 
        If thresholds contains more than one threshold, then this will plot Fractal Dimension vs thresholds. 
        Clumps identified using skimage algorithm.
        Args:
            ax: matplotlib axis
            suffixes: If load multiple predictions caches, e.g _cinn to load OrionB_cinn.npy as prediction if name of observation is OrionB.
            distance: Physical distance between Earth and Cloud in parsec. (By default: 400pc or distance already given in a previous function call.)
            thresholds: Can be lower of higher than 1.: if lower -> compute quantile on the density map, if higher -> use it as a density value. 
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        if suffixes is not None:
            suffixes = suffixes if type(suffixes) is list else [suffixes]
        else:
            suffixes = ["!COLUMN_DENSITY"]

        try:
            pixscale_deg = np.abs(self.wcs.pixel_scale_matrix.diagonal()).mean()
        except AttributeError:
            cdelt = self.wcs.wcs.cdelt
            pixscale_deg = np.mean(np.abs(cdelt))

        if distance is None:
            distance = self.distance
        else:
            self.distance = distance
        pixscale_rad = np.deg2rad(pixscale_deg)
        pc_per_pix = distance * pixscale_rad

        colors = FIGURE_CMAP(np.linspace(FIGURE_CMAP_MIN, FIGURE_CMAP_MAX, len(suffixes))) if colors is None else colors

        def _fit_function(x, a, b):
            return (a/2)*x+b

        D = [[] for i in range(len(suffixes))]
        D_err = [[] for i in range(len(suffixes))]
        T = [[] for i in range(len(suffixes))]
        for i,s in enumerate(suffixes):
            label = s.replace("_","")
            data = self.data
            if s != "!COLUMN_DENSITY":
                self.load(suffix=s)
                data = self.prediction
            for j,t in enumerate(thresholds):
                printProgressBar(i*len(thresholds)+j, len(thresholds)*len(suffixes), length=20, prefix='Finding clumps')
                P,A = get_clumps(data, threshold=t)
                P = P * pc_per_pix
                A = A * pc_per_pix**2

                x = np.log(A)
                y = np.log(P)
                popt, pcov = curve_fit(_fit_function, x, y, p0=[1., 0.])
                fit_a, fit_b = popt
                if len(thresholds) < 2:
                    func = lambda X: np.exp(_fit_function(np.log(X), fit_a, fit_b))
                    plot_function(func, ax=ax, scatter=False, logspace=True, lims= (np.min(A), np.max(A)), color=colors[i], linestyle="--")
                    label = label + f", D={(fit_a):.3}"
                    ax.scatter(A, P, marker="+", color=colors[i], label=label)
                else:
                    if(len(x) > 20):
                        D_err[i].append(np.sqrt(pcov[0,0]))
                        D[i].append(fit_a)
                        T[i].append(float(np.nanpercentile(data, t*100) if t < 1 else t))
        print("")
        if len(D) > 0 and len(thresholds) > 1:
            for i in range(len(D)):
                ax.errorbar(T[i],D[i],yerr=D_err[i],color=colors[i],label=suffixes[i].replace("_", ""),linestyle="-")

        if len(thresholds) < 2:
            ax.set_xlabel(r"Area [$pc^2$]")
            ax.set_ylabel("Perimeter [pc]")
            ax.set_xscale("log")
            ax.set_yscale("log")
        else:
            ax.set_xlabel(r"Threshold $<n>_H$ [$cm^{-3}$]" if suffixes[0] != "!COLUMN_DENSITY" else r"Threshold $N_H$ [$cm^{-2}$]")
            ax.set_ylabel("D")
            ax.set_xscale("log")
        ax.legend()
        ax.grid()

        return fig, ax

    def plot_dcmf(self, ax:Optional["axes.Axes"]=None, bins:int=10, ext_lims:Tuple[Union[float,None],Union[float,None]]=[None,None],
                   logM:bool=True, fit=True, method:Literal['constant','gaussian']="constant",
                    monte_carlo:int=100 , correction:bool=True 
                ):
        """
        Plot the dense core mass function
        Args:
            ax: matplotlib axis
            bins: number of bins in the DCMF
            ext_lims: Choose only the dense cores between the two extinctions Av given.
            logM: plot dN/dlogM or dN/dM.
            fit: try to fit the dcmf by a lognormal with power law.
            method: method used to compute the core mass.
            monte_carlo: Compute a dcmf error by taking sample from error distribution of dense core mass. Integer controls how many samples are taken.
        """

        if ext_lims[0] is None:
            ext_lims[0] = -1000
        if ext_lims[1] is None:
            ext_lims[1] = 1000.

        cores = self.get_cores()

        predicted_masses = np.array([c.compute_mass(method=method,correction=correction) for c in cores])
        derived_cores = [c.data for c in cores]
        predicted_densities = np.array(self.get_predicted_density_at_cores(), dtype=np.float64)
        column_densities = np.array(self.get_predicted_density_at_cores(column_density=True), dtype=np.float64)
        derived_densities =  np.array([c.data["average_n"] for c in cores], dtype=np.float64)
        derived_masses = np.array([c.data["mass"] for c in cores], dtype=np.float64)
        mask = (~np.isnan(predicted_densities)) & (predicted_densities > 0) & (derived_densities > 0) & (~np.isnan(predicted_masses))
        column_densities = column_densities[mask]
        predicted_masses = predicted_masses[mask]
        derived_masses = derived_masses[mask]

        extinctions = []
        derived_radius = []
        for i in range(len(cores)):
            extinctions.append(CONVERT_NH_TO_EXTINCTION(cores[i].get_center_density(column_density=True)))
            derived_radius.append(derived_cores[i]['radius_pc'])
        extinctions = np.array(extinctions)[mask]
        derived_radius = np.array(derived_radius, dtype=np.float64)[mask]
        derived_densities = derived_densities[mask]

        global_mask = (extinctions >= ext_lims[0]) & (extinctions <= ext_lims[1])

        def _get_dcmf(masses:np.ndarray):
            m = masses[global_mask]
            m = np.log10(m)
            log_min, log_max = (-2, 2)
            bin_edges = np.linspace(log_min, log_max, bins + 1)
            hist, bin_edges = np.histogram(m, bins=bin_edges)
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
            fig = ax.figure

        derived_masses = _compute_mass(derived_radius,derived_densities)
        derived_dcmf, derived_bin_centers = _get_dcmf(derived_masses)
        derived_bin_centers = 10**derived_bin_centers

        ax.plot(derived_bin_centers, derived_dcmf, drawstyle="steps-mid", color="blue", label=f"{self.name} ({self.catalog_name})")
        ax.scatter(derived_bin_centers, derived_dcmf, marker="^", color="blue")
        if fit:
            popt, _ = curve_fit(_dcmf_function, derived_bin_centers, derived_dcmf,
                            p0=[np.max(derived_dcmf), 1.5, np.std(np.log(derived_masses)), 2.3, 1.6])
            amp_fit, mu_fit, sigma_fit, alpha_fit ,cutoff_fit = popt
            LOGGER.log(f"Best DCMF fit for estimated cores: amp={amp_fit:.2e}, mu={mu_fit:.3f}, sigma={sigma_fit:.3f}, alpha={alpha_fit}, cutoff={cutoff_fit}")
            func = lambda X: _dcmf_function(X, popt[0], popt[1], popt[2], popt[3], popt[4])
            plot_function(func, ax=ax, scatter=False, logspace=True, lims= (0.01, 100), color="blue", linestyle="--")

        LOGGER.log(f"DCMF with {len(derived_radius[global_mask])} cores.")

        if(self.prediction is not None):            
            

            predicted_dcmf, predicted_bin_centers = _get_dcmf(predicted_masses)
            predicted_bin_centers = 10**predicted_bin_centers
            if monte_carlo > 0 and self.prediction_error is not None:
                all_pred_dcmfs = []
                for mc in range(monte_carlo):
                    printProgressBar(mc, monte_carlo, prefix="MC-DCMF", length=20)
                    random_predicted_masses = np.array([c.compute_mass(method=method, density_error=self.prediction_error) for c in cores])[mask]
                    _dcmf, _bin_centers = _get_dcmf(random_predicted_masses)
                    all_pred_dcmfs.append(_dcmf)
                print("")
                all_pred_dcmfs = np.array(all_pred_dcmfs)
                dcmf_std = np.std(all_pred_dcmfs, axis=0)
                x_step, y_lower_step, y_upper_step = step_fill(predicted_bin_centers,predicted_dcmf - 2*dcmf_std,
                    predicted_dcmf + 2*dcmf_std, log_bins=True, offset=1.03)
                #ax.errorbar(predicted_bin_centers, predicted_dcmf, yerr=dcmf_std, fmt='none', color="black")
                ax.fill_between(x_step, y_lower_step, y_upper_step, color="red", alpha=0.2)
            ax.plot(predicted_bin_centers, predicted_dcmf, drawstyle="steps-mid", color="red", label=f"{self.name} (Neural Network)")
            ax.scatter(predicted_bin_centers, predicted_dcmf, color="red")

            if fit:
                popt, _ = curve_fit(_dcmf_function, predicted_bin_centers[predicted_bin_centers>0.], predicted_dcmf[predicted_bin_centers>0.],
                            p0=[np.max(predicted_dcmf), 0.22, np.std(np.log(predicted_masses)), 2.3, 1.6])
                amp_fit, mu_fit, sigma_fit, alpha_fit ,cutoff_fit = popt
                LOGGER.log(f"Best DCMF fit for predicted cores: amp={amp_fit:.2e}, mu={mu_fit:.3f}, sigma={sigma_fit:.3f}, alpha={alpha_fit}, cutoff={cutoff_fit}")
                func = lambda X: _dcmf_function(X, popt[0], popt[1], popt[2], popt[3], popt[4])
                plot_function(func, ax=ax, scatter=False, logspace=True, lims= (0.01, 100), color="red", linestyle="--")

        #plot_sim_dcmf(ax, factor=0.035, logM=logM)
        ax.axvline(0.4, 0., 1., color='black', ls='--')
        ax.text(0.4 - 0.1,0.5,r'Completeness limit: $M=0.4M_\odot$',
            rotation=90,va='center',ha='left',color='black',fontsize=11, transform=ax.get_xaxis_transform())


        #plot_imf_chabrier(ax, logM=logM, dcmf=0.4, x_min=1e-3, amp = np.max(predicted_dcmf))
        
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"Mass [$M_\odot$]")
        ax.set_ylabel(r"$dN/d\log M$" if logM else r"$dN/dM$")

        ax.set_xlim([0.01, 100])
        ax.set_ylim([0.8,600])

        plt.legend()

        return fig, ax

    def plot_cores_mass(self, ax:Optional["axes.Axes"]=None, method:Literal['constant','gaussian']="constant", mov_average:int=0, bins_mean:int=0,
                         label:Optional[str]=None, show_errors:bool=False, linestyle:Optional[str]="-"):
        """
        Plot the derived versus predicted mass
        Args:
            ax: matplotlib axis
            
        """
        cores = self.get_cores()

        assert self.prediction is not None

        predicted_masses = np.array([c.compute_mass(method=method) for c in cores])
        predicted_densities = np.array(self.get_predicted_density_at_cores(), dtype=np.float64)
        derived_densities =  np.array([c.data["average_n"] for c in cores], dtype=np.float64)
        derived_masses = np.array([c.data["mass"] for c in cores], dtype=np.float64)
        
        mask = (~np.isnan(predicted_densities)) & (predicted_densities > 0) & (derived_densities > 0) & (~np.isnan(predicted_masses))
        predicted_masses = predicted_masses[mask]
        derived_masses = derived_masses[mask]
        derived_radius = np.array([c.data['radius_pc'] for c in cores], dtype=np.float64)[mask]
        derived_densities = derived_densities[mask]
        predicted_densities = predicted_densities[mask]

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
        
        ax_was_none = ax is None
        if ax_was_none:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        derived_masses = _compute_mass(derived_radius,derived_densities)

        sorted_indexes = np.argsort(derived_masses)
        derived_masses = derived_masses[sorted_indexes]
        predicted_masses = predicted_masses[sorted_indexes]
        predicted_densities = predicted_densities[sorted_indexes]

        has_errors = True if self.prediction_error is not None else False
        if has_errors:
            bin_centers, q1, q2, means = self.prediction_error            
            log10_pd = np.log10(predicted_densities)
            interp_mean = np.interp(log10_pd, bin_centers, means)
            interp_q1 = np.interp(log10_pd, bin_centers, q1)
            interp_q2 = np.interp(log10_pd, bin_centers, q2)
            interp_masses = np.log10(predicted_masses)
            interp_densities = np.log10(predicted_densities)
            yerr_lower = interp_mean - interp_q1
            yerr_upper = interp_q2 - interp_mean
            yerr_lower = yerr_lower / interp_densities * interp_masses
            yerr_upper = yerr_upper / interp_densities * interp_masses
            yerr_lower = 10**yerr_lower
            yerr_upper = 10**yerr_upper
            if mov_average > 0:
                yerr_lower = moving_average(yerr_lower, n=mov_average)
                yerr_upper = moving_average(yerr_upper, n=mov_average)
            elif bins_mean > 0:
                _, yerr_lower = bin_mean(derived_masses, yerr_lower, nbins=bins_mean, logspace=True, min_per_bin=3)
                _, yerr_upper = bin_mean(derived_masses, yerr_upper, nbins=bins_mean, logspace=True, min_per_bin=3)
            #ax.fill_between(column_densities, residuals-yerr_lower, yerr_upper+residuals, color="black", alpha=0.2
                            
        if mov_average > 0:
            derived_masses = moving_average(derived_masses, n=mov_average)
            predicted_masses = moving_average(predicted_masses, n=mov_average)
        elif bins_mean > 0:
            derived_masses, predicted_masses = bin_mean(derived_masses, predicted_masses, nbins=bins_mean, logspace=True, min_per_bin=3)

        if has_errors and show_errors:
            ax.errorbar(derived_masses, predicted_masses,yerr=[yerr_lower, yerr_upper],fmt='none', color="black",alpha=0.8)
        ax.plot(derived_masses, predicted_masses, marker="+", color="black", label="Cores" if label is None else label, linestyle=linestyle)

        if ax_was_none:
            plot_function(lambda x:x, ax=ax, lims=[ax.get_xlim()[0],ax.get_xlim()[1],ax.get_ylim()[0],ax.get_ylim()[1]], color="red", label="y=x")
        
            
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"Derived Mass [$M_\odot$]")
        ax.set_ylabel(r"Predicted Mass [$M_\odot$]")

        ax.grid(visible=True)

        ax.legend()

        return fig, ax

    def plot_cores_space(self, ax:Optional["axes.Axes"]=None, region:Union[Tuple[float,float,float,float],None]=None):
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
        plot_lines(ax, x=column_densities, y=merged_list)

        #ax.grid()

        return fig, ax
    
    def plot_cores_error(self, ax:Optional["axes.Axes"]=None, region:Union[Tuple[float,float,float,float],None]=None, alpha:float=1.,
                          mov_average:int=5, log_average:int=0 ,show_errors:bool=True, show_model_errors:bool=False,
                            correction:bool=True, color=None, linestyle=None, label=None):
        """
        Args:
            ax: matplotlib axis
            region: [ra_max, ra_min, dec_min, dec_max]
            alpha (float): opacity
            mov_average (int): data is downsampled/smoothed using moving average method.
            log_average (int): data is smoothed using average of log bins, this controls the number of bins.
            show_errors (bool): show std deviation induced by the moving average.
            show_model_errors (bool): show error given by the neural network validation test.
            correction(bool): If True, apply density correction (mass-weighted average to core)
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        predicted_densities, derived_densities, column_densities = self._get_cores_predicted_values(region=region, return_ncol=True, correction=correction)

        residuals = predicted_densities-derived_densities
        if mov_average > 1:
            residuals, residuals_std = moving_average(residuals, n=mov_average, return_std=True)
            predicted_densities = moving_average(predicted_densities, n=mov_average) 
            column_densities = moving_average(column_densities, n=mov_average)
        if log_average > 1:
            _, predicted_densities = bin_mean(10**column_densities, predicted_densities, dx=None,min_per_bin=2, nbins=log_average)
            column_densities, residuals = bin_mean(10**column_densities, residuals, dx=None,min_per_bin=2, nbins=log_average)
            column_densities = np.log10(column_densities)

        ax.axhline(0., color="red")

        column_densities = 10**column_densities
        if self.prediction_error is not None and show_model_errors:
            bin_centers, q1, q2, means = self.prediction_error            
            interp_mean = np.interp(predicted_densities, bin_centers, means)
            interp_q1 = np.interp(predicted_densities, bin_centers, q1)
            interp_q2 = np.interp(predicted_densities, bin_centers, q2)
            yerr_lower = interp_mean - interp_q1
            yerr_upper = interp_q2 - interp_mean

            #ax.errorbar(column_densities,residuals,yerr=[yerr_lower, yerr_upper],fmt='none', color="black",alpha=0.8)
            ax.fill_between(column_densities, residuals-yerr_lower, yerr_upper+residuals, color="black", alpha=0.2)
        line, = ax.plot(column_densities,residuals, marker="+", alpha=alpha, label=self.name if label is None else label, color=color, linestyle=linestyle if linestyle is not None else "-")
        if mov_average > 1 and show_errors:
            ax.fill_between(column_densities,residuals-residuals_std,residuals+residuals_std, color=line.get_color(), alpha=0.2)

        ax.set_xlabel(r"$N_{H}(\mathrm{cm}^{-2})$")
        ax.set_ylabel(r"$\log_{10}(n_{\mathrm{neural network}})-\log_{10}(n_{\mathrm{catalog}})$")
        ax.set_xscale("log")
        ax.grid(True, which='both', axis='x')

        ax.legend()

        return fig, ax

    def plot_power_spectrum(self, ax:Optional["axes.Axes"]=None, bins:int=30, label:Optional[str]="$<n_H>_m$", color:Optional[str]=None, plot_coldens:bool=True, normalize:bool=True):
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        pixel_size = 1./self.pc_to_pixels(1)
        def _power_spectrum_2d(map2d):
            mask = np.isnan(map2d)
            map2d[mask] = np.nanmean(map2d)
            
            map2d = map2d - np.mean(map2d)
            fft_map = np.fft.fft2(map2d)
            fft_map = np.fft.fftshift(fft_map)
            power = np.abs(fft_map)**2

            ny, nx = map2d.shape
            kx = np.fft.fftfreq(nx, d=pixel_size)
            ky = np.fft.fftfreq(ny, d=pixel_size)
            kx, ky = np.meshgrid(kx, ky)
            k = np.sqrt(kx**2 + ky**2)
            k = np.fft.fftshift(k)

            k_bins = np.linspace(0, k.max(), bins)
            Pk = np.zeros(len(k_bins)-1)
            k_centers = 0.5 * (k_bins[1:] + k_bins[:-1])

            for i in range(len(k_bins)-1):
                mask = (k >= k_bins[i]) & (k < k_bins[i+1])
                Pk[i] = power[mask].mean()

            return k_centers, Pk

        if plot_coldens:
            k_coldens, Pk_coldens = _power_spectrum_2d(self.data)
            if normalize:
                Pk_coldens = Pk_coldens / np.max(Pk_coldens)
            ax.plot(k_coldens, Pk_coldens, color="black", label="$N_H$")

        if self.prediction is not None:
            k_voldens, Pk_voldens = _power_spectrum_2d(self.prediction)
            if normalize:
                Pk_voldens = Pk_voldens / np.max(Pk_voldens)
            ax.plot(k_voldens, Pk_voldens, label=label, color=color)
        else:
            LOGGER.warn("Can't plot the power spectrum of prediction map > there is no pred map loaded.")

        ax.set_xscale("log")
        ax.set_yscale("log")

        ax.set_xlabel(r"$k\ \mathrm{[pc^{-1}]}$")
        ylabel = r"$P(k)$"
        if normalize:
            ylabel += " (normalized)"
        ax.set_ylabel(ylabel)
        ax.grid(visible=True)


        ax.legend()

        return fig, ax

    #-------SAVE-------

    def serialize_cores(self, region:Union[Tuple[float,float,float,float],None]=None)->str:
        """Serialize the core properties into a file named 'cores.txt' within the observation folder."""
        cores = [c.data for c in self.get_cores()]
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

    def save(self,replace:bool=True,suffix=""):
        """
        Args:
            replace (bool): if set to False, if there is an existing file then this function does nothing.
            suffix (str): add suffix in file name.
        """
        if self.prediction is None:
            LOGGER.error(f"Can't save cache for prediction on {self.name} because there has no prediction on this observation, use .predict(model)")
            return
        if not(os.path.exists(CACHES_FOLDER)):
            os.mkdir(CACHES_FOLDER)
        path = os.path.join(CACHES_FOLDER,self.name+suffix+".npy")
        if os.path.exists(path):
            if not(replace):
                LOGGER.error(f"Can't save cache for prediction on {self.name} because there is already a cache and replace is set to False")
                return
            os.remove(path)
        LOGGER.log(f"Observation prediction {self.name} saved")
        np.save(path,self.prediction)

    def load(self,suffix="")->np.ndarray:
        path = os.path.join(CACHES_FOLDER,self.name.split(".npy")[0]+suffix+".npy")
        if not(os.path.exists(path)):
            LOGGER.error(f"File: {path} doesn't exist -> Unable to load observation.")
            return
        self.prediction = np.load(path) 
        return self.prediction
    
    def load_error(self, model_name)->np.ndarray:
        """
        Load prediction error using the validation error computed when training the neural network.
        Validation error is a list with 4 elements:
            -bin_centers in log10 of predicted quantity
            - 3 quantiles: first two for confidence (default: 90%) and the third for mean.
            Note that the validation error is computed on the validation set without residuals fitting.
        Args:
            model_name(str): Model (folder) name.
        """
        path = os.path.join(MODEL_FOLDER, model_name, "validation_error.npy")
        if not(os.path.exists(path)):
            LOGGER.error(f"File: {path} doesn't exist -> Unable to load neural network errors.")
            return
        self.prediction_error = np.load(path)
        return self.prediction_error

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
    #cropped_region = [Angle("5h48m").deg, Angle("5h39m").deg, Angle("-3d09m").deg, Angle("-0d58m").deg]
    #script_data_and_figures("OrionB", suffix="NGC20232024_cores", normcol=[1e21,None], normvol=[0.5e1,1e5], save_fig=True, crop=cropped_region, show=True, plot_cores=True)

    #script_data_and_figures("Taurus_L1495", normcol=[0.5e21,3e22], normvol=[1e1,2.5e4], save_fig=True, plot_cores=False, show=True)
    
    def gamma_correct(rgb, gamma=0.8):
        rgb = rgb ** gamma
        return rgb/np.max(rgb)

    from POLARIScore.networks.Trainer import load_trainer
    from POLARIScore.networks.INNTrainer import INNTrainer
    from POLARIScore.networks.DDPTrainer import DDPTrainer
    from POLARIScore.config import DATA_NORMALIZATION_CDENS, DATA_NORMALIZATION_VDENS
    obs = Observation("OrionB","column_density_map")
    obs.distance = 400
    obs.load("_ddpm")
    fig, ax = obs.plot_power_spectrum(plot_coldens=False, label="ddpm", normalize=False)
    obs.load("_cinn")
    obs.plot_power_spectrum(ax=ax, plot_coldens=False, label="cinn", normalize=False)
    obs.load("_unet")
    obs.plot_power_spectrum(ax=ax, plot_coldens=False, label="unet", normalize=False)
    obs.load("_fit")
    obs.plot_power_spectrum(ax=ax, plot_coldens=False, label="fit", normalize=False)

    #obs.get_cores(use_deconvolved_values=False)
    #obs.plot_dcmf(monte_carlo=0, bins=15, fit=False) 
    #obs.get_cores(use_deconvolved_values=True, force_compute=True)
    #obs.plot_dcmf(monte_carlo=0, bins=15, fit=False) 
    #obs.plot_cores_error(mov_average=0, log_average=50, show_errors=False, show_model_errors=False,correction=True, color="black") 

    #obs.load_error(model_name="UNet")
    #delta = obs.rectify_error_baseline() - obs.predict(trainer,patch_size=(128,128), overlap=0.5, downsample_factor=obs.find_scale(3.30474,128,400), nan_value=-1., apply_baseline=True)
    #print(delta)


    #trainer = load_trainer("cINN", trainer_class=INNTrainer)
    #trainer.norms = {
    #    "cdens": DATA_NORMALIZATION_CDENS,
    #    "vdens": DATA_NORMALIZATION_VDENS,
    #}
    #for f in [1.5,2,3]:
    #    obs = Observation("OrionB","column_density_map")
    #    obs.distance = 400
    #    obs.apply_filter(factor=f)
    #    obs.predict(trainer,patch_size=(128,128), overlap=0.5, downsample_factor=obs.find_scale(3.30474,128,obs.distance), nan_value=1e20, apply_baseline=True)
    #    obs.save(suffix=f"_cinn_{str(f)}")
    #    obs.plot(data=obs.prediction, norm=LogNorm(vmin=1e2, vmax=3e5), plot_skeleton=False)

    """
    trainer = load_trainer("DDPM", trainer_class=DDPTrainer)
    trainer.norms = {
       "cdens": DATA_NORMALIZATION_CDENS,
        "vdens": DATA_NORMALIZATION_VDENS,
    }
    obs.predict(trainer,patch_size=(128,128), overlap=0.25, downsample_factor=obs.find_scale(3.30474,128,obs.distance), nan_value=1e20, apply_baseline=True)
    obs.save(suffix="_ddpm")
    obs.plot(data=obs.prediction, norm=LogNorm(vmin=1e2, vmax=3e5), plot_skeleton=False)

    trainer = load_trainer("cINN", trainer_class=INNTrainer)
    trainer.norms = {
       "cdens": DATA_NORMALIZATION_CDENS,
        "vdens": DATA_NORMALIZATION_VDENS,
    }
    obs.predict(trainer,patch_size=(128,128), overlap=0.25, downsample_factor=obs.find_scale(3.30474,128,obs.distance), nan_value=1e20, apply_baseline=True)
    obs.save(suffix="_cinn")
    obs.plot(data=obs.prediction, norm=LogNorm(vmin=1e2, vmax=3e5), plot_skeleton=False)
    """

    #print(obs.get_cores()[200].data["name"])

    #obs.get_cores()[25].plot()
    #for i,c in enumerate(obs.get_cores()):
    #    try:
    #        fig, axes = c.plot(save_path=FIGURE_FOLDER + "/cores/")
    #        fig2, axes = c.plot(save_path=FIGURE_FOLDER + "/cores/", cdens=True)
    #    except:
    #        continue
    #    plt.close(fig)
    #    plt.close(fig2)
    
    #obs.plot_validity_with_model("batch_training", patch_size=(512,512), c_x=lambda x: np.std(np.log10(x)), c_y=lambda x: np.log10(np.mean(x)), logspace=False)
    #obs.plot_validity_with_model("batch_highres_2", patch_size=(512,512), c_x=lambda x: np.std(np.log10(x)), c_y=lambda x: np.log10(np.mean(x)), logspace=False)
    plt.show()