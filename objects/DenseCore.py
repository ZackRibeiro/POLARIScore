import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import skycoord_to_pixel, pixel_to_skycoord
import astropy.units as u
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit
from scipy.integrate import quad
from POLARIScore.utils.physics_utils import PC_TO_CM, density_gaussian, CONVERT_massn_TO_n_coldens
from POLARIScore.utils.utils import plot_function
from POLARIScore.config import LOGGER
from copy import deepcopy
from typing import Literal, Dict, Tuple, List, Union
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm

class DenseCore():
    def __init__(self, obs, data:Dict):
        self.obs = obs
        """Observation instance which is host to the dense core"""
        self.data:Dict = data
        """Data of the dense core"""
        self.fit_settings = None
        """Gaussian fit settings of the better slice"""
        self.wcs = obs.wcs
        self.coord = SkyCoord(data["ra"], data["dec"], unit="deg")
        """Coordinates of the dense core in deg"""

    def get_center_density(self, correction=False, column_density=False):
        """Get volume density predicted at the presumed 2D core center
        Args:
            correction(bool, default=False): Apply correction by making the assumption of two mediums: dense core and diffuse along the l.o.s
            column_density(bool, default:False): Return the column density instead.
        """
        if column_density:
            assert self.obs.data is not None, LOGGER.error(f"No column density on the observation: {self.obs.name}.")
        else:
            assert self.obs.prediction is not None, LOGGER.error(f"No predicted density on the observation: {self.obs.name}.")
        densities = self.obs.data if column_density else self.obs.prediction
        x_pix, y_pix = skycoord_to_pixel(self.coord, self.wcs)

        x_int, y_int = int(round(float(x_pix))), int(round(float(y_pix)))
        if (0 <= y_int < densities.shape[0]) and (0 <= x_int < densities.shape[1]):
            rslt_density = densities[y_int, x_int]
            if correction and not(column_density):
                rslt_density = CONVERT_massn_TO_n_coldens(float(self.get_center_density(column_density=True)),10,float(rslt_density),float(self.data["radius_pc"]), is_density=False)
            return rslt_density
        else:
            return np.nan
    
    def compute_mass(self, method:Literal["gaussian","constant"]="gaussian", density_error:Union[np.ndarray, None]=None):
        """Compute mass of the core using predicted density.
        Args:
            method(str,default='constant'): Method used to compute the mass, if constant then this is just the volume*density, if gaussian: this is a 3D isotrope gaussian. 
            density_error: (Only works for constant method) If an error is passed, then a mass is computed using a random sample from the gaussian distribution of density.
        """

        assert self.obs.prediction is not None, LOGGER.error(f"No predicted density on the observation: f{self.obs.name}")

        m_H = 1.67e-24  # g
        mu = 1.4        # mean molecular weight for H (not H2)
        pc_to_cm = PC_TO_CM
        Msun = 1.989e33 # g

        r_px = self.obs.pc_to_pixels(self.data["radius_pc"])
        pix_to_pc = 5 / self.obs.pc_to_pixels(5)

        if method in ["gaussian"]:
            fit_set = self.fit()
            if fit_set[0] is None and fit_set[1] is None:
                return np.nan

            if fit_set[0] is not None:
                popt_v = deepcopy(fit_set[0])
                err_v = self.fit_error(popt_v, fit_set[2])
            if fit_set[1] is not None:
                popt_h = deepcopy(fit_set[1])
                err_h = self.fit_error(popt_h, fit_set[3])


            if(self.data["peak_ncol"] < 1e22):
                if fit_set[0] is not None:
                    popt_v[0] = CONVERT_massn_TO_n_coldens(self.get_center_density(column_density=True),10,popt_v[0],self.data["radius_pc"], is_density=False)
                if fit_set[1] is not None:
                    popt_h[0] = CONVERT_massn_TO_n_coldens(self.get_center_density(column_density=True),10,popt_h[0],self.data["radius_pc"], is_density=False)

            def mass_integrand(r, n0, sigma, r0):
                return 4 * np.pi * r**2 * n0 * np.exp(-0.5 * ((r-r0) / sigma)**2)
            
            if fit_set[0] is not None:
                M_v, _ = quad(mass_integrand, 0, r_px*pix_to_pc, args=tuple(popt_v))
            if fit_set[1] is not None:
                M_h, _ = quad(mass_integrand, 0, r_px*pix_to_pc, args=tuple(popt_h))

            if fit_set[0] is not None and fit_set[1] is not None:
                mass = M_h if err_v > err_h else M_v
            elif fit_set[0] is not None:
                mass = M_v
            else:
                mass = M_h

            mass = mu*m_H*mass*(pc_to_cm)**3
        elif method == "constant":
            r_cm = self.data["radius_pc"] * pc_to_cm
            volume = (4/3) * np.pi * (r_cm**3)
            mw_density = float(self.get_center_density())
            n = CONVERT_massn_TO_n_coldens(float(self.get_center_density(column_density=True)),10,mw_density,float(self.data["radius_pc"]), is_density=False)
            
            if density_error is not None:
                try:
                    bin_centers, q1, q2, means = density_error
                except:
                    LOGGER.error("Density error is not in the good format in DenseCore -> Can't sample a random mass given the error.")
                    return (mu*m_H*n*volume)/Msun
                q1_interp = np.interp(np.log10(mw_density), bin_centers, q1)
                q2_interp = np.interp(np.log10(mw_density), bin_centers, q2)
                gauss_sigma = (q2_interp-q1_interp)/(2*1.64485)
                n = 10**np.random.normal(loc=np.log10(n),scale=gauss_sigma)
            
            mass = mu * m_H *n* volume

        return mass/Msun
    
    def fit_error(self, popt:List, perr:List):
        if popt is None or popt[0] is None:
            return 1e10
        n0, sigma, r0 = popt
        dn0, dsigma, dr0 = perr
        if n0 == 0 or sigma == 0:
            return np.inf
        return sigma+np.abs(r0)
    def fit(self, force_compute:bool=False):
        assert self.obs.prediction is not None, LOGGER.error(f"No predicted density on the observation: f{self.obs.name}")
        if not(self.fit_settings is None) and not(force_compute):
            return self.fit_settings
        env_size = 5
        densities = self.obs.prediction
        r_px = self.obs.pc_to_pixels(self.data["radius_pc"])
        x_center, y_center = skycoord_to_pixel(self.coord, self.wcs)
        region_half_px = self.obs.pc_to_pixels(env_size)
        x_min = max(0,int(x_center - region_half_px))
        x_max = min(densities.shape[1],int(x_center + region_half_px))
        y_min = max(0,int(y_center - region_half_px))
        y_max = min(densities.shape[0],int(y_center + region_half_px))
        region = densities[y_min:y_max, x_min:x_max]
        x_c_rel = (x_center - x_min)
        y_c_rel = (y_center - y_min)
        y_start = int(y_c_rel - r_px)
        y_end   = int(y_c_rel + r_px)
        x_start = int(x_c_rel - r_px)
        x_end   = int(x_c_rel + r_px)
        horizontal_cut = region[int(y_c_rel), x_start:x_end]
        vertical_cut = region[y_start:y_end, int(x_c_rel)]
        pix_to_pc = env_size / region_half_px
        y_axis_pc = (np.arange(y_start, y_end) - y_c_rel) * pix_to_pc
        x_axis_pc = (np.arange(x_start, x_end) - x_c_rel) * pix_to_pc

        def clean_data(x, y):
            mask = np.isfinite(x) & np.isfinite(y) & (y > 0)
            return x[mask], y[mask]
        
        if len(y_axis_pc) < 5 or len(x_axis_pc) < 5 or len(vertical_cut) < 5 or len(horizontal_cut) < 5:
            LOGGER.warn(f"Core {self.data['name']} has insufficient data for fit.")
            self.fit_settings = [None, None, None, None]
            return self.fit_settings

        y_axis_pc, vertical_cut = clean_data(y_axis_pc, vertical_cut)
        x_axis_pc, horizontal_cut = clean_data(x_axis_pc, horizontal_cut)
        
        amp_v = np.nanmax(vertical_cut) if len(vertical_cut) > 0 else 1
        amp_h = np.nanmax(horizontal_cut) if len(horizontal_cut) > 0 else 1

        popt_v = None
        popt_h = None
        perr_h = None
        perr_v = None
        try:
            popt_v, pcov_v = curve_fit(density_gaussian, y_axis_pc, vertical_cut, p0=[amp_v, 0.1, 0.], bounds=([0,0,-self.data["radius_pc"]*.8],[amp_v,1,self.data["radius_pc"]*.8]))
            perr_v = np.sqrt(np.diag(pcov_v))
        except Exception as e:
            LOGGER.warn(f"Core {self.data['name']}: Fit (y) failed → {e}")
        try:
            popt_h, pcov_h = curve_fit(density_gaussian, x_axis_pc, horizontal_cut, p0=[amp_h, 0.1, 0.], bounds=([0,0,-self.data["radius_pc"]*.8],[amp_h,1,self.data["radius_pc"]*.8]))
            perr_h = np.sqrt(np.diag(pcov_h))
        except Exception as e:
            LOGGER.warn(f"Core {self.data['name']}: Fit (x) failed → {e}")
        self.fit_settings = [popt_v, popt_h, perr_v, perr_h]

        return self.fit_settings

    def plot(self, env_size:float=1., cmap:str="rainbow", cdens:bool=False, contour:bool=True, save_path:Union[None,str]=None, nearby_cores:bool=True, show_fit:bool=True):
        """Plot the dense core environment with horizontal and vertical density slices.
        Args:
            env_size(float, default=1): size of the environment(of the image) in parsecs.
            cmap: color of the map
            cdens(bool, default=False): Instead of using predicted volume density, plot column density.
            contour(bool, default=True): Add contours on the environment map.
            save_path: If not None, save the figure in the given folder path.
            nearby_cores(bool, default=True): Plot the nearby cores in the environment map as small white triangles.
            show_fit(bool, default=True): Try to fit the vertical and horizontal slices of the dense core.
        """
        if not(cdens):
            assert self.obs.prediction is not None, LOGGER.error(f"No predicted density on the observation: f{self.obs.name}")
            densities = self.obs.prediction
        else:
            densities = self.obs.data

        x_center, y_center = skycoord_to_pixel(self.coord, self.wcs)
        region_half_px = self.obs.pc_to_pixels(env_size)
        x_min = max(0,int(x_center - region_half_px))
        x_max = min(densities.shape[1],int(x_center + region_half_px))
        y_min = max(0,int(y_center - region_half_px))
        y_max = min(densities.shape[0],int(y_center + region_half_px))
        region = densities[y_min:y_max, x_min:x_max]

        fig, axes = plt.subplot_mosaic(
            [['A','A','B'],['A','A','C']],
            constrained_layout=True,
            figsize=(10, 6),
        )
        ax_reg = fig.add_subplot(axes['A'], projection=self.wcs)
        ax_cut_v = axes['B']
        ax_cut_h = axes['C']

        x_ticks_pix = ax_reg.get_xticks()
        y_ticks_pix = ax_reg.get_yticks()
        x_ticks_full = x_ticks_pix + x_min
        y_ticks_full = y_ticks_pix + y_min
        sky_x = pixel_to_skycoord(x_ticks_full, np.full_like(x_ticks_full, y_center), self.wcs)
        sky_y = pixel_to_skycoord(np.full_like(y_ticks_full, x_center), y_ticks_full, self.wcs)
        # RA in hours (hh:mm:ss), Dec in degrees (dd:mm:ss)
        ra_labels = [ra.ra.to_string(unit=u.hour, sep=':', precision=1, pad=True) for ra in sky_x]
        dec_labels = [dec.dec.to_string(unit=u.deg, sep=':', precision=1, alwayssign=True, pad=True) for dec in sky_y]
        ax_reg.set_xticklabels(ra_labels, rotation=45)
        ax_reg.set_yticklabels(dec_labels)
        ax_reg.set_xlabel("RA [h:m:s]")
        ax_reg.set_ylabel("Dec [°:′:″]")
        ax_reg.invert_xaxis()

        ax_reg.set_title(f"{self.data['name']} | ±{env_size} pc | r={self.data['radius_pc']} pc | $N_H=${self.get_center_density(column_density=True):.0e} | M={self.compute_mass():.2}")
        vmin = np.nanpercentile(region, 0)
        vmax = np.nanpercentile(region, 100)
        if vmin <= 0:
            vmin = np.nanmin(region[region > 0])
        levels = np.logspace(np.log10(vmin), np.log10(vmax), 20)
        img_plt = ax_reg.imshow(region, cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax), origin="lower")
        plt.colorbar(img_plt, ax=ax_reg, label=r"$N_H$ [cm$^{-2}$]" if cdens else r"$<n_H>_m$ [cm$^{-3}$]")
        if contour:
            contour_plt = ax_reg.contour(region, levels=levels, colors="black", origin="lower")
            
        x_c_rel = (x_center - x_min)
        y_c_rel = (y_center - y_min)
        ax_reg.scatter(x_c_rel, y_c_rel, color="black", lw=2., marker="+", zorder=10, label="Core")

        r_pc = self.data["radius_pc"]
        r_px = self.obs.pc_to_pixels(r_pc)
        circle = plt.Circle((x_c_rel, y_c_rel), r_px, color='black', lw=1., fill=False, ls="-", zorder=10)
        inner_circle = plt.Circle((x_c_rel, y_c_rel), r_px*0.9, lw=2., color='white', fill=False, zorder=10)
        ax_reg.add_patch(inner_circle)
        inner_circle_2 = plt.Circle((x_c_rel, y_c_rel), r_px*0.8, lw=1., color='black', fill=False, zorder=10)
        ax_reg.add_patch(inner_circle_2)
        ax_reg.add_patch(circle)

        y_start = int(y_c_rel - r_px * 3)
        y_end   = int(y_c_rel + r_px * 3)
        vertical_cut = region[y_start:y_end, int(x_c_rel)]
        x_start = int(x_c_rel - r_px * 3)
        x_end   = int(x_c_rel + r_px * 3)
        horizontal_cut = region[int(y_c_rel), x_start:x_end]
        pix_to_pc = env_size / region_half_px

        ax_cut_v.axvline(+self.data["radius_pc"], color="red", linestyle="--")
        ax_cut_v.axvline(-self.data["radius_pc"], color="red", linestyle="--")
        ax_cut_v.axvline(0, color="black", linestyle="--")

        ax_cut_h.axvline(+self.data["radius_pc"], color="red", linestyle="--")
        ax_cut_h.axvline(-self.data["radius_pc"], color="red", linestyle="--")
        ax_cut_h.axvline(0, color="black", linestyle="--")

        y_axis_pc = (np.arange(y_start, y_end) - y_c_rel) * pix_to_pc
        x_axis_pc = (np.arange(x_start, x_end) - x_c_rel) * pix_to_pc

        ax_cut_v.plot(y_axis_pc, vertical_cut, color="black", label="density profile")
        ax_cut_v.set_xlabel("y [pc]")
        ax_cut_h.plot(x_axis_pc, horizontal_cut, color="black", label="density profile")
        ax_cut_h.set_xlabel("x [pc]")
        ax_cut_v.set_yscale("log")
        ax_cut_h.set_yscale("log")

        if(show_fit):
            fit_set = self.fit()
            popt_v = fit_set[0]
            popt_h = fit_set[1]
            if popt_v is not None:
                perr_v = self.fit_error(popt_v, fit_set[2])
                plot_function(lambda x: density_gaussian(x, *popt_v),ax=ax_cut_v,color="red",lims=[np.nanmin(y_axis_pc),np.nanmax(y_axis_pc),0,1], label=f"fit s={perr_v:.2e}")
            if popt_h is not None:
                perr_h = self.fit_error(popt_h, fit_set[3])
                plot_function(lambda x: density_gaussian(x, *popt_h),ax=ax_cut_h,color="red",lims=[np.nanmin(x_axis_pc),np.nanmax(x_axis_pc),0,1], label=f"fit s={perr_h:.2e}")

        ax_cut_h.set_ylim([np.nanmin(horizontal_cut)*.9,np.nanmax(horizontal_cut)*1.1])
        ax_cut_v.set_ylim([np.nanmin(vertical_cut)*.9,np.nanmax(vertical_cut)*1.1])

        if(nearby_cores):
            cores = [c.data for c in self.obs.get_cores()]
            ra_c = [c['ra'] for c in cores]
            dec_c = [c['dec'] for c in cores]
            all_coords = SkyCoord(ra=ra_c, dec=dec_c, unit="deg")
            sep = self.coord.separation(all_coords)
            D_pc = self.obs.distance
            theta_max = (1 / D_pc)*180/np.pi*u.deg
            near_mask = (sep < theta_max) & (sep > 0.)
            near_coords = all_coords[near_mask]
            near_x, near_y = skycoord_to_pixel(near_coords, self.wcs)
            ax_reg.scatter(near_x - x_min, near_y - y_min,marker='^', facecolor='white', edgecolor='black', s=25, zorder=10, label="Nearby cores")

        scale_bar_px = self.obs.pc_to_pixels(env_size/5)

        fontprops = fm.FontProperties(size=9)

        scalebar = AnchoredSizeBar(
            ax_reg.transData,
            scale_bar_px,
            f"{env_size/5:.1f} pc",
            loc="lower right",
            pad=0.4,
            color="black",
            frameon=True,
            size_vertical=.0,
            fontproperties=fontprops,
        )

        ax_reg.add_artist(scalebar)

        ax_reg.legend(loc="best")
        ax_cut_v.legend(loc="best")
        ax_cut_h.legend(loc="best")

        if save_path is not None:
            fig.savefig(save_path+f"core_{self.data['name']}_{('cdens' if cdens else 'vdens')}.jpg", dpi=300)

        return fig, axes