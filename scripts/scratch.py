from POLARIScore.utils.vtk_io import readVTKCart
from POLARIScore.utils import compute_pdf
from POLARIScore.config import DATA_FOLDER
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib
from scipy.optimize import curve_fit

from POLARIScore.objects.Simulation_DC import Simulation_DC
from POLARIScore.utils.sim_utils import init_idefix,init_ramses
from POLARIScore.utils.utils import compute_mass_weighted_density

from POLARIScore.objects.SimulationArray import SimulationArray
#sims = SimulationArray(simulations=[] ,name="turb_sim_C")
#sims.plot(Simulation_DC.plot_power_spectrum, mode="mixed", normalize=True, plot_coldens=False, vdens_method=compute_mass_weighted_density,
#          colors = "copper")
#sims.plot(Simulation_DC.plot, mode="slider", color_bar=False, plot_pdf=True)

#sim = Simulation_DC("turb_sim_A")
#sim.plot()
#from POLARIScore.objects.SpectrumMap import SpectrumMap, getSimulationSpectra
#plt.show()

#maps = getSimulationSpectra(simulation=sim)
#maps[0].plot(simulation=sim)

#v_map = sim.compute_velocity_decomposition()
#v_map.plot()


#plt.show()


#sim = Simulation_DC("orionHD_all_512")
sim = Simulation_DC("idefix_sim_A")
sim.plot_power_spectrum(what_to_plot="rms_velocity", bins=30, energy=True )

"""
from POLARIScore.utils.physics_utils import dcmf_func, density_gaussian
from POLARIScore.utils.utils import plot_function
_dcmf_function = lambda M,amp,mu,sigma,alpha,cutoff: dcmf_func(M,amp,mu,sigma,alpha,cutoff, enable_cutoff=False)


pdf = compute_pdf(sim.data['RHO']/np.mean(sim.data['RHO']))
bin_centers = pdf[1][:100]
values = pdf[0]

popt, _ = curve_fit(_dcmf_function, (10**bin_centers), values,
                    p0=[np.max(values), np.mean(bin_centers), np.std(bin_centers), 1., 1])
func = lambda X: _dcmf_function(X, popt[0], popt[1], popt[2], popt[3], popt[4])

fig = plt.figure()
ax = fig.subplots()

M = 5
b = np.sqrt((np.exp(popt[2]**2)-1 )/M**2)

ax.plot(10**pdf[1][:100], pdf[0], marker="+", color="black", label=f"b={b:.3}")
plot_function(func, ax=ax, scatter=False, logspace=True, lims= (np.min(10**bin_centers), np.max(10**bin_centers)), color="red", linestyle="--")

ax.set_xscale("log")
ax.set_yscale("log")
fig.legend()

fig = plt.figure()
ax = fig.subplots()
def _plot_velocity(key, fit:bool=False):
    pdf = compute_pdf(sim.data[key], func=lambda x: x)
    bin_centers = pdf[1][:100]
    values = pdf[0]

    ax.plot(pdf[1][:100], pdf[0], marker="+", label=rf"{key}: $M=${np.sqrt(np.mean(np.power(sim.data[key],2)))*np.sqrt(3)/3.2591e+4:.2}")

    if fit:
        popt, _ = curve_fit(density_gaussian, bin_centers, values,
                            p0=[np.max(values), np.std(bin_centers), np.mean(bin_centers)])
        func = lambda X: density_gaussian(X, *popt)
        plot_function(func, ax=ax, scatter=False, logspace=False, lims= (np.min(bin_centers), np.max(bin_centers)), color="red", linestyle="--")

_plot_velocity('VX1')
_plot_velocity('VX2')
_plot_velocity('VX3')
fig.legend()


sim.plot_slice(slice=100,)
from POLARIScore.utils.utils import compute_mass_weighted_density, compute_volume_weighted_density
"""
plt.show()
