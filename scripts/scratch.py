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
from POLARIScore.utils.sim_utils import init_idefix

sim = Simulation_DC("idefix_grav", global_size=5)
init_idefix(simulation=sim)

from POLARIScore.utils.physics_utils import dcmf_func
from POLARIScore.utils.utils import plot_function
_dcmf_function = lambda M,amp,mu,sigma,alpha,cutoff: dcmf_func(M,amp,mu,sigma,alpha,cutoff)
pdf = compute_pdf(sim.data['RHO']/np.mean(sim.data['RHO']))
bin_centers = pdf[1][:100]
values = pdf[0]

popt, _ = curve_fit(_dcmf_function, (10**bin_centers), values,
                    p0=[np.max(values), np.mean(10**bin_centers), np.std(10**bin_centers), 200.3, 1])
func = lambda X: _dcmf_function(X, popt[0], popt[1], popt[2], popt[3], popt[4])

fig = plt.figure()
ax = fig.subplots()

M = 5
b = np.sqrt((np.exp(popt[2]**2)-1 )/M**2)

ax.plot(10**pdf[1][:100], pdf[0], marker="+", color="black", label="b="+str(b))
plot_function(func, ax=ax, scatter=False, logspace=True, lims= (np.min(10**bin_centers), np.max(10**bin_centers)), color="red", linestyle="--")

ax.set_xscale("log")
fig.legend()

sim.plot_slice(slice=100)

plt.show()