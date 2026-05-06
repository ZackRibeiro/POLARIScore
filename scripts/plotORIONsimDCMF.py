import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import lognorm
from scipy.optimize import curve_fit
from POLARIScore.utils.utils import plot_function
from POLARIScore.utils.physics_utils import CONVERT_massn_TO_n, dcmf_func


CORES_PATH = "/home/ribeiroz/forks/Polaris/POLARIScore/data/sims/orionMHD_lowB_multi_5/catalog_search_results.json"
bins = 15
ax = None

def plot_sim_dcmf(ax=None,factor=1.,fit=False, logM=True, path=None):
    if path is None:
         path = CORES_PATH
    with open(path) as file:
        cores = json.load(file)
    c_masses = np.array(list(cores["mass"].values()))
    #c_alphvir = np.array(list(cores["alpha_vir"].values()))
    #c_masses = c_masses[c_alphvir <= 0.5]

    ax_was_none = ax is None

    def _get_dcmf(masses:np.ndarray):
            m = np.log10(masses) if logM else masses
            hist, bin_edges = np.histogram(m, bins=bins)
            bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
            dcmf = hist / (bin_edges[1:] - bin_edges[:-1])
            return dcmf, bin_centers

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    derived_dcmf, derived_bin_centers = _get_dcmf(c_masses)

    if fit:
        _dcmf_function = lambda M,amp,mu,sigma,alpha,cutoff: dcmf_func(M,amp,mu,sigma,alpha,cutoff, logM=logM)
        popt, _ = curve_fit(_dcmf_function, (10**derived_bin_centers), derived_dcmf,
                            p0=[np.max(derived_dcmf), 0.22, np.std(np.log(c_masses)), 2.3, 1])
        func = lambda X: _dcmf_function(X, popt[0], popt[1], popt[2], popt[3], popt[4])*factor
        plot_function(func, ax=ax, scatter=False, logspace=True, lims= (0.01, 100), color="green", linestyle="--")

    ax.plot(10**derived_bin_centers, derived_dcmf*factor, drawstyle="steps-mid", color="green", label="Sim (Ntormousi & Hennebelle, 2019)")
    ax.scatter(10**derived_bin_centers, derived_dcmf*factor, color="green")

    if ax_was_none:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"Mass [$M_\odot$]")
        ax.set_ylabel(r"$dN/d\log M$")
        ax.set_xlim([0.01, 100])
    #ax.set_ylim([1e1,None])

    return fig, ax

if __name__ == "__main__":

    #plot_sim_dcmf()
    n_d = 100 #cm^-3
    L_d = 5 #pc
    r_c = 0.1#pc
    fct = lambda n,r : CONVERT_massn_TO_n(n_d,L_d,n,r)/n

    plot_function(fct, lims=[1e3,1e5,0.03,0.3], logspace=True)

    plt.show()