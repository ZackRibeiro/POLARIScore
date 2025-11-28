from POLARIScore.utils.physics_utils import CONVERT_massn_TO_n_coldens, PC_TO_CM
from POLARIScore.utils.utils import plot_function
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    col_dens = 1e22
    core_radius = 0.05

    fig, ax = plt.subplots(figsize=(5,5))
    fig2, ax2 = plt.subplots(figsize=(5,5))

    ax.set_xlabel(r"$L_d (pc)$")
    ax2.set_xlabel(r"$n_d (\text{particles}/cm^{-3})$")
    ax.set_ylabel(r"$\frac{<n_H>_m}{n_c}$")
    ax2.set_ylabel(ax.get_ylabel())
    ax2.set_xscale("log")

    ax.grid()
    ax2.grid()


    lss = ["-","--","-."]
    for i,n in enumerate(np.linspace(1e3,2.e3,3)):
        
        ld_min, ld_max = 0.01, 3
        function_ld = lambda X: CONVERT_massn_TO_n_coldens(col_dens, X, n*np.ones((100)), core_radius, filter=None)/n
        nd_min, nd_max = 30, 1e3
        function_nd = lambda X: CONVERT_massn_TO_n_coldens(col_dens, X, n*np.ones((100)), core_radius, filter=None, is_density=True)/n

        #print(n,n/np.sqrt((n*col_dens)/(core_radius*2*PC_TO_CM)))

        #print nd
                
        plot_function(function_ld, ax=ax, lims=[ld_min,ld_max,None,None], logspace=False, label=rf"$<n_H>_m=${n:.2e}", ls=lss[i], color="black")
        plot_function(function_nd, ax=ax2, lims=[nd_min,nd_max,None,None], logspace=False, label=rf"$<n_H>_m=${n:.2e}", ls=lss[i], color="black")

    ax.legend()
    ax2.legend()

    plt.show()