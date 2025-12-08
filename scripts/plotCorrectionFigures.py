from POLARIScore.utils.physics_utils import CONVERT_massn_TO_n_coldens, PC_TO_CM
from POLARIScore.utils.utils import plot_function
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    col_dens = 1e22
    core_radius = 0.05

    fig, ax = plt.subplots(figsize=(5,5))
    fig2, ax2 = plt.subplots(figsize=(5,5))
    fig3, ax3 = plt.subplots(figsize=(5,5))

    ax.set_xlabel(r"$L_d (pc)$")
    ax2.set_xlabel(r"$n_d (\text{particles}/cm^{-3})$")
    ax3.set_xlabel(r"$N_\text{H} (\text{particles}/cm^{-2})$")
    ax.set_ylabel(r"$\frac{n_c}{<n_H>_m}$")
    ax2.set_ylabel(ax.get_ylabel())
    ax3.set_ylabel(ax.get_label())
    ax2.set_xscale("log")
    ax3.set_xscale("log")

    ax.grid()
    ax2.grid()
    ax3.grid()

    def _compute_tX(n,t=0.9):
        Lc = 2*core_radius*PC_TO_CM
        Y  = (t*(np.sqrt(col_dens/(n*Lc))-1)+1)*n/col_dens
        a = Y**2 - n/(Lc*col_dens)
        b = -2*Y+2*Y**2*Lc+1/(Lc)-n/(col_dens)
        c = Y**2*Lc**2-2*Y*Lc+1

        disc = b**2-4*a*c
        return (-b+np.sqrt(disc))/(2*a)/PC_TO_CM


    lss = ["-","--","-."]
    for i,n in enumerate(np.linspace(3e3,1.e4,3)):
        
        ld_min, ld_max = 0.01, 10
        function_ld = lambda X: CONVERT_massn_TO_n_coldens(col_dens, X, n*np.ones((100)), core_radius, filter=None)/n
        nd_min, nd_max = 30, 1e4
        function_nd = lambda X: CONVERT_massn_TO_n_coldens(col_dens, X, n*np.ones((100)), core_radius, filter=None, is_density=True)/n
        Nd_min, Nd_max = 1e21, col_dens
        function_Nd = lambda X: CONVERT_massn_TO_n_coldens(col_dens, X, n*np.ones((100)), core_radius, filter=None, is_column_density=True)/n

        #print(n,np.sqrt((n*col_dens)/(core_radius*2*PC_TO_CM))/n)

        #print((col_dens-2*core_radius*PC_TO_CM*function_ld(5.)[0]*n)/(5.*PC_TO_CM))
        print(_compute_tX(n=n, t=0.9999999999999))

        plot_function(function_ld, ax=ax, lims=[ld_min,ld_max,None,None], logspace=False, label=rf"$<n_H>_m=${n:.2e}", ls=lss[i], color="black")
        plot_function(function_nd, ax=ax2, lims=[nd_min,nd_max,None,None], logspace=False, label=rf"$<n_H>_m=${n:.2e}", ls=lss[i], color="black")
        plot_function(function_Nd, ax=ax3, lims=[Nd_min,Nd_max,None,None], logspace=True, label=rf"$<n_H>_m=${n:.2e}", ls=lss[i], color="black")


    ax.legend()
    ax2.legend()

    plt.show()