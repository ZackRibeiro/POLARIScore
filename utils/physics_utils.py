import numpy as np
from POLARIScore.config import LOGGER
from typing import Union, Dict, List, Tuple

LIGHT_SPEED = 299792458
"""Velocity of the light in m/s"""
PLANCK_CONSTANT = 6.62607e-34
"""Planck constant in J s"""
BOLTZMANN_CONSTANT = 1.380649e-23
"""Boltzmann constant in J/K"""
PC_TO_CM = 3.086e+18
"""How many centimeters in a parsec"""

BLACKBODY_EMISSION = lambda nu,T: (2*PLANCK_CONSTANT*np.power(nu,3)/(LIGHT_SPEED**2))*(1/(np.exp(PLANCK_CONSTANT*nu/(BOLTZMANN_CONSTANT*T))-1))
"""Emmision of a blackbody in function of the frequency and temperature"""
CMB_TEMPERATURE = 2.725

CO_ABUNDANCE = 1e-4
CO_A = [7.203e-8,6.9e-7,2.5e-6]
"""CO line spontaneous emission coefficient in s^-1"""
CO_FREQUENCY= [115.271e9,230.538e9,345.796e9]
"""CO line frequency in Hz"""

CO13_ABUNDANCE = CO_ABUNDANCE / 70
CO13_A = [6.33e-8, 6.0e-7, 2.1e-6]
CO13_FREQUENCY = [110.201354e9, 220.398684e9, 330.587965e9]
CO13_ROT_CST = 55.101e9

CO_ROT_CST = 57.64e9
ROT_ENERGY = lambda l,rot_cst: PLANCK_CONSTANT*rot_cst*l*(l+1)/BOLTZMANN_CONSTANT

GAUSSIAN = lambda x,sigma: (1/(np.sqrt(2*np.pi)*sigma))*np.exp(-np.power(x,2)/(2*sigma**2))

CONVERT_INTENSITY_TO_KELVIN = lambda I,nu: I*LIGHT_SPEED**2 / (2.*BOLTZMANN_CONSTANT*nu**2)

CONVERT_NH_TO_EXTINCTION = lambda c: c/(2*0.94e21) #(Bohlin et al. 1978)
CONVERT_EXTINCTION_TO_NH = lambda a: a*2*0.94e21

CONVERT_massn_TO_n = lambda n_d,L_d,n,r_c: (n+np.sqrt(n**2-(2*L_d/r_c)*(n_d**2-n*n_d)))/2
def CONVERT_massn_TO_n_coldens(N:Union[np.ndarray[float],float], L_d:Union[np.ndarray[float],float], n:Union[np.ndarray[float],float], r_c:Union[np.ndarray[float],float]
                               , filter:Union[None,float]=22.1, is_density=False, is_column_density=False):
    """
    Args:
        N: column density (particles/cm^-2)
        L_d: size of the diffuse medium along l.o.s (pc) if 'is_density' is False else is the diffuse density (particles/cm^-3). If 'is_column_density' is set to True, this is indeed the column density of background medium.
        n: mass weighted density along l.o.s (particles/cm^-3)
        r_c: radius of the dense core (pc)
        filter: if not None, set the threshold of log10 column density where the conversion will be made.
    """
    #bad code
    if type(N) is np.ndarray or type(N) is list:
        N = np.array(N, dtype=np.float64)
        r_c = np.array(r_c, dtype=np.float64)
        L_d = np.array(L_d, dtype=np.float64)
        n = np.array(n, dtype=np.float64)

    if filter is not None:
        log10_N = np.log10(N)
        factor = 1.-(1/(1+np.exp(-5.*(log10_N-filter))))
        #print(factor[np.argsort(log10_N)])
    else:
        factor = np.ones_like(N)

    r_c = PC_TO_CM*r_c
    if is_column_density:
        N_d = L_d
    else:
        L_d = L_d if is_density else PC_TO_CM*L_d 
    
    L_c = 2*r_c
    if is_density:
        n_d = L_d
        with np.errstate(invalid='ignore'):
            n_c = n_d/2 * (1+np.sqrt(1-4*N/(L_c*n_d)*(1-n/n_d)))
    elif is_column_density:
        alpha = N/N_d
        n_c=n*(alpha)/(alpha-1)
    else:
        with np.errstate(invalid='ignore'):
            n_c = N/(L_c+L_d)*(1+np.sqrt(1-(L_d/L_c+1)*(1-(n*L_d)/(N))))

    mask = np.isnan(n_c)
    if type(n) is list or type(n) is np.ndarray:
        n_c[mask] = n[mask]
        if np.sum(mask) > 0:
            LOGGER.warn(f"{np.sum(mask)}/{len(n_c)} cores densities are replaced with mass average density, bcs discriminant is < 0")
    else:
        if mask.any():
            return n

    n_c = np.maximum(n_c, n)
    n_c = n_c*factor+n*(1-factor)
    return n_c

from scipy.stats import lognorm
def plot_lognorm(ax, mean, std, amp=1., x_min=1e-2, x_max=1e3, n_points=100, 
                 color='red', label=None, lw=1, ls="--"):
    x = np.logspace(np.log10(x_min), np.log10(x_max), n_points)
    pdf = lognorm.pdf(x, s=std, scale=mean)
    pdf = pdf * x
    ax.plot(x, amp*pdf, color=color, lw=lw, label=label, linestyle=ls)
    return ax

def dcmf_func(M, amp, mu, sigma, alpha, cutoff, logM:bool=True, enable_cutoff:bool=True):
        pdf_low = lognorm.pdf(M, s=sigma, scale=np.abs(mu))

        if not(enable_cutoff):
            pdf_low *= amp
            return pdf_low*M if logM else pdf_low
        pdf_high = M**(-alpha)
        join_mass = cutoff
        scale_factor = (pdf_low[np.argmin(np.abs(M - join_mass))] /
                        pdf_high[np.argmin(np.abs(M - join_mass))])
        pdf_high *= scale_factor

        pdf_low *= amp
        pdf_high *= amp

        if type(M) is np.ndarray or type(M) is list:
            return np.concatenate((pdf_low[M <= cutoff],pdf_high[M > cutoff]),axis=0)*M
        else:
          if M > cutoff:
              return pdf_high*M if logM else pdf_high
          else:
              return pdf_low*M if logM else pdf_low

def plot_imf_chabrier(ax, color='black', x_min=1e-2, x_max=1e3, n_points=100,
                      Mc=0.22, sigma_ln=1.31, alpha=2.3, amp=25.0, logM=True, dcmf=1.):

    x = np.logspace(np.log10(x_min), np.log10(x_max), n_points)
    x = x / dcmf

    pdf_low = lognorm.pdf(x, s=sigma_ln, scale=Mc/dcmf)
    pdf_low = pdf_low * x * np.log(10)

    pdf_high = x**(1 - alpha) if logM else x**(-alpha)

    join_mass = 1./dcmf
    scale_factor = (pdf_low[np.argmin(np.abs(x - join_mass))] /
                    pdf_high[np.argmin(np.abs(x - join_mass))])
    pdf_high *= scale_factor

    amp_scaled = amp/np.max(pdf_low)

    pdf_low *= amp_scaled
    pdf_high *= amp_scaled

    if dcmf != 1.:
        label = f'DCMF (Chabrier, 2003; IMF with efficiency of {(dcmf*100.)}%)'
    else:
        label = 'IMF (Chabrier, 2003)'

    ax.plot(x[x <= join_mass], pdf_low[x <= join_mass],
            ls='--', color=color, label=label)
    ax.plot(x[x > join_mass], pdf_high[x > join_mass],
            ls='--', color=color)

    return ax
    
def density_gaussian(r, n0, sigma, r0):
    return n0 * np.exp(-0.5 * ((r-r0) / sigma)**2)

def power_spectrum_2d(map2d, px_size, bins=20):
    mask = np.isnan(map2d)
    map2d[mask] = np.nanmean(map2d)
    
    map2d = map2d - np.mean(map2d)
    fft_map = np.fft.fft2(map2d)
    fft_map = np.fft.fftshift(fft_map)
    power = np.abs(fft_map)**2

    ny, nx = map2d.shape
    kx = np.fft.fftfreq(nx, d=px_size)
    ky = np.fft.fftfreq(ny, d=px_size)
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