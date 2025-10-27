import numpy as np
from POLARIScore.utils.utils import *
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from typing import List
import scipy

def compute_smoothness(matrix:np.ndarray)->float:
    log_matrix = np.log1p(matrix)
    laplacian = scipy.ndimage.laplace(log_matrix-np.min(log_matrix))
    raw_score = np.var(laplacian)
    return raw_score

def compute_img_score(cdens:np.ndarray,vdens:np.ndarray):
    score = 0
    sm1 = compute_smoothness(cdens)
    sm2 = compute_smoothness(vdens)*0.5
    score = sm1+sm2
    diff_matrix = (cdens-np.min(cdens))/(np.max(cdens)-np.min(cdens))-(vdens-np.min(vdens))/(np.max(vdens)-np.min(vdens))
    sr1 = np.var(diff_matrix.flatten())*5
    score += sr1
    return (score,(sm1,sm2,sr1))

def rebuild_batch(cdens:np.ndarray, vdens:np.ndarray):
    batch = []
    for i in range(len(cdens)):
        batch.append((cdens[i], vdens[i]))
    return batch

def plot_batch(batch, b_name:str="",same_limits:bool=True, number_per_row:int=8, number:int=16):
    batch_nbr = min(len(batch),number)
    fig, axes = plt.subplots(int(2*np.ceil(batch_nbr/number_per_row)),number_per_row)
    if number_per_row==1:
        axes = [[axes[0]],[axes[1]]]
    fig.suptitle(b_name)
    for i in range(batch_nbr):
        data1 = batch[i][0]
        data2 = batch[i][1]
        #score = 0.
        #axes[2*(i//8)][i%8].set_title(str(np.round(score[0],3)))
        min_dat1 = np.min(data1)
        max_dat1 = np.max(data1) 
        d1 = axes[2*(i//number_per_row)][i%number_per_row].imshow(data1, cmap="jet", norm=LogNorm(vmin=np.min(data1), vmax=np.max(data1)))
        d2 = axes[2*(i//number_per_row)+1][i%number_per_row].imshow(data2, cmap="jet", norm=(LogNorm(vmin=np.min(data2), vmax=np.max(data2)) if not(same_limits) else LogNorm(min_dat1, max_dat1)))
    fig.subplots_adjust( left=None, bottom=None,  right=None, top=None, wspace=None, hspace=None)

    return fig, axes

def plot_batch_correlation(batch, ax=None, bins_number:int=256, show_yx:bool=True, lines:List[float]=[0,1,2]):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    column_density = np.array([np.log(b[0])/np.log(10) for b in batch]).flatten()
    volume_density = np.array([np.log(b[1])/np.log(10) for b in batch]).flatten()

    nan_indices = np.isnan(column_density) | np.isnan(volume_density)
    good_indices = ~nan_indices
    column_density= column_density[good_indices]
    volume_density = volume_density[good_indices]

    _, _, _,hist = ax.hist2d(column_density, volume_density, bins=(bins_number,bins_number), norm=LogNorm())
    if show_yx:
        yx = np.linspace(np.min(column_density), np.max(column_density), 10)
        p = ax.plot(yx,yx,linestyle="--",color="red",label=r"$y=x$")
        #plt.legend(p)

    plt.colorbar(hist, ax=ax, label="counts")
    ax = plt.gca()

    plot_lines(column_density, volume_density, ax, lines=lines)

    ax.grid(True)
    ax.set_axisbelow(True)

    fig.tight_layout()

    return fig, ax