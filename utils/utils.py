import numpy as np
from scipy.ndimage import rotate
from POLARIScore.config import LOGGER
import matplotlib.pyplot as plt
import platform
from typing import *
import os

try:
    import psutil
    psutil_available = True
except ImportError:
    psutil_available = False
try:
    import GPUtil
    gputil_available = True
except ImportError:
    LOGGER.warn("No GPU detected on the machine, can leads to very long waiting time when using neural networks.")
    gputil_available = False
from POLARIScore.config import LOGGER
import inspect
import ast, re
import matplotlib.patheffects as pe
import threading

def ask_to_user(sentence: str, seconds: int = 10, default=None):
    result = {"value": default}

    def get_input():
        result["value"] = input(sentence)

    thread = threading.Thread(target=get_input, daemon=True)
    thread.start()

    thread.join(timeout=seconds)

    return result["value"]

def convert_pc_to_index(pc:float,nres:int,size:float,start:float=0.,clip:bool=True,flip:bool=False)->int:
    """
    Args:
        pc(float): value in parsec unit
        nres(int): resolutions of the sim datacube
        size(float): physical size of the datacube in pc
        start(float): if there is an offset in the datacube.
    Returns:
        float: index
    """
    idx = (pc-start)/(size)
    if flip:
        idx = 1. - idx
    if clip and (idx > 1 or idx < 0):
        return -1
    idx = np.floor(idx*nres)
    if isinstance(idx, np.ndarray):
        return idx.astype(int)
    return int(idx)

def compute_column_density(data_cube:np.ndarray,cell_size:float, axis:int=0)->np.ndarray:
    return np.sum(data_cube, axis=axis) * cell_size
def compute_volume_weighted_density(data_cube:np.ndarray, axis:int=0)->np.ndarray:
    return np.sum(data_cube, axis=axis) / data_cube.shape[0]
def compute_mass_weighted_density(data_cube:np.ndarray, axis:int=0)->np.ndarray:
    return np.sum(np.power(data_cube,2), axis=axis) / np.sum(data_cube, axis= axis)
def compute_squared_weighted_density(data_cube:np.ndarray, axis:int=0)->np.ndarray:
    return np.sum(np.power(data_cube,2), axis=axis) / data_cube.shape[0]
def compute_max_density(data_cube:np.ndarray, axis:int=0)->np.ndarray:
    return np.max(data_cube, axis=axis)
def compute_derivative(data_slice:np.ndarray, order:int=1, axis:int=0):
    d = data_slice
    for o in range(order):
        d = np.gradient(d)[axis]
    return d

from scipy.ndimage import gaussian_filter

def convolve_map(map, resolution, beam_size=18.2, distance=400):
    """
    Convolve map with a Gaussian beam to match observational resolution.

    Args:
        map (2D array): The map to convolve.
        resolution (float): Pixel size in parsec (pc/pixel) of the map.
        beam_size (float): Observational beam size in arcseconds (default 18.2").
        distance (float): Distance to the cloud in parsec (default 400 pc).
        replace (bool): If True, replace self.data with the smoothed map.

    Returns:
        2D array: The convolved map.
    """
    beam_rad = np.deg2rad(beam_size / 3600.0)
    beam_pc = distance * beam_rad

    sigma_pixels = beam_pc / resolution / (2.0 * np.sqrt(2.0 * np.log(2.0)))  
    print(sigma_pixels)
    # Gaussian sigma = FWHM / (2*sqrt(2*ln2))

    smoothed = gaussian_filter(map, sigma=sigma_pixels)
    return smoothed

def rotate_cube(data_cube:np.ndarray, angle:float, axis:int)->np.ndarray:
    """Rotates the cube around a given axis (0=X, 1=Y, 2=Z) by a given angle in degrees."""
    return rotate(data_cube, angle, axes=axis, reshape=False, mode="nearest")
def compute_pdf(data_slice:np.ndarray, bins:int=100, func:Callable=lambda x: np.log10(x), center:bool=False, density:bool=True)->Tuple[List[float],List[float]]:
    """
    Compute the probability density function of a matrix

    Args:
        data_slice(np array): the matrix
        bins(int): How many bins
        func: function applied to the flatten data, like convert to log10 by default
        center(bool, default:False): Center the bins and pdf to 0
    
    Returns:
        probabilities, edges
    """
    mask = np.isfinite(data_slice.flatten())
    if np.any(~np.isfinite(func(data_slice.flatten()[mask]))):
        mask = mask & (data_slice.flatten() > 0)
    hist = np.histogram(func(data_slice.flatten()[mask]),bins=bins, density=density)
    probabilities = hist[0]
    edges = hist[1]
    if(center):
        center_i = np.argmax(probabilities)
        edges = edges - (edges[center_i+1]+edges[center_i])/2
    return [probabilities, edges]

def format_time(seconds:float)->str:
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{int(hours):02}h:{int(minutes):02}m:{int(seconds):02}s"

import time
_time_0 = 0
last_iteration = 0
def printProgressBar(iteration,total,prefix='',suffix='',decimals=1,length=50,fill='█',printEnd="\r",show_tr=True,show_speed=True,disable_done=False):
    global _time_0
    global last_iteration
    delta_time = 0.
    if last_iteration > iteration:
        last_iteration = 0
    if iteration==0 or iteration==1 or last_iteration == 0:
        _time_0 = time.time()
    else:  
        delta_time = (time.time()-_time_0)/(iteration) * (total-iteration)

    if delta_time != 0:
        speed = (total-iteration)/delta_time
    else:
        speed = 0

    if total == 0:
        total = 1

    percent = iteration / float(total)
    percent_str = f"{100 * percent:.{decimals}f}"

    filledLength = int(length * iteration // total)

    if percent < 0.5:
        color = "\033[91m"
    elif percent < 0.8:
        color = "\033[93m"
    else:
        color = "\033[92m"

    reset = "\033[0m"

    bar = (
        color + fill * filledLength + reset +
        '█' * (length - filledLength)
    )

    string = f'\r{prefix} {bar} {color+percent_str}%{reset} {suffix}'
    if show_tr:
        string += f" tr~{format_time(delta_time)}"
    if show_speed:
        string += f" s~{speed:0.1f}ite/s"

    print(string, end=printEnd)
        
    last_iteration = iteration
    if iteration >= total-1 and not(disable_done):
        last_iteration = 0
        print(f'\r{prefix} - Done in {format_time(time.time()-_time_0)}'+' '*(length+10))

def divide_matrix_to_sub(matrix:np.ndarray,final_dim:int=128)->List[np.ndarray]:
    final_dim = int(2**round(np.log2(final_dim)))
    img_number = int(matrix.shape[0]/final_dim)
    imgs = []
    for i in range(img_number):
        for j in range(img_number):
            imgs.append(matrix[i*final_dim:(i+1)*final_dim,j*final_dim:(j+1)*final_dim])
    return imgs

def group_matrix(mats:List[np.ndarray]):
    grid_size = int(np.sqrt(len(mats)))
    final_dim = len(mats[0])
    new_mat_shape = final_dim * grid_size
    result = np.zeros((new_mat_shape, new_mat_shape))
    for idx, mat in enumerate(mats):
        row_idx = idx // grid_size
        col_idx = idx % grid_size
        result[row_idx * final_dim: (row_idx + 1) * final_dim,
               col_idx * final_dim: (col_idx + 1) * final_dim] = np.array(mat)
    return result

def moving_average(l, n=5, return_std=False):
    l = np.asarray(l, dtype=float)
    cs = np.cumsum(l)
    cs[n:] = cs[n:] - cs[:-n]
    moving_avg = cs[n-1:] / n
    
    if return_std:
        stds = np.array([np.std(l[i:i+n], ddof=0) for i in range(len(l) - n + 1)])
        return moving_avg, stds
    
    return moving_avg

def moving_minimum(l, n=5, exclude_zeros=False):
    result = []
    for i in range(len(l) - n + 1):
        window = l[i:i+n]
        if exclude_zeros:
            flag = [False for _ in range(n)]
            for j,w in enumerate(window):
                if w < 1e-5:
                    flag[j] = True
            if True in flag and False in flag:
                result.extend(window)
                continue
        result.append(min(window))
    return np.array(result, dtype=object)

def bin_mean(x, y, dx=None, nbins=None, logspace=True, method:Literal['mean','median']='mean', min_per_bin=1, return_deviation:bool=False, return_bins:bool=False):
    x = np.asarray(x)
    y = np.asarray(y)
    mask = (~np.isnan(x)) & (~np.isnan(y)) & (x > 0) #& (y > 0)
    x = x[mask]
    y = y[mask]

    if logspace:
        lx = np.log10(x)
        if dx is not None:
            bins = np.arange(np.min(lx), np.max(lx) + dx, dx)
        else:
            if nbins is None:
                nbins = 20
            bins = np.linspace(np.min(lx), np.max(lx), nbins + 1)
        bin_centers = 10**(0.5 * (bins[1:] + bins[:-1]))
        which_bin = np.digitize(lx, bins) - 1
    else:
        if dx is not None:
            bins = np.arange(np.min(x), np.max(x) + dx, dx)
        else:
            if nbins is None:
                nbins = 20
            bins = np.linspace(np.min(x), np.max(x), nbins + 1)
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        which_bin = np.digitize(x, bins) - 1

    x_binned, y_binned = [], []
    x_std, y_std = [], []
    y_bins = []
    for i in range(len(bin_centers)):
        in_bin = which_bin == i
        if np.sum(in_bin) >= min_per_bin:
            if method == 'mean':
                x_binned.append(np.mean(x[in_bin]))
                y_binned.append(np.mean(y[in_bin]))
            elif method == 'median':
                x_binned.append(np.median(x[in_bin]))
                y_binned.append(np.median(y[in_bin]))
            if return_bins:
                y_bins.append(y[in_bin])
            x_std.append(np.std(x[in_bin]))
            y_std.append(np.std(y[in_bin], dtype=np.float64))
        
    if return_bins:
        return np.array(x_binned), np.array(y_binned), y_bins
    if return_deviation:
        return np.array(x_binned), np.array(y_binned), np.array(x_std), np.array(y_std)

    return np.array(x_binned), np.array(y_binned)


def applyBaseline(t,y,T,Y):

    last_t = 0
    last_y = [y[0]]

    coefs = []
    for i in range(len(y)):
        y1 = y[i]
        t1 = t[i]
        coefs.append((y1 - last_y[-1]) / (t1 - last_t))
        last_t = t1
        last_y.append(y1)
    coefs.append(0.)

    int_time = []
    for j in range(len(t)+1):
        t_left = 0
        if j > 0:
            t_left = t[j-1]
        t_right = T[-1]
        if j < len(t):
            t_right = t[j]
        int_time.append((t_left,t_right))
    t_edges = np.array([0] + list(t))
    indices = np.searchsorted(t_edges[1:], T, side='right')
    for i in range(len(T)):
        j = indices[i]
        tl = t_edges[j]
        Y[i] = Y[i] - (coefs[j] * (T[i] - tl) + last_y[j])

    return Y
def dictsToString(dicts: List[Dict]) -> str:
    """Combine a list of dicts into an aligned table string."""
    if not dicts:
        return ""

    keys = []
    for d in dicts:
        for k in d:
            if k not in keys:
                keys.append(k)

    col_widths = {}
    for k in keys:
        max_len = len(str(k))
        for d in dicts:
            if k in d:
                max_len = max(max_len, len(str(d[k])))
        col_widths[k] = max_len

    lines = []
    header = " ".join(f"{k:<{col_widths[k]}}" for k in keys)
    lines.append(header)

    for d in dicts:
        row = " ".join(
            f"{str(d[k]) if k in d else '':<{col_widths[k]}}"
            for k in keys
        )
        lines.append(row)

    return "\n".join(lines)
def plot_function(function:Callable, ax=None, res:int=100, lims:Tuple[float]=[0,1,0,1], logspace=False, scatter=False, contour=False, **args):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    n_args = len(inspect.signature(function).parameters)

    if n_args == 1:
        xmin, xmax = lims[:2]
        X = np.logspace(np.log10(xmin), np.log10(xmax), res) if logspace else np.linspace(xmin, xmax, res)
        Y = function(X)

        ax.plot(X, Y, **args)
        if scatter:
            ax.scatter(X, Y, color="black")

    elif n_args == 2:
        xmin, xmax, ymin, ymax = lims
        x = np.logspace(np.log10(xmin), np.log10(xmax), res) if logspace else np.linspace(xmin, xmax, res)
        y = np.logspace(np.log10(ymin), np.log10(ymax), res) if logspace else np.linspace(ymin, ymax, res)
        X, Y = np.meshgrid(x, y)
        Z = function(X, Y)

        if contour:
            contourf = ax.contourf(X, Y, Z, levels=50, cmap=args.get("cmap", "viridis"))
            fig.colorbar(contourf, ax=ax)
        else:
            im = ax.imshow(Z, extent=[xmin, xmax, ymin, ymax], origin="lower", cmap=args.get("cmap", "viridis"), aspect='auto')
            fig.colorbar(im, ax=ax)
        
        if logspace:
            ax.set_xscale("log")
            ax.set_yscale("log")
    else:
        LOGGER.error("Function must take 1 or 2 arguments.")
        raise ValueError("Function must take 1 or 2 arguments.")

    return fig, ax


def plot_lines(ax, x:Union[np.ndarray,List,None]=None,y:Union[np.ndarray,List,None]=None, lines:List[float]=[0,1,2], x_max:float=None, x_min:float=None, y_max:float=None, y_min:float=None, logspace=False):
    """
    Plots lines on matplotlib plot. 
    """
    x_max = ax.get_xlim()[1] if x_max is None else x_max
    x_min = ax.get_xlim()[0] if x_min is None else x_min
    y_max = ax.get_ylim()[1] if y_max is None else y_max
    y_min = ax.get_ylim()[0] if y_min is None else y_min
    if not(lines is None):
        axisx_length = (x_max-x_min)
        axisy_length = (y_max-y_min)
        x_corner = axisx_length*0.7+x_min
        y_corner = axisy_length*0.1+y_min
        length = axisx_length*0.2  
        for l in lines:
            ax.plot([x_corner, x_corner + length],
                    [10**y_corner if logspace else y_corner, 10**(y_corner + length*l) if logspace else y_corner + length*l],
                    '--', lw=1, color="black")
            if l != 0:
                ax.text(x_corner + length + length*0.1, 10**(y_corner + l*length) if logspace else y_corner + l*length, fr'$x^{l}$', color='black')
    return ax

def get_system_info():
    system_info = {}
    if not(psutil_available) or not(psutil_available):
        LOGGER.warn("Can't get system config informations because GPUtil or psutil are not installed.")
        return system_info

    # CPU
    system_info['CPU'] = {
        'Processor': platform.processor(),
        'Physical Cores': psutil.cpu_count(logical=False),
        'Total Cores': psutil.cpu_count(logical=True),
        'Max Frequency (MHz)': psutil.cpu_freq().max,
        'Current Frequency (MHz)': psutil.cpu_freq().current,
    }

    # RAM
    svmem = psutil.virtual_memory()
    system_info['RAM'] = {
        'Total (GB)': round(svmem.total / (1024 ** 3), 2),
    }

    # GPU
    gpus = GPUtil.getGPUs()
    gpu_info = []
    for gpu in gpus:
        gpu_info.append({
            'Name': gpu.name,
            'Memory Total (MB)': gpu.memoryTotal,
            'Driver Version': gpu.driver,
        })
    system_info['GPU'] = gpu_info if gpu_info else 'No GPU Found'

    # System
    system_info['System'] = {
        'System': platform.system(),
        'Node Name': platform.node(),
        'Release': platform.release(),
        'Version': platform.version(),
        'Machine': platform.machine(),
    }

    return system_info

import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return {
                "__ndarray__": True,
                "dtype": str(obj.dtype),
                "shape": obj.shape,
                "data": obj.tolist(),
            }
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        else:
            return super().default(obj)

def numpy_decoder(obj):
    if "__ndarray__" in obj:
        return np.array(obj["data"], dtype=obj["dtype"]).reshape(obj["shape"])
    return obj

def merge_dicts(dic1: Dict, dic2: Dict) -> Dict:
    """Merge two dicts ('dic1' and 'dic2') into one."""
    merged = {}
    all_keys = set(dic1.keys()).union(dic2.keys())

    for k in all_keys:
        v1 = dic1.get(k)
        v2 = dic2.get(k)
        if isinstance(v1, str):
            try:
                temp_v1 = ast.literal_eval(re.sub(r'\barray\(', 'np.array(', v1))
                if temp_v1 is not None:
                    v1 = temp_v1
            except:
                pass
        if isinstance(v2, str):
            try:
                temp_v2 = ast.literal_eval(re.sub(r'\barray\(', 'np.array(', v2))
                if temp_v2 is not None:
                    v2 = temp_v2
            except:
                pass

        try:
            if(isinstance(v1, list)):
                v1 = np.array(v1)
            if(isinstance(v2, list)):
                v2 = np.array(v2)
            if isinstance(v1, np.ndarray) and isinstance(v2, np.ndarray) and len(v1.shape) <= 1 and len(v2.shape) <= 1:
                merged[k] = np.concatenate((v1, v2))
            elif v1 is not None and v2 is None:
                merged[k] = v1
            elif v2 is not None and v1 is None:
                merged[k] = v2
            else:
                merged[k] = v1+v2
        except:
            merged[k] = v1

    return merged

def split_dict(dic:Dict, cut_index:int)->Tuple[Dict,Dict]:
    """Split a dict into two childrens by cutting it at 'cut_index'."""
    dic1 = {}
    dic2 = {} 
    for k in dic.keys():
        v = dic[k]
        if type(v) is str:
            try:
                temp_v = ast.literal_eval(re.sub(r'\barray\(', 'np.array(', v))
                if temp_v is None:
                    raise
                v = temp_v
            except:
                pass
        if type(v) is list or type(v) is np.ndarray:
            dic1[k] = v[:cut_index]
            dic2[k] = v[cut_index:]
            continue
        if not(k in dic1):
            dic1[k] = v
        if not(k in dic2):
            dic2[k] = v
    return (dic1, dic2)

def step_fill(x, y_lower, y_upper, log_bins=False, offset=1):
    x = np.asarray(x)
    y_lower = np.asarray(y_lower)
    y_upper = np.asarray(y_upper)
    
    if log_bins:
        log_centers = np.log10(x*offset)
        log_edges = np.zeros(len(x)+1)
        log_edges[1:-1] = 0.5 * (log_centers[1:] + log_centers[:-1])
        log_edges[0] = log_centers[0] - 0.5*(log_centers[1]-log_centers[0])
        log_edges[-1] = log_centers[-1] + 0.5*(log_centers[-1]-log_centers[-2])
        edges = 10**log_edges
    else:
        dx = np.diff(x)/2
        edges = np.zeros(len(x)+1)
        edges[1:-1] = x[:-1] + dx
        edges[0] = x[0] - dx[0]
        edges[-1] = x[-1] + dx[-1]
    
    x_step = np.repeat(edges, 2)[1:-1]
    y_lower_step = np.repeat(y_lower, 2)
    y_upper_step = np.repeat(y_upper, 2)
    
    return x_step, y_lower_step, y_upper_step

def plot_map(map, ax=None, cmap=None, norm=None, toplabel:Optional[str]=None, show_ax_labels:bool=True, return_im:bool=True,
             contour:Optional[np.ndarray]=None, contour_levels=10, contour_sigma=0, cbar=False, cbar_separate=True, clabel="", save:Optional[str]=None):
    if ax is None:
        fig = plt.figure(figsize=(4,4), dpi=300)
        ax = fig.add_subplot()
    else:
        fig = ax.figure

    im = ax.imshow(map, norm=norm, cmap=cmap)

    if contour is not None:
        if contour_sigma > 0:
            contour = gaussian_filter(contour, sigma=contour_sigma)
        cs = ax.contour(contour,levels=contour_levels,linewidths=2,alpha=1., colors="black")
        cs = ax.contour(contour,levels=contour_levels,linewidths=1,alpha=1, cmap="autumn")
        texts = ax.clabel(cs, inline=False, fontsize=8)
        for t in texts:
            t.set_path_effects([
                pe.withStroke(linewidth=2, foreground="black")
            ])
    if toplabel is not None:
        ax.text(0.02, 0.98,toplabel,transform=ax.transAxes,
        ha="left",va="top",fontsize=10,color="black", bbox=dict(facecolor="white",edgecolor="black", boxstyle="round,pad=0.2",alpha=1.))

    if save is not None:
        fig_s, _ = plot_map(map=map, ax=None, cmap=cmap, cbar=False, clabel=clabel, norm=norm, toplabel=toplabel, show_ax_labels=True, return_im=False, 
                               contour=contour, contour_levels=contour_levels, contour_sigma=contour_sigma, save=None)
        fig_s.tight_layout(pad=0.)
        fig_s.savefig(save)
        plt.close(fig_s)

    if cbar:
        if cbar_separate:
            fig.canvas.draw()
            pos = ax.get_position()
            cbar_fig = plt.figure(figsize=(1.2, fig.get_figheight()), dpi=fig.dpi)
            cax = cbar_fig.add_axes([0.2, pos.y0, 0.3, pos.height])
            cbar_obj = cbar_fig.colorbar(im, cax=cax)
            cbar_obj.set_label(clabel)
            if save is not None:
                cbar_fig.savefig(save+"_colorbar.png", bbox_inches="tight")
        fig.colorbar(im, ax=ax, label=clabel)

    if not(show_ax_labels):
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    if return_im:
        return im
    
    return fig, ax
    
from matplotlib.patches import Rectangle, FancyBboxPatch

def plot_rect_bg(fig, axes, color, pad=0.01, text=None, opacity=0.3, text_offset=+0.05, text_pos="top", show_bbox=None):
    bboxes = [ax.get_position() for ax in axes]

    xmin = min(bb.x0 for bb in bboxes)
    ymin = min(bb.y0 for bb in bboxes)
    xmax = max(bb.x1 for bb in bboxes)
    ymax = max(bb.y1 for bb in bboxes)

    rect = FancyBboxPatch(
        (xmin-pad, ymin-pad),
        (xmax - xmin) +2*pad,
        (ymax - ymin) +2*pad,
        boxstyle="round,pad=0.01",
        transform=fig.transFigure,
        facecolor=color,
        alpha=opacity,
        zorder=0
    )

    if text is not None:
        fig.text(
            xmin + 0.005,ymax + text_offset if text_pos == "top" else ymin - text_offset,
            text,ha="left",va=text_pos,
            fontsize=10,fontweight="bold",color="black",
            zorder=1,
            bbox=dict(facecolor=color,alpha=opacity,edgecolor="black",pad=2) if show_bbox else None
        )

    fig.add_artist(rect)

def longest_common_substring(strings):
    if not strings:
        return ""
    if len(strings) == 1:
        return strings[0]

    shortest = min(strings, key=len)

    def has_common(length):
        seen = {shortest[i:i+length] for i in range(len(shortest) - length + 1)}
        for s in strings[1:]:
            seen = {sub for sub in seen if sub in s}
            if not seen:
                return None
        return next(iter(seen))

    left, right = 0, len(shortest)
    result = ""

    while left <= right:
        mid = (left + right) // 2
        common = has_common(mid)
        if common:
            result = common
            left = mid + 1
        else:
            right = mid - 1

    return result

def find_roots(X, Y, interp="linear"):
    idx = np.where(np.diff(np.signbit(Y)))[0]
    if interp is None:
        return idx
    roots = []
    for i in idx:
        x0, x1 = X[i], X[i+1]
        y0, y1 = Y[i], Y[i+1]
        xr = x0 - y0 * (x1 - x0) / (y1 - y0)
        roots.append(xr)
    return np.array(roots)

def contour_3d(data_cube: np.ndarray, pos: Tuple[int, int, int], threshold: float) -> List[np.ndarray]:
    pos = tuple(pos)
    init_value = data_cube[*pos]
    threshold = threshold*init_value

    shape = data_cube.shape
    stack = [pos]
    visited = set([pos])
    volume = []

    neighbors = [
        (0, 0, 1), (0, 0, -1),
        (1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, -1, 0)
    ]

    while stack:
        cell = stack.pop()
        volume.append(np.array(cell))

        for dx, dy, dz in neighbors:
            nx, ny, nz = cell[0] + dx, cell[1] + dy, cell[2] + dz

            if not (0 <= nx < shape[0] and 0 <= ny < shape[1] and 0 <= nz < shape[2]):
                continue

            new_pos = (nx, ny, nz)

            if new_pos in visited:
                continue

            n_value = data_cube[nx, ny, nz]
            if n_value > init_value:
                continue
            if n_value < threshold:
                continue

            visited.add(new_pos)
            stack.append(new_pos)
        print(f"{len(volume)} : {new_pos} - {n_value}")

    return volume

def is_vector_in_box(pos:np.ndarray, center:np.ndarray, half_length:float):
        assert len(pos) == len(center), LOGGER.error(f"Position vector {len(pos)} has not the same size as Center vector {len(center)}")
        return all([pos[i] <= center[i] + half_length and pos[i] >= center[i] - half_length for i in range(len(pos))])

def is_vector_in_box_2(pos:np.ndarray, min:np.ndarray, max:np.ndarray):
        assert len(pos) == len(min), LOGGER.error(f"Position vector {len(pos)} has not the same size as min or max vector {len(min),len(max)}")
        return all([pos[i] <= max[i] and pos[i] >= min[i] for i in range(len(pos))])
