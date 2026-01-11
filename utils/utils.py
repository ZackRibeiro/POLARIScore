import numpy as np
from scipy.ndimage import rotate
from POLARIScore.config import LOGGER
import matplotlib.pyplot as plt
import platform
from typing import List, Tuple, Callable, Union, Dict
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

def convert_pc_to_index(pc:float,nres:int,size:float,start:float=0.,clip=True)->int:
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
    if (idx > 1 or idx < 0) and clip:
        return -1
    return (int(np.floor(idx*nres)))

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

def rotate_cube(data_cube:np.ndarray, angle:float, axis:int)->np.ndarray:
    """Rotates the cube around a given axis (0=X, 1=Y, 2=Z) by a given angle in degrees."""
    return rotate(data_cube, angle, axes=axis, reshape=False, mode="nearest")
def compute_pdf(data_slice:np.ndarray, bins:int=100, func:Callable=lambda x: np.log(x)/np.log(10), center:bool=False)->Tuple[List[float],List[float]]:
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
    hist = np.histogram(func(data_slice.flatten()[~np.isnan(data_slice.flatten())]),bins=bins, density=True)
    probabilities = hist[0]
    edges = hist[1]
    if(center):
        center_i = np.argmax(probabilities)
        edges = edges - (edges[center_i+1]+edges[center_i])/2
    return [probabilities, edges]

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """Print a progress bar"""
    if total == 0:
        total = 1
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    if iteration == total: 
        print()

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

def bin_mean(x, y, dx=None, nbins=None, logspace=True, stat='mean', min_per_bin=1):
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
    for i in range(len(bin_centers)):
        in_bin = which_bin == i
        if np.sum(in_bin) >= min_per_bin:
            if stat == 'mean':
                x_binned.append(np.mean(x[in_bin]))
                y_binned.append(np.mean(y[in_bin]))
            elif stat == 'median':
                x_binned.append(np.median(x[in_bin]))
                y_binned.append(np.median(y[in_bin]))
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

def dictsToString(dicts:List[Dict])->str:
    """Combine a list of dicts to a string with first line of keys and one line per dict."""
    string = ""
    keys = []
    for d in dicts:
        for k in d.keys():
            if k not in keys:
                keys.append(k)
    for k in keys:
        string = string + k + " "
    string = string + "\n"
    for d in dicts:
        for k in keys:
            if k in d:
                string = string + str(d[k])
            else:
                string = string + " "
            string = string + " "
        string = string + "\n"
    return string

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


def plot_lines(x:Union[np.ndarray,List,None],y:Union[np.ndarray,List,None], ax, lines:List[float]=[0,1,2], x_max:float=None, x_min:float=None, y_max:float=None, y_min:float=None, logspace=False):
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
