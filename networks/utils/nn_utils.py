import numpy as np
from torch.nn import init
from typing import *
import torch
import torch.nn.functional as F
from POLARIScore.utils.utils import printProgressBar
from POLARIScore.config import LOGGER, CACHES_FOLDER
import os

def init_network(model, init_method:Callable=init.kaiming_uniform_):
    """
    Init network weights using 'init_method' like kaiming or xavier.
    Args:
        model (nn.Module): Model instance to init
        init_method:
    """
    for layer in model.modules():
        if hasattr(layer, 'initialize') and callable(layer.initialize):
            continue
        if hasattr(layer, 'weight') and layer.weight is not None:
            if layer.weight.ndim >= 2:
                init_method(layer.weight)
            else:
                init.ones_(layer.weight)
        if hasattr(layer, 'bias') and layer.bias is not None:
            init.zeros_(layer.bias)

def compute_accuracy(label:np.ndarray, pred:np.ndarray, sigma:float=.1, log10:bool=True, bins:Union[List[float],None]=None, col_dens:Union[np.ndarray, None]=None)->Union[float,List[float]]:
    """
    Compute accuracy for a model using comparaison between true/target data(label) to prediction data.
    Args:
        label (np.ndarray): Ground truth data.
        pred (np.ndarray): Predicted data.
        sigma (float): Threshold for considering a prediction correct (in log space if log10=True).
        log10 (bool): If True, compute accuracy in log10 space.
        bins (list[float] or None): If provided, compute accuracy per bin of 'col_dens'.
        col_dens (np.ndarray or None): Column density data (same shape as label/pred).
    Returns:
        accuracy:float or List[float] if binned accuracy
    """
    if log10:
        pred = np.log(pred)/np.log(10)
        label = np.log(label)/np.log(10)
    corrects = (np.abs(pred-label) <= sigma)

    if bins is None:
        acc = corrects.sum() / (corrects.shape[0]*corrects.shape[1])
        return acc
    
    if(col_dens is None):
        col_dens = label
    
    col_dens_flat = col_dens.flatten()
    corrects_flat = corrects.flatten()

    bin_acc = []
    for i in range(len(bins) - 1):
        mask = (col_dens_flat >= bins[i]) & (col_dens_flat < bins[i + 1])
        if mask.sum() > 0:
            acc_bin = corrects_flat[mask].sum() / mask.sum()
        else:
            acc_bin = np.nan
        bin_acc.append(acc_bin)

    return bin_acc

def compute_batch_accuracy(batch:List[Tuple[np.ndarray,np.ndarray]], sigma:float=.1, log10:bool=True, bins:Union[List[float],None]=None, col_dens:Union[np.ndarray, None]=None)->Union[Tuple[float,float],List[Tuple[float,float]]]:
    """
    Compute batch accuracy for a model using comparaison between true/target data(label) to prediction data.
    Args:
        batch: List with tuples: (Ground truth data, Predicted data).
        sigma (float): Threshold for considering a prediction correct (in log space if log10=True).
        log10 (bool): If True, compute accuracy in log10 space.
        bins (list[float] or None): If provided, compute accuracy per bin of 'col_dens'.
        col_dens (np.ndarray or None): Column density data (same shape as label/pred).
    """
    acc = []
    for label, pred in batch:
        acc.append(compute_accuracy(label, pred, sigma, log10, bins=bins, col_dens=col_dens))
    if isinstance(acc[0], list) and bins is not None:
        return [
            (np.nanmean([a[i] for a in acc]), np.nanstd([a[i] for a in acc]))
            for i in range(len(bins) - 1)
        ]
    return (np.nanmean(acc), np.nanstd(acc))

def find_error_for_batch_accuracy(batch, accuracy=0.8, epsilon=0.01):
    acc = epsilon+1.
    ite = 0
    sigma0 = 0.
    sigma1 = 1.
    acc0 = compute_batch_accuracy(batch, sigma=sigma0)[0]-accuracy
    while abs(acc) > epsilon and ite < 100:
        ite += 1
        sigma = (sigma1+sigma0)/2
        acc = compute_batch_accuracy(batch, sigma=sigma)[0]-accuracy
        if acc*acc0 >= 0:
            acc0 = acc
            sigma0 = sigma
        else:
            sigma1 = sigma
    return sigma

def predict_map(data, model_trainer:'Trainer', 
                method:Literal['mean','max','likeliest','median','min','sampling']="mean",
                kernel:Union[Literal['gaussian','quadratic','uniform'],Callable[[float],float]]='uniform',
                repeat:int=1, repeat_batch:int=1,
                save_samples:Optional[str]=None, skip_using_saved_samples:bool=False, only_error:bool=True,
                patch_size:Tuple[int,int]=(128, 128), nan_value:float=-1.0, overlap:float=0.5, downsample_factor:float=1., apply_baseline:bool=True, give_error:bool=False):
    """
    Predict a quantity by applying a neural network to a map (2D tensor).
    Args:
        model_trainer (Trainer): Model wrapped in a Trainer object.
        method (str): Method used to make the final prediction (by default mean), be aware that other methods can need the full distributions and so use a lot of memory.
        kernel (str): Kernel used to weight the count.
        repeat (int): How many time the network is runned on the same context region,
        repeat_batch (int): Batch in the same tensor
        save_samples (str): If not none, save the pdf map as a file named by this arg (in the cache folder)
        skip_using_saved_samples (bool): If True, try to fetch the samples file and use it to skip prediction.
        patch_size (tuple[int, int]): Shape of the 2D patches on which the model will be applied. The observation will be divided into patches of this shape.
        nan_value (float): Value used to replace NaNs in the observation.
        overlap (float): Fraction of overlap between consecutive patches.
        downsample_factor (float): Factor by which the observation is downsampled.
        baseline (bool): Whether to apply baseline correction to the model.
    Returns:
        predicted_observation
    """

    repeat = max(repeat,1)
    method = method.lower()
    has_kernel = not(isinstance(kernel, str)) or kernel != "uniform"
    if isinstance(kernel, str):
        kernel = kernel.lower()
    LOGGER.log(f"Predicting map using {method} method with {model_trainer.network_type} and kernel: {kernel}")
    if method in ["mean","max","min"] and not(save_samples is not None and len(save_samples) > 0) and not(has_kernel):
        return (predict_map_reduced(data, model_trainer, method, patch_size, nan_value, overlap, downsample_factor, apply_baseline),None)

    input_matrix = data
    nan_mask = np.isnan(input_matrix) | (input_matrix <= 0)
    if nan_value < 0:
        nan_value = float(np.nanmin(data[data>0]))
    input_matrix[nan_mask] = nan_value
    input_tensor = torch.tensor(input_matrix.astype(np.float32))
    downsampled_tensor = F.interpolate(input_tensor.unsqueeze(0).unsqueeze(0), 
                                    scale_factor=1.0/downsample_factor, 
                                    mode='bilinear', align_corners=True).squeeze(0).squeeze(0)
    
    downsampled_nan_mask = F.interpolate(torch.tensor(nan_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0),
                                            scale_factor=1.0 / downsample_factor,mode='nearest'
                                            ).squeeze(0).squeeze(0).numpy().astype(bool)

    height, width = downsampled_tensor.shape
    patch_height, patch_width = patch_size
    stride_height = int(patch_height * (1 - overlap))
    stride_width = int(patch_width * (1 - overlap))
    i_range = range(0, height - patch_height + 1, stride_height)
    j_range = range(0, width - patch_width + 1, stride_width)
    
    coverage = np.zeros((height, width), dtype=np.int32)
    for i in i_range:
        for j in j_range:
            coverage[i:i+patch_height, j:j+patch_width] += 1
    max_samples = int(coverage.max())*repeat
    LOGGER.log(f"Samples per pixel: {max_samples}")
    LOGGER.log(f"Number of predictions: {len(i_range)*len(j_range)*repeat}")

    samples_tensor = np.empty((height, width, max_samples), dtype=np.float32)
    """make memmap:
    samples_tensor = np.memmap(samples_path,dtype=np.float32,mode='w+',shape=(max_samples, height, width))
    memmap output
    """
    samples_tensor.fill(np.nan)
    count_tensor = np.zeros((height, width), dtype=np.int32)

    if has_kernel:
        weight_tensor = np.zeros_like(samples_tensor)
    skip_prediction = False
    skip_kernel = not(has_kernel)
    if skip_using_saved_samples:
        samples_path = os.path.join(CACHES_FOLDER, save_samples)
        if ".npy" not in samples_path:
            samples_path += ".npy"
        if os.path.exists(samples_path):
            samples_tensor = np.load(samples_path, mmap_mode='r')
            kernel_path = samples_path.split(".npy")[0]+"_"+kernel+"_weight.npy"
            if os.path.exists(kernel_path):
                if has_kernel:
                    weight_tensor = np.load(kernel_path, mmap_mode='r')
                skip_kernel = True
                count_tensor = np.sum(~np.isnan(samples_tensor), axis=-1)
            skip_prediction = True
    if skip_prediction:
        LOGGER.log("Skip prediction is ON and the samples file was found -> Skipping model application")
    if not(skip_prediction and skip_kernel):
        for i0,i in enumerate(i_range):
            for j0,j in enumerate(j_range):
                patch = downsampled_tensor[i:i+patch_height, j:j+patch_width].cpu().detach().numpy()
                patch = np.expand_dims(patch, axis=0)
                valid_patch_mask = downsampled_nan_mask[i:i + patch_height, j:j + patch_width]

                if np.any(valid_patch_mask):
                    printProgressBar(i0*len(j_range)*repeat+j0*repeat,len(i_range)*len(j_range)*repeat,prefix="Obs Pred")
                    continue
                for k in range(repeat):
                    printProgressBar(i0*len(j_range)*repeat+j0*repeat+k,len(i_range)*len(j_range)*repeat,prefix="Obs Pred")
                    
                    #Work only for 1 output: col density
                    if not(skip_prediction):
                        output_patch = model_trainer.predict_tensor(patch, input_names="cdens", output_names="vdens")[0]
                        if apply_baseline:
                            output_patch = model_trainer.apply_baseline(output_patch, log=False)

                    co = count_tensor[i:i+patch_height, j:j+patch_width]
                    for di in range(patch_height):
                        for dj in range(patch_width):
                            idx = co[di, dj]
                            if not(skip_prediction):
                                samples_tensor[i+di, j+dj, idx] = output_patch[di, dj]
                            count_tensor[i+di, j+dj] += 1
                            if has_kernel:
                                weight = 1.
                                distance_to_center = np.sqrt((patch_height/2-di)**2+(patch_width/2-dj)**2)
                                distance_to_center /= max(patch_height,patch_width)/2
                                if isinstance(kernel, str):
                                    if kernel == "gaussian":
                                        sigma = 0.3
                                        weight = np.exp(-(distance_to_center**2) / (2 * sigma**2))
                                    elif kernel == "quadratic":
                                        weight = 1/(distance_to_center**2+1e-3)
                                else:
                                    weight = kernel(distance_to_center)
                                weight_tensor[i+di, j+dj, idx] = weight
    H,W,C = samples_tensor.shape


    if give_error:
        error = np.std(samples_tensor, axis=-1)
        upsampled_error = F.interpolate(torch.from_numpy(error).unsqueeze(0).unsqueeze(0), 
                                    size=(input_matrix.shape[0], input_matrix.shape[1]), 
                                    mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
        error = upsampled_error.numpy()
        error[nan_mask] = np.nan
        if only_error:
            return None, error
    

    if save_samples is not None:
        samples_path = os.path.join(CACHES_FOLDER, save_samples)
        if ".npy" not in samples_path:
            samples_path += ".npy"
        if not(skip_prediction):
            if os.path.exists(samples_path):
                os.remove(samples_path)
            np.save(samples_path, samples_tensor)
            LOGGER.log(f"Samples tensor saved at {samples_path}")
        if has_kernel:
            kernel_path = samples_path.split(".npy")[0]+"_"+kernel+"_weight.npy"
            if os.path.exists(kernel_path):
                os.remove(kernel_path)
            np.save(kernel_path, weight_tensor)
            LOGGER.log(f"Weight tensor saved at {kernel_path}")        
        
                
    
    if method == "mean":
        if has_kernel:

            K = samples_tensor.shape[-1]
            idx = np.arange(K)
            mask = idx < count_tensor[..., None]
            weighted_sum = np.sum(samples_tensor * weight_tensor * mask,axis=-1)

            weight_sum = np.sum(weight_tensor * mask,axis=-1)

            output = np.divide(weighted_sum,weight_sum,
                out=np.full((H, W), np.nan, dtype=np.float32),where=weight_sum > 0)
        else:
            output = np.nanmean(samples_tensor, axis=-1)
    elif method == "max":
        if has_kernel:
            LOGGER.warn(f"Kernel not yet implemented in {method}.")
        output = np.nanmax(samples_tensor, axis=-1)
    elif method == "min":
        if has_kernel:
            LOGGER.warn(f"Kernel not yet implemented in {method}.")
        output = np.nanmin(samples_tensor, axis=-1)
    elif method == "median":
        if has_kernel:
            LOGGER.warn(f"Kernel not yet implemented in {method}.")
        output = np.nanmedian(samples_tensor, axis=-1)

    elif method == "sampling":
        if has_kernel:
            LOGGER.warn(f"Kernel not yet implemented in {method}.")
        idx = np.random.randint(0, count_tensor, size=(H, W))
        output = samples_tensor[np.arange(H)[:,None], np.arange(W), idx]

    elif method == "likeliest":
        output = np.zeros((H, W), dtype=np.float32)
        for y in range(H):
            for x in range(W):
                printProgressBar(y*W+x,W*H,prefix="Estimating likeliest values")
                n = count_tensor[y, x]
                if n == 0:
                    output[y, x] = np.nan
                    continue
                vals = samples_tensor[y, x, :n]
                                
                finite_mask = np.isfinite(vals)
                vals = vals[finite_mask]

                if has_kernel:
                    w = weight_tensor[y, x, :n][finite_mask]
                else:
                    w = None

                if len(vals) == 0:
                    output[y, x] = np.nan
                    continue

                vmin = np.min(vals)
                vmax = np.max(vals)
                if not np.isfinite(vmin) or not np.isfinite(vmax):
                    output[y, x] = np.nan
                    continue

                if np.isclose(vmin, vmax, rtol=1e-6, atol=1e-12):
                    output[y, x] = vmin
                    continue
                hist, bin_edges = np.histogram(vals, bins=min(32, max(8, int(np.sqrt(n)))), density=True, weights=w)
                max_bin = np.argmax(hist)
                output[y, x] = 0.5 * (bin_edges[max_bin] + bin_edges[max_bin + 1])
                del vals
    else:
        raise NotImplementedError()


    upsampled_output = F.interpolate(torch.from_numpy(output).unsqueeze(0).unsqueeze(0), 
                                    size=(input_matrix.shape[0], input_matrix.shape[1]), 
                                    mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
    output_matrix = upsampled_output.numpy()
    output_matrix[nan_mask] = np.nan

    if give_error:
        return output_matrix, error
    return output_matrix


#TODO Estimate Error (compute X**2)
def predict_map_reduced(data, model_trainer:'Trainer', method:Literal["mean","max","min"],
                patch_size:Tuple[int,int]=(128, 128), nan_value:float=-1.0, overlap:float=0.5, downsample_factor:float=1., apply_baseline:bool=True):
    """
    Predict a quantity by applying a neural network to an observation.
    Args:
        model_trainer (Trainer): Model wrapped in a Trainer object.
        patch_size (tuple[int, int]): Shape of the 2D patches on which the model will be applied. The observation will be divided into patches of this shape.
        nan_value (float): Value used to replace NaNs in the observation.
        overlap (float): Fraction of overlap between consecutive patches.
        downsample_factor (float): Factor by which the observation is downsampled.
        baseline (bool): Whether to apply baseline correction to the model.
    Returns:
        predicted_observation
    """

    input_matrix = data
    nan_mask = np.isnan(input_matrix) | (input_matrix <= 0)
    if nan_value < 0:
        nan_value = float(np.nanmin(data[data>0]))
    input_matrix[nan_mask] = nan_value
    input_tensor = torch.tensor(input_matrix.astype(np.float32))
    downsampled_tensor = F.interpolate(input_tensor.unsqueeze(0).unsqueeze(0), 
                                    scale_factor=1.0/downsample_factor, 
                                    mode='bilinear', align_corners=True).squeeze(0).squeeze(0)
    
    downsampled_nan_mask = F.interpolate(torch.tensor(nan_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0),
                                            scale_factor=1.0 / downsample_factor,mode='nearest'
                                            ).squeeze(0).squeeze(0).numpy().astype(bool)

    height, width = downsampled_tensor.shape
    patch_height, patch_width = patch_size
    stride_height = int(patch_height * (1 - overlap))
    stride_width = int(patch_width * (1 - overlap))

    output_tensor = torch.zeros_like(downsampled_tensor)
    count_tensor = torch.zeros_like(downsampled_tensor)

    i_range = range(0, height - patch_height + 1, stride_height)
    j_range = range(0, width - patch_width + 1, stride_width)

    for i0,i in enumerate(i_range):
        for j0,j in enumerate(j_range):
            printProgressBar(i0*len(j_range)+j0,len(i_range)*len(j_range),prefix="Obs Pred")
            patch = downsampled_tensor[i:i+patch_height, j:j+patch_width].cpu().detach().numpy()
            patch = np.expand_dims(patch, axis=0)
            valid_patch_mask = downsampled_nan_mask[i:i + patch_height, j:j + patch_width]

            if np.any(valid_patch_mask):
                continue
            
            #Work only for 1 output: col density
            output_patch = model_trainer.predict_tensor(patch, input_names="cdens", output_names="vdens")[0]
            if apply_baseline:
                output_patch = model_trainer.apply_baseline(output_patch, log=False)
            
            if method == 'mean':
                output_tensor[i:i+patch_height, j:j+patch_width] += torch.from_numpy(output_patch)
            elif method == 'max':
                output_tensor[i:i+patch_height, j:j+patch_width] = torch.maximum(torch.from_numpy(output_patch), output_tensor[i:i+patch_height, j:j+patch_width])
            elif method == 'min':
                output_tensor[i:i+patch_height, j:j+patch_width] = torch.minimum(torch.from_numpy(output_patch), output_tensor[i:i+patch_height, j:j+patch_width])
            count_tensor[i:i+patch_height, j:j+patch_width] += 1

    print("")
    if method == "mean":
        output_tensor = output_tensor / count_tensor

    upsampled_output = F.interpolate(output_tensor.unsqueeze(0).unsqueeze(0), 
                                    size=(input_matrix.shape[0], input_matrix.shape[1]), 
                                    mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
    output_matrix = upsampled_output.numpy()
    output_matrix[nan_mask] = np.nan

    return output_matrix

from POLARIScore.objects.SpectrumMap import SpectrumMap
def open_samples_as_spectrummap(path:str, bins:int=32, is_in_cache_folder:bool=True)->'SpectrumMap':
    if is_in_cache_folder and CACHES_FOLDER not in path:
        path = os.path.join(CACHES_FOLDER, path)
    if ".npy" not in path:
        path += ".npy"
    if not(os.path.exists(path)):
        LOGGER.error(f"Can't open samples because there is no such file: {path}.") 
    kernel_path = path.split(".npy")[0]+"_weight.npy"
    has_kernel = os.path.exists(kernel_path)
    if has_kernel:
        weight_tensor = np.load(kernel_path, mmap_mode='r')
    else:
        weight_tensor = None
    samples_tensor = np.load(path, mmap_mode='r')
    count_tensor = np.sum(~np.isnan(samples_tensor), axis=-1)
    
    H,W,C = samples_tensor.shape
    pdf_map = np.zeros((H,W,bins))
    for y in range(H):
        for x in range(W):
            printProgressBar(y*W+x,W*H,prefix="Transforming samples tensor to a pdf tensor")
            n = count_tensor[y, x]
            if n == 0:
                continue
            vals = samples_tensor[y, x, :n]
                            
            finite_mask = np.isfinite(vals)
            vals = vals[finite_mask]

            if has_kernel:
                w = weight_tensor[y, x, :n][finite_mask]
            else:
                w = None

            if len(vals) == 0:
                continue

            vmin = np.min(vals)
            vmax = np.max(vals)
            if not np.isfinite(vmin) or not np.isfinite(vmax):
                continue

            if np.isclose(vmin, vmax, rtol=1e-6, atol=1e-12):
                continue
            hist, bin_edges = np.histogram(vals, bins=min(bins, max(8, int(np.sqrt(n)))), density=True, weights=w)
            if len(hist) < bins:
                hist = np.pad(hist, (0, bins - len(hist)), mode='constant') 
            del vals
            max_bin = np.argmax(hist)
            #Multiply by the likeliest value to have a good looking integrated map when opened with SpectrumMap.
            hist *= .5*(bin_edges[max_bin] + bin_edges[max_bin + 1])/np.sum(hist)
            pdf_map[y,x] = hist
    
    map = SpectrumMap(name="pdf_map", map=pdf_map, load=False)
 
    map.output_settings['velocity_channels'] = bins
    map.output_settings['velocity_resolution'] = 1
    map.output_settings['v_function'] = lambda lsr,chan,res: np.array(range(chan))
    map.output_settings['velocity_unit'] = ""
    map.output_settings['velocity_name'] = "Channels"
    map.output_settings['intensity_unit'] = ""
    map.output_settings['intensity_name'] = "Prediction"

    return map
    