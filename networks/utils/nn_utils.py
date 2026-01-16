import numpy as np
from torch.nn import init
from typing import List, Tuple, Union, Callable

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
    while acc > epsilon and ite < 100:
        ite += 1
        sigma = (sigma1-sigma0)/2
        acc = compute_batch_accuracy(batch, sigma=sigma)[0]-accuracy
        if acc*acc0 >= 0:
            acc0 = acc
            sigma0 = sigma
        else:
            sigma1 = sigma
    return acc+accuracy