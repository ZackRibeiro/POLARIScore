import torch
import torch.nn as nn
import numpy as np
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from POLARIScore.config import LOGGER
from typing import Union, List
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

class BaseModule(nn.Module):
    def __init__(self, **args):
        super(BaseModule, self).__init__(**args)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._plot = None
        self.to(self.device)

    def _plot_tensor(self, tensor:torch.tensor, subfolder:Union[str, None]=None, name:str="feature"):
        if self._plot is None:
            return
        folder_path = self._plot
        if subfolder is not None:
            folder_path = os.path.join(folder_path, subfolder)
        if not(os.path.exists(folder_path)):
            os.mkdir(folder_path)

        np_tensor = tensor.clone().squeeze(0).cpu().detach().numpy()
        for c in range(np_tensor.shape[0]):
            fig, ax = plt.subplots()
            try:
                ch_tensor = np_tensor[c]
                vmin = np.nanpercentile(ch_tensor, 0)
                vmax = np.nanpercentile(ch_tensor, 100)
                if vmin <= 0:
                    vmin = np.nanmin(ch_tensor[ch_tensor > 0])
                levels = np.linspace(vmin, vmax, 20)
                img_plt = ax.imshow(ch_tensor, cmap="rainbow", vmin=vmin, vmax=vmax, origin="lower")
                plt.colorbar(img_plt, ax=ax, label=r"intensity")
                contour_plt = ax.contour(ch_tensor, levels=levels, colors="black", origin="lower", linewidths=0.1)

                path = os.path.join(folder_path, name+"_"+str(c)+".jpg")
                plt.savefig(path, dpi=300)
            except:
                pass
            plt.close(fig)
            del fig
        del np_tensor

    def plot_features(self, x:Union[np.ndarray, torch.tensor], save_path:str):
        """Plot and saves all features created by using input x into the network.
        Args:
            x: Tensor input, ndarray will be transformed into torch tensors.
            save_path: Folder save path
        """

        if isinstance(x, np.ndarray):
            x = self.shape_tensor(x)

        self._plot = save_path
        if not(type(save_path) is str):
            LOGGER.error("Save path need to be a string.")
            return
        if not(os.path.exists(save_path)):
            LOGGER.error(f"Root path need to exists: {save_path}")
            return
        LOGGER.log(f"Plotting & Saving features of neural network in {save_path}")
        self._plot_tensor(self.forward(x), name="output")
        self._plot = None

    def shape_tensor(self, tensor:Union[np.ndarray,torch.tensor], name=None, reverse=False, **args):
        """
        Shape a ndarray to a torch tensor or reverse.
        Args:
            tensor:
        """

        if reverse:
            tensor = tensor.squeeze(0).squeeze(0).cpu().detach().numpy()

        is_segmentation = False
        if 'segmentation' in args and type(args['segmentation']) is bool:
            is_segmentation = args['segmentation']

        if name is not None and 'norms' in args and name in args['norms']:
            tensor = args['norms'][name][0 if not(reverse) else 1](tensor)
        else:
            tensor = np.log(1.+np.clip(tensor, a_min=0, a_max=None)) if not(reverse) else (tensor if is_segmentation else np.exp(tensor)-1.)
        
        if(reverse and not(is_segmentation)):
            np.clip(tensor,a_min=1.,a_max=None)
        if(np.isinf(tensor).any()):
            tensor = np.zeros_like(tensor)+1
            
        if reverse:
            return tensor
        
        rslt = torch.from_numpy(tensor).float()
        #TODO adapt to 3D tensors (B,C,H,W,D)
        if(len(tensor.shape)==2):
            rslt = rslt.unsqueeze(0)
        if len(tensor.shape)<4:
            rslt = rslt.unsqueeze(1)
        return rslt.to(self.device)

    def shape_batch(self, batch, target_indexes:Union[int,List[int]]=[], input_indexes:Union[int, List[int]]=[], target_names:Union[str,List[str],None]=None, input_names:Union[str,List[str],None]=None, **args):
        """
        Shape a input batch, i.e list of objects like ndarray or list of ndarray, to list of torch tensor.
        Args:
            batch:
            target_indexes:
            input_indexes:
            reverse (bool): if True, the data is from torch to numpy
        """
        
        assert (len(batch)) > 0, LOGGER.error("Can't apply the model on the batch, the batch is empty.")

        if target_names:
            target_names = target_names if type(target_names) is list else [target_names]
        if input_names:
            input_names = input_names if type(input_names) is list else [input_names]
        input_indexes = input_indexes if type(input_indexes) is list else [input_indexes]
        input_tensors = None
        for i,t in enumerate(input_indexes):
            if input_tensors is None:
                input_tensors = [self.shape_tensor(np.array([b[t] for b in batch]), name=input_names[i] if input_names else None, **args)]
            else:
                input_tensors.append(self.shape_tensor(np.array([b[t] for b in batch]), name=input_names[i] if input_names else None, **args))

        target_indexes = target_indexes if type(target_indexes) is list else [target_indexes]
        target_tensors = None
        for i,t in enumerate(target_indexes):
            if target_tensors is None:
                target_tensors = [self.shape_tensor(np.array([b[t] for b in batch]), name=target_names[i] if target_names else None, **args)]
            else:
                target_tensors.append(self.shape_tensor(np.array([b[t] for b in batch]), name=target_names[i] if target_names else None, **args))

        if target_tensors != None and input_tensors != None:
            return input_tensors, target_tensors
        elif target_tensors != None:
            return target_tensors
        elif input_tensors != None:
            return input_tensors
        else:
            LOGGER.warn("Shape batch didn't work because there is no input AND target tensors.")
            return None
    
    def forward(self, x):
        if type(x) is list:
            if len(x) > 1:
                return torch.cat(x, dim=1)
            else:
                return x[0]
        return x
