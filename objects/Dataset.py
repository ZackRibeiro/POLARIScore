import uuid
import os
import shutil
import re
from POLARIScore.config import *
from POLARIScore.utils.utils import plot_lines, printProgressBar
import json
import glob
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import shutil
from matplotlib.widgets import Slider
from matplotlib.axis import Axis
from typing import Any, Dict, List, Union, Tuple, Literal, Callable
import copy
import re
from POLARIScore.utils.utils import NumpyEncoder, numpy_decoder, merge_dicts, split_dict

""" 
Example of what a batch can contains:
- 'cdens': tensor NxN
- 'vdens': tensor NxN
- 'cospectra': tensor NxNxdepth
- 'density': tensor NxNxdepth
- 'cdens_context': tensor 2xNxN (cdens, crop_mask)
- 'physize': tensor n (1 if context is off else 2)
"""

def _formate_name(name:str):
    return name

def _open_batch(batch_name:str)->Tuple[Dict[int, List], List[str]]:
    assert os.path.exists(TRAINING_BATCH_FOLDER), LOGGER.error(f"Can't open batch {batch_name}, no folder exists.")
    batch_path = os.path.join(TRAINING_BATCH_FOLDER,batch_name)

    pattern = re.compile(r"^(?!.*\/(settings|data)\.json$).*\.(npy|json)$")
    files = [f for f in glob.glob(batch_path + "/*") if pattern.match(f)]
    files = [f.split("/")[-1] for f in files]
    files = sorted(files)

    data_dict = {}
    order = []

    batch_contains = []
    for f in files:
        name = f.split(".")[0].split("_", 1)[1]
        if not(name in batch_contains):
            batch_contains.append(name)
    
    for bc in batch_contains:
        pot_files = [f for f in files if bc == f.split(".")[0].split("_", 1)[1]]
        if len(pot_files) <= 0 :
            continue

        ids = [int(f.split("_")[0]) for f in pot_files]
        indexes = np.argsort(ids)
        for i in indexes:
            if i not in data_dict:
                data_dict[i] = []
            data_dict[i].append(os.path.join(batch_path,pot_files[i]))
        order.append(bc)
    return data_dict, order

def getDataset(name:str)->"Dataset":
    """Get dataset by name"""
    ds = Dataset()
    try:
        ds.load_from_name(name, change_name=True)
    except AssertionError:
        LOGGER.error(f"Can't load dataset: {name}")
        return None
    except FileNotFoundError:
        if not("batch_" in name):
            name = "batch_"+name
        try:
            ds.load_from_name(name, change_name=True)
        except FileNotFoundError:
            LOGGER.error(f"Dataset folder was not found. ({name})")
            return None
    return ds

class Dataset():
    """Datasets contains just the imgs paths for reducing the memory usage"""
    def __init__(self):
        self.batch:Dict[int,List[str]] = {}
        """A batch is a dict of file paths list. 
        For example for one element of the list, it can have two paths: one for volume density and another one for column density."""
        self.settings:Dict = {}
        """settings (can) contains:
            'order', eg: 'order':['cdens','vdens','cospectra']
            'img_number': nbr of imgs
            'img_size': size of an image in parsec (can be a list of sizes)
            'scores': score for each image computed using score_fct
            'scores_fct', (make score_settings TODO)
            'random_rotate': Does the images were randomly rotated when the dataset was generated.
        """
        self.data: Dict = {}
        """
        Datas which are not np matrix. Like a float per image. <br /> It can contains:
        'physical_size': physical size of the regions in parsec (List[float])
        """
        self.name:str = str(uuid.uuid4())
        self.active_batch:List[Union[List[np.ndarray],np.ndarray]] = []
        """Same as self.batch but instead of paths, it's loaded arrays."""

    def get_element_index(self, names:Union[str, List[str]])->Union[List[int],int,None]:
        """In self.batch, an element like col density or vol density can be at any place in the list.
          So to get the path or data, we need to get the corresponding index. 
          <br />*(Yes, i keep digging in my mistake of not using dicts (TODO?))*  
        """
        assert "order" in self.settings, LOGGER.error("No order list in dataset settings")

        names = names if type(names) is list else [names]

        indexes = []
        for n in names:
            found = False
            for i,o in enumerate(self.settings["order"]):
                if o == _formate_name(n):
                    indexes.append(i)
                    found = True
                    break
            assert found, LOGGER.error(f"Index not found for {n}")

        if len(indexes) == 1:
            return indexes[0]
        if len(indexes) == 0:
            return None
        
        return indexes

    def load_from_name(self, name:str, change_name:bool=False):
        """
        Args:
            name(str): batch folder name
            change_name(bool): if we change self.name with name
        """
        LOGGER.log(f"Loading dataset {name}")
        if change_name:
            self.name = name
        batch, order = _open_batch(name)
        self.batch = {**self.batch, **batch}

        settings = {}
        with open(os.path.join( os.path.join(TRAINING_BATCH_FOLDER,name),'settings.json')) as file:
            settings = json.load(file, object_hook=numpy_decoder)

        self.settings = settings            
        self.settings["order"] = order
        if "areas_explored" in settings:
            self.settings["areas_explored"] = eval(settings["areas_explored"].replace('array', 'np.array')) if type(settings["areas_explored"]) is str else settings["areas_explored"]
        if "img_size" in settings:
            self.settings["img_size"] = settings["img_size"]

    def add(self,imgs_path:Union[str,List[str]]):
        self.batch.append(imgs_path)

    def remove(self, indexes:Union[int,List[int]])->List[str]:
        
        assert os.path.exists(TRAINING_BATCH_FOLDER)
        batch_path = os.path.join(TRAINING_BATCH_FOLDER,"batch_"+str(self.name).split("batch_")[-1])
        assert os.path.exists(batch_path)
        indexes = [indexes] if isinstance(indexes, int) else indexes
        for i in indexes:
            if i not in self.batch:
                continue
            element_paths = self.batch[i]
            for path in element_paths:
                os.remove(path)
            del self.batch[i]
        LOGGER.log(f"{len(indexes)} removed form dataset.")
        return indexes
    
    def get(self, indexes:Union[List[int],int,None] = None):
        """Returns a list of pairs if indexes is a list of integers, else returns just the pair of data corresponding to the given index."""
        if len(self.batch) == 0:
            LOGGER.error("Can't load images in dataset because it's empty.")
            return
        if not(indexes is None):
            if not(isinstance(indexes, (torch.Tensor, np.ndarray, list))):
                return self.load(self.batch[indexes])
            elif len(indexes) < 2:
                return self.load(self.batch[int(indexes[0])])
            else:
                to_load = []
                for i in indexes:
                    if np.int64(i) in self.batch:
                        to_load.append(self.batch[np.int64(i)])
                    elif np.int32(i) in self.batch:
                            to_load.append(self.batch[np.int32(i)])
                    else:
                        to_load.append(self.batch[i])
                return self.load(to_load)
        else:
            return self.load(self.batch.values())

    def load(self, paths:List[Union[List, np.ndarray, str]])->List[Union[List[np.ndarray],np.ndarray]]:
        result = []
        def _load(path):
            ext = path.split(".")[-1]
            if ext == "json":
                with open(path) as file:
                    return json.load(file, object_hook=numpy_decoder)
            elif ext == "npy":
                return np.load(path)
        for pair in paths:
            if not(type(pair) is list or type(pair) is np.ndarray):
                result.append(_load(pair))
                continue
            temp = []
            for p in pair:
                temp.append(_load(p))
            result.append(temp)
        del self.active_batch
        self.active_batch = result
        return result
    
    def transform(self, channel_names:Union[List[str],str], method:Literal["split"], new_names:Union[List[str],str]=None):
        """
        Transform channels into new ones.
        """
        LOGGER.log(f"Transforming {channel_names} using {method}.")
        channel_names = channel_names if type(channel_names) is list else [channel_names]
        if new_names is not None:
            new_names = new_names if type(new_names) is list else [new_names]
            assert len(channel_names) == len(new_names), LOGGER.error("When specified, new names need to be the same length of channel names.")
        

        channel_indexes = ([self.get_element_index(c) for c in channel_names])

        order = self.settings['order']
        for bi in self.batch.keys():
            batch = self.get(bi)
            for ci, channel_index in enumerate(channel_indexes):
                img = batch[channel_index]

                if method=="split":
                    assert len(img.shape) == 3, LOGGER.error("To use 'split', the tensor need to be 3D.")
                    for s,sli in enumerate(img.transpose()):
                        if not(order[channel_index]+str(s) in order):
                            order.append(order[channel_index]+str(s))
                        batch.append(sli)
            self.save_batch(batch=batch, i=bi)
        self.settings['order'] = order
        self.save_settings()    
                

    def split(self, cutoff:float=0.7)->Tuple['Dataset','Dataset']:
        """
        Divide the dataset into two subsets, split at the cutoff parameter.
        """
        LOGGER.log(f"Splitting dataset {self.name} with cutoff at {cutoff}")
        batch_keys = np.array(list(self.batch.keys()))
        cut_index = int(cutoff * len(batch_keys))

        b1_settings, b2_settings = split_dict(self.settings, cut_index)
        b1_settings["order"] = self.settings["order"]
        b2_settings["order"] = self.settings["order"]
        b1_data, b2_data = split_dict(self.data, cut_index)

        def _make_new_dict(data, indexes):
            new_dict = {}
            for i in indexes:
                new_dict[i] = data[i]
            return new_dict 
                    
        b1 = Dataset()
        b1.batch = _make_new_dict(self.batch,batch_keys[:cut_index])
        b1.settings = b1_settings
        b1.data = b1_data
        b1.name = self.name + "_b1"
        b2 = Dataset()
        b2.batch = _make_new_dict(self.batch,batch_keys[cut_index:])
        b2.settings = b2_settings
        b2_data = b2_data
        b2.name = self.name + "_b2"

        return (b1, b2)
    
    def merge(self, dataset:Union['Dataset',List['Dataset']],force:bool=False,delete:bool=False,name:str=None,save:bool=False)->'Dataset':
        """Merge the dataset with another dataset (or list of datasets).
        Args:
            dataset: the other dataset(s)
            force: force the merging, like if a dataset have extra elements that others don't have, the extra channel will be removed.
            delete: delete older datasets
            save: save the new merged dataset
            name: Name of the new merged dataset
        Returns:
            merged_dataset
        """
        def _merge(ds1, ds2):
            LOGGER.log(f"Merging dataset {ds1.name} with dataset {ds2.name}")

            result_ds = Dataset()

            flag = False
            o1 = ds1.settings['order']
            o2 = ds2.settings['order']
            for s in o2:
                if not(s in o1):
                    flag = True
                    break
            if len(o2) != len(o1):
                flag = True
                
            assert force or not(flag), LOGGER.error("The datasets don't contain the same elements -> Merging cancelled.")

            merged_order = []
            for o in [x for x in o1+o2 if x in o1 and x in o2]:
                if o not in merged_order:
                    merged_order.append(o)

            def _arrange_batch(ds:'Dataset'):
                new_batch = {}
                sorted_indexes = None
                for b in ds.batch.keys():
                    new_b = []
                    order = []
                    for i,data in enumerate(ds.batch[b]):
                        o = ds.settings['order'][i]
                        if not(o in merged_order):
                            continue
                        new_b.append(data)
                        order.append(o)
                    if sorted_indexes is None:
                        pos = {v: i for i, v in enumerate(merged_order)}
                        sorted_indexes = sorted(range(len(order)), key=lambda i: pos.get(order[i], float('inf')))
                    new_b = np.array(new_b)[sorted_indexes].tolist()
                    new_batch[b] = new_b
                return new_batch
            
            b1 = _arrange_batch(ds1)
            b2 = _arrange_batch(ds2)
            result_ds.batch = b1
            i = len(b1.keys())
            for j in b2.keys():
                result_ds.batch[i] = b2[j]
                i += 1
            del i

            result_settings = merge_dicts(ds1.settings, ds2.settings)
            if "SIM_name" in ds1.settings:
                ds1_sname = ds1.settings["SIM_name"]
                ds2_sname = ds2.settings["SIM_name"]
                sname = ""
                if ds1_sname in ds2_sname:
                    sname = ds2_sname
                elif ds2_sname in ds1_sname:
                    sname = ds1_sname
                else:
                    sname = ds1_sname+"+"+ds2_sname
                result_settings["SIM_name"] = sname
            result_settings['order'] = merged_order
            result_ds.settings = result_settings

            result_ds.data = merge_dicts(ds1.data, ds2.data)

            return result_ds
 
        datasets = dataset if type(dataset) is list else [dataset]
        ds = self
        for d in datasets:
            assert isinstance(d, Dataset), LOGGER.error("There is an object that isn't a dataset in merge function.")
            ds = _merge(ds, d)

        if save:
            ds.save(name=name, force=True)

        if delete:
            for o_ds in datasets:
                o_ds.delete()
            self.delete()

        return ds
    
    def check_sanity(self, what_to_check:Dict={"nan":True}, remove:bool=False)->List[int]:
        corrupted_indexes = []
        check_nan = "nan" in what_to_check and what_to_check["nan"]

        wrong_channels = []
        for i in self.batch.keys():
            printProgressBar(i, len(self.batch.keys()), prefix="Sanity check")
            datas = self.get(i)
            flag = False
            for j,data in enumerate(datas):  
                if check_nan and isinstance(data, np.ndarray):
                    flag = flag or np.isnan(data).any()
                    if flag:
                        if self.settings['order'][j] not in wrong_channels:
                            wrong_channels.append(self.settings['order'][j])
                        break
            if flag:
                corrupted_indexes.append(i)

        print("")
        if len(corrupted_indexes) > 0:
            LOGGER.log(f"{len(corrupted_indexes)} elements don't correspond to standards. ({str(wrong_channels)})")
        else:
            LOGGER.log("Sanity check done without problems.")

        if remove:
            self.remove(corrupted_indexes)

        self.save_diagnostic(channels=None)
        
        return corrupted_indexes


    
    def clone(self, new_name:str)->'Dataset':
        ds = Dataset()
        ds.batch = copy.deepcopy(self.batch)
        ds.settings = copy.deepcopy(self.settings)
        ds.data = copy.deepcopy(self.data)
        ds.name = new_name
        return ds

    def downsample(self, channel_names:Union[List[str],str], target_sizes:Union[List[int], int], axis:Union[int,List[int]]=2, methods:Literal['mean','max','crop','first','nn']="mean", replace:bool=False):
        """
        Downsample the dataset and save into a new folder.
        <br />e.g: dataset.downsample(channel_names=["cospectra"], target_sizes=[128], methods=["mean"])
        Args:
                channel_names: Channel name(s) to downsample.
                target_sizes: Target size(s) along the specified axis (or axes).
                axis: Axis or list of axes to downsample (default: 2 for z-axis).
                methods: Downsampling method(s) - 'mean', 'max', 'crop', or 'nn' (nearest neighbor).
                replace: If True, remove the current dataset and save the downsampled one
        Returns:
            Dataset: the downsampled dataset
        """
        LOGGER.log(f"Downsampling ({methods}) channels: {channel_names} to sizes {target_sizes} along axis {axis}")

        if replace:
            ds = self
        else:
            ds = self.clone(self.name + "_downsampled")
            ds.save(force=True)

        channel_indexes = (
            [ds.get_element_index(c) for c in channel_names]
            if isinstance(channel_names, list)
            else [ds.get_element_index(channel_names)]
        )
        

        target_sizes = target_sizes if isinstance(target_sizes, list) else [target_sizes]
        methods = methods if isinstance(methods, list) else [methods]
        axes = axis if isinstance(axis, list) else [axis]

        for bi in ds.batch.keys():
            batch = ds.get(bi)
            for ci, channel_index in enumerate(channel_indexes):
                img = batch[channel_index]

                # Apply downsampling along one or multiple axes
                for ai, ax in enumerate(axes):
                    target_size = target_sizes[min(ci, len(target_sizes) - 1)]
                    method = methods[min(ci, len(methods) - 1)]

                    original_size = img.shape[ax]
                    factor = original_size // target_size
                    if original_size % target_size != 0:
                        LOGGER.warn(f"Warning: axis {ax} size {original_size} is not perfectly divisible by {target_size}, possible data loss.")

                    if method == "mean":
                        img = np.moveaxis(img, ax, -1)
                        new_shape = img.shape[:-1] + (target_size, factor)
                        img = img.reshape(new_shape).mean(axis=-1)
                        img = np.moveaxis(img, -1, ax)

                    elif method == "max":
                        img = np.moveaxis(img, ax, -1)
                        new_shape = img.shape[:-1] + (target_size, factor)
                        img = img.reshape(new_shape).max(axis=-1)
                        img = np.moveaxis(img, -1, ax)

                    elif method == "crop":
                        start = (original_size - target_size) // 2
                        end = start + target_size
                        slicer = [slice(None)] * img.ndim
                        slicer[ax] = slice(start, end)
                        img = img[tuple(slicer)]

                    elif method == "first":
                        start = 0
                        end = start + target_size
                        slicer = [slice(None)] * img.ndim
                        slicer[ax] = slice(start, end)
                        img = img[tuple(slicer)]
                        
                    elif method == "nn":
                        step = max(1, original_size // target_size)
                        slicer = [slice(None)] * img.ndim
                        slicer[ax] = slice(0, original_size, step)
                        img = img[tuple(slicer)]

                    else:
                        raise ValueError(f"Unsupported downsampling method: {method}")

                batch[channel_index] = img

            ds.save_batch(batch, bi)
            del batch

        return ds

    def compute_over(self,function:Callable[[np.ndarray], Any], channel:str='cdens'):
        """Apply the 'function' to the dataset on a specific 'channel'"""
        map_index = self.get_element_index(channel)
        result = []
        for i in self.batch.keys():
            result.append(function(np.array(self.get(i)[map_index]).flatten()))
        return result

    def delete(self):
        LOGGER.log(f"Deleting dataset {self.name}")
        shutil.rmtree(os.path.join(TRAINING_BATCH_FOLDER,self.name))

    #-------SAVE-------

    def save_batch(self, batch:List[np.ndarray], i:int):
        """
        Save a batch, here this means a pair of data(numpy arrays or json) not a list of pairs.
        """
        if not(os.path.exists(TRAINING_BATCH_FOLDER)):
            os.mkdir(TRAINING_BATCH_FOLDER)
        batch_path = os.path.join(TRAINING_BATCH_FOLDER,"batch_"+str(self.name).split("batch_")[-1])
        if not(os.path.exists(batch_path)):
            os.mkdir(batch_path)
        order = self.settings["order"]
        for j,o in enumerate(order):
            if isinstance(batch[j],(np.ndarray,list,tuple,float)):
                np.save(os.path.join(batch_path,str(i)+"_"+o+".npy"), batch[j])
            elif isinstance(batch[j],(dict)):
                with open(os.path.join(batch_path,str(i)+"_"+o+".json"), 'w') as file:
                    json.dump(batch[j], file, indent=4, cls=NumpyEncoder)

    def save_settings(self):
        if not(os.path.exists(TRAINING_BATCH_FOLDER)):
            os.mkdir(TRAINING_BATCH_FOLDER)
        batch_path = os.path.join(TRAINING_BATCH_FOLDER,"batch_"+str(self.name).split("batch_")[-1])
        if not(os.path.exists(batch_path)):
            os.mkdir(batch_path)
        with open(os.path.join(batch_path,'settings.json'), 'w') as file:
            json.dump(self.settings, file, indent=4, cls=NumpyEncoder)

    def save_data(self):
        if not(os.path.exists(TRAINING_BATCH_FOLDER)):
            os.mkdir(TRAINING_BATCH_FOLDER)
        batch_path = os.path.join(TRAINING_BATCH_FOLDER,"batch_"+str(self.name).split("batch_")[-1])
        if not(os.path.exists(batch_path)):
            os.mkdir(batch_path)
        with open(os.path.join(batch_path,'data.json'), 'w') as file:
            json.dump(self.data, file, indent=4, cls=NumpyEncoder)

    def save_diagnostic(self,channels:Union[str,List[str],None]='cdens')->Dict:
        """
        Save & return a diagnostic for each image in the dataset.
        Args:
            channels: Channels on which the diagnostic will be made, like 'cdens'.
        Returns:
            Dict: if more that one channel in channels: dicts will be nested into 'channel_{channel_name}'.
        """

        if channels is None:
            channels = self.settings['order']
        channels = channels if type(channels) is list else [channels]

        batch = self.get()
        map_indexes = self.get_element_index(channels)
        map_indexes = map_indexes if type(map_indexes) is list else [map_indexes]
        result_dicts = {}

        result_dicts["global"] = {}
        for i,b in enumerate(batch):
            temp_dict = {"index": i}
            for j,map_index in enumerate(map_indexes):
                data = np.array(b[map_index]).flatten()
                stats = {
                    "type": channels[j],
                    "mean": float(np.mean(data)),
                    "std_log10": float(np.std(np.log10(data))),
                    "min": float(np.min(data)),
                    "max": float(np.max(data)),
                    "median": float(np.median(data)),
                }
                temp_dict['channel_'+channels[j]] = stats
                if 'channel_'+channels[j] not in result_dicts["global"]:
                    result_dicts["global"]['channel_'+channels[j]] = {}
                if "min" not in result_dicts["global"]['channel_'+channels[j]]:
                    result_dicts["global"]['channel_'+channels[j]]["min"] = np.inf
                    result_dicts["global"]['channel_'+channels[j]]["max"] = -np.inf
                result_dicts["global"]['channel_'+channels[j]]["min"] = min(stats["min"], result_dicts["global"]['channel_'+channels[j]]["min"])
                result_dicts["global"]['channel_'+channels[j]]["max"] = max(stats["max"], result_dicts["global"]['channel_'+channels[j]]["max"])

            result_dicts[i] = temp_dict

        if not(os.path.exists(TRAINING_BATCH_FOLDER)):
            os.mkdir(TRAINING_BATCH_FOLDER)
        batch_path = os.path.join(TRAINING_BATCH_FOLDER,"batch_"+str(self.name).split("batch_")[-1])
        if not(os.path.exists(batch_path)):
            os.mkdir(batch_path)
        path = os.path.join(batch_path, "diagnostic.json")
        if os.path.exists(path):
            LOGGER.warn(f"Previous diagnostic file was removed for dataset {self.name}.")
            os.remove(path)
        with open(path, "w") as file:
            json.dump(result_dicts, file, indent=4, cls=NumpyEncoder)
        LOGGER.log(f"Diagnostic of {self.name} saved to {path}.")

        return result_dicts

    def save(self,batch:Union[List[List[np.ndarray]],None]=None, name:Union[str,None]=None, force:bool=False)->bool:

        if not(os.path.exists(TRAINING_BATCH_FOLDER)):
            os.mkdir(TRAINING_BATCH_FOLDER)

        old_name = self.name
        if name is not None:
            self.name = name
        batch_uuid = self.name

        delete_temp_ds = False
        if os.path.exists(os.path.join(TRAINING_BATCH_FOLDER,"batch_"+str(batch_uuid).split("batch_")[-1])) and force:
            LOGGER.warn(f"Dataset {batch_uuid} already exists, but force save enabled so previous batch was removed.")
            
            temp_ds = self.clone(new_name="temp_ds")
            shutil.rmtree(os.path.join(TRAINING_BATCH_FOLDER,"batch_"+str(batch_uuid).split("batch_")[-1]))
            temp_ds.name = self.name
            self = temp_ds
            delete_temp_ds = True

        while os.path.exists(os.path.join(TRAINING_BATCH_FOLDER,"batch_"+str(batch_uuid).split("batch_")[-1])):
            self.name = str(uuid.uuid4())
            LOGGER.warn(f"Dataset {batch_uuid} already exists, change to: {str(self.name)}")
            batch_uuid = self.name

        batch_path = os.path.join(TRAINING_BATCH_FOLDER,"batch_"+str(batch_uuid).split("batch_")[-1])
        os.mkdir(batch_path)

        self.save_settings()
        self.save_data()

        if batch is not None:
            for i,imgs in enumerate(batch):
                self.save_batch(imgs,i)
        else:
            batch = self.batch
            for i in self.batch.keys():
                imgs = self.get(i)
                self.save_batch(imgs,i)
                del imgs

        LOGGER.log(f"Dataset with {len(batch)} images saved.")
        self.name = old_name

        if delete_temp_ds:
            temp_ds.delete()

        return True

    #-------PLOT-------

    def plot(self, enable_slider:bool=True, element_index:int=0):

        element_index = element_index
        
        fig = plt.figure(figsize=(8,8))
        fig.suptitle("Dataset "+self.name+" "+str(element_index+1))
        
        ax1 = plt.subplot(2,2,1)
        ax2 = plt.subplot(2,2,2)
        ax3 = plt.subplot(2,2,3)
        ax4 = plt.subplot(2,2,4)

        axes_histo = None
        axes_map = None
        axes_map2 = None

        def update_element_index(val):
            nonlocal axes_histo
            nonlocal axes_map
            nonlocal axes_map2
            element_index = int(val)
            ax1.clear()
            ax1.set_visible(False)
            if axes_map is not None:
                for a in axes_map:
                    a.remove()
            axes_map = None
            if element_index > -1:
                _, axes_map = self.plot_map(ax=ax1, element_index=element_index, enable_slider=False)
            ax2.clear()
            self.plot_correlation(ax=ax2, element_index=element_index, PDF=True)#, contour_levels=[0.38,0.69, 0.95]
            ax3.clear()
            ax3.set_visible(False)
            if axes_map2 is not None:
                for a in axes_map2:
                    a.remove()
            axes_map2 = None
            if element_index > -1:
                _, axes_map2 = self.plot_map(ax=ax3, element_index=element_index, map_index= 1, enable_slider=False)
            if axes_histo is not None:
                for a in axes_histo:
                    a.remove()
            _, axes_histo = self.plot_histo(ax=ax4, element_index=element_index, enable_slider=False)
            fig.canvas.draw_idle()

        update_element_index(element_index)

        if enable_slider:
            ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
            slider = Slider(ax_slider, 'Element index', -1, len(self.batch.keys()) - 1, valinit=element_index, valfmt='%0i')
            slider.on_changed(update_element_index)  
            plt.show()

        return fig

    def plot_histo(self, ax=None, element_index=-1, map_index=1, method=np.log10, enable_slider=True, lims=[1,8]):
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        bbox = ax.get_position()
        width = bbox.width
        height = bbox.height
        left = bbox.x0
        bottom = bbox.y0
        ax.set_visible(False)

        ax_histo = fig.add_axes([left, bottom+0.1, width, height-0.1])
        histo_bins = 20

        batch = self.get(indexes=self.batch.keys()[element_index] if element_index > -1 else None)
        if element_index > -1:
            batch = [batch]

        def update_map_index(val):
            map_index = int(val)
            ax_histo.clear()
            data = np.array([method(b[map_index]) for b in batch]).flatten()
            counts, bin_edges = np.histogram(data, bins=histo_bins, density=True)
            bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
            ax_histo.hist(data, bins=histo_bins, alpha=1.0, label=self.settings["order"][map_index], density=True)
            ax_histo.plot(bin_centers, counts, color='black', linestyle='-')

            if lims is not None:
                ax_histo.set_xlim((lims[0],lims[1]))

            fig.canvas.draw_idle()
            ax_histo.set_yscale('log')
            fig.canvas.draw_idle()
            ax_histo.legend()

        update_map_index(map_index)

        if enable_slider:
            ax_slider = fig.add_axes([left, bottom, width, 0.03], zorder=10)
            slider = Slider(ax_slider, 'i', 0, len(self.settings['order']) - 1, valinit=map_index, valfmt='%0i')
            slider.on_changed(update_map_index) 

            return fig, [ax_histo, ax_slider] 

        return fig, [ax_histo]

    def plot_map(self, ax=None, element_index=0, map_index=0, enable_slider=True, show_title=True):        
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        element_index = max(0, element_index)

        bbox = ax.get_position()
        width, height = bbox.width, bbox.height
        left, bottom = bbox.x0, bbox.y0
        ax.set_visible(False)

        ax_map = fig.add_axes([left, bottom+0.12, width, height-0.12])

        def get_batch(e_idx):
            key = list(self.batch.keys())[e_idx]
            return self.get(indexes=key)

        def get_map_data(batch, m_idx):
            data = batch[m_idx]

            if not isinstance(data, np.ndarray):
                raise ValueError(f"Data at map_index {m_idx} is not a numpy array")

            if data.ndim == 2:
                return data
            elif data.ndim > 2:
                return np.sum(data, axis=-1)
            else:
                raise ValueError(f"Data at map_index {m_idx} is not plottable as a map (shape={data.shape})")

        batch = get_batch(element_index)
        data = get_map_data(batch, map_index)

        def get_extent(e_idx, m_idx):
            return None
            if "areas_explored" not in self.settings:
                return None
            return [
                val
                for ae in self.settings["areas_explored"][e_idx]
                for val in (ae - self.settings["img_size"], ae + self.settings["img_size"])
            ]

        extent = get_extent(element_index, map_index)

        artist = ax_map.imshow(data, norm=LogNorm(), cmap="jet", extent=extent)

        def plot_positions(batch):
            for d in batch:
                if isinstance(d, dict):
                    for v in d.values():
                        if 'position_x' in v and 'position_y' in v:
                            ax_map.scatter(v['position_x'], v['position_y'], marker="+", color="white")

        plot_positions(batch)

        if show_title:
            ax_map.set_title(self.settings['order'][map_index])

        if "areas_explored" in self.settings:
            ax_map.set_xlabel("[pc]")
            ax_map.set_ylabel("[pc]")

        cbar = plt.colorbar(artist, ax=ax_map, label=self.settings['order'][map_index])

        if enable_slider:
            ax_slider_map = fig.add_axes([left, bottom-0.03, width, 0.03])
            fig._slider_map = Slider(ax_slider_map, 'map', 0, len(self.settings['order']) - 1,
                                valinit=map_index, valfmt='%0.0f')

            ax_slider_elem = fig.add_axes([left, bottom-0.08, width, 0.03])
            fig._slider_elem = Slider(ax_slider_elem, 'element', 0, len(self.batch.keys()) - 1,
                                valinit=element_index, valfmt='%0.0f')

            def update(val):
                nonlocal batch

                e_idx = int(fig._slider_elem.val)
                m_idx = int(fig._slider_map.val)

                batch = get_batch(e_idx)
                data = get_map_data(batch, m_idx)

                artist.set_data(data)
                artist.set_norm(LogNorm())
                #artist.set_extent(get_extent(e_idx, m_idx))

                ax_map.clear()
                ax_map.imshow(data, norm=LogNorm(), cmap="jet", extent=get_extent(e_idx, m_idx))
                plot_positions(batch)

                if show_title:
                    ax_map.set_title(self.settings['order'][m_idx])

                if "areas_explored" in self.settings:
                    ax_map.set_xlabel("[pc]")
                    ax_map.set_ylabel("[pc]")

                cbar.update_normal(artist)
                cbar.set_label(self.settings['order'][m_idx])

                fig.canvas.draw_idle()

            fig._slider_map.on_changed(update)
            fig._slider_elem.on_changed(update)

            return fig, [ax_map, ax_slider_map, ax_slider_elem]

        return fig, [ax_map]
    def plot_correlation(self, X_i=0, Y_i=1, ax=None, bins_number=256, show_yx = False, method=np.log10, contour_levels=0, PDF=False, lines=[0,1,2], element_index=-1):
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        batch = self.get(indexes=self.batch.keys()[element_index] if element_index > -1 else None)
        if element_index > -1:
            batch = [batch]
        c1 = np.array([method(b[X_i]) for b in batch]).flatten()
        c2 = np.array([method(b[Y_i]) for b in batch]).flatten()

        ax.set_xlabel(self.settings["order"][X_i])
        ax.set_ylabel(self.settings["order"][Y_i])


        nan_indices = np.isnan(c1) | np.isnan(c2)
        good_indices = ~nan_indices
        c1= c1[good_indices]
        c2 = c2[good_indices]

        if type(contour_levels) is list or contour_levels > 1:
            hist, xedges, yedges = np.histogram2d(c1, c2, bins=(bins_number, bins_number), density=PDF)
            xcenters = 0.5 * (xedges[:-1] + xedges[1:])
            ycenters = 0.5 * (yedges[:-1] + yedges[1:])
            X, Y = np.meshgrid(xcenters, ycenters)

            if PDF and type(contour_levels) is list:
                hist_flat = hist.flatten()
                idx = np.argsort(hist_flat)[::-1]
                hist_sorted = hist_flat[idx]
                cumsum = np.cumsum(hist_sorted)
                cumsum /= cumsum[-1]

                level_values = []
                for cl in contour_levels:
                    try:
                        i = np.where(cumsum >= cl)[0][0]
                        level_values.append(hist_sorted[i])
                    except IndexError:
                        level_values.append(hist_sorted[-1])
                level_prob_map = dict(zip(level_values, contour_levels))

                contour = ax.contour(X, Y, hist.T, levels=sorted(level_values), colors="black")
                ax.clabel(contour, fmt=lambda x: f"{level_prob_map.get(x, x):.2f}", inline=True, fontsize=8)
            else:
                contour = ax.contour(X, Y, hist.T, levels=contour_levels if type(contour_levels) is list else int(contour_levels), norm=LogNorm(), colors="black")
                ax.clabel(contour, fmt=lambda x: r"$10^{{{:.0f}}}$".format(np.log10(x)) if not(PDF) else r"${:.2f}$".format(x), inline=True, fontsize=8)

        _, _, _,hist = ax.hist2d(c1, c2, bins=(bins_number,bins_number), norm=LogNorm(), density=PDF)
        
        x_min, x_max = np.min(c1), np.max(c1)
        y_min, y_max = np.min(c2), np.max(c2)
        if type(contour_levels) is list or contour_levels > 1:
            if contour.collections:
                outer_contour = contour.collections[0]
                all_paths = outer_contour.get_paths()

                if all_paths:
                    all_vertices = np.concatenate([p.vertices for p in all_paths])
                    x_min, x_max = np.min(all_vertices[:, 0]), np.max(all_vertices[:, 0])
                    y_min, y_max = np.min(all_vertices[:, 1]), np.max(all_vertices[:, 1])

                    ax.set_xlim(x_min, x_max)
                    ax.set_ylim(y_min, y_max)

        if type(lines) is list and len(lines) > 0:
            plot_lines(None, None, ax, lines=lines, x_max=x_max, x_min=x_min, y_max=y_max, y_min=y_min)

        
        #plt.colorbar(hist, ax=ax, label="PDF" if PDF else "counts")
        #ax.grid(True)
        ax.set_axisbelow(True)
        
        
        if show_yx:
            yx = np.linspace(np.min([c1.min(), c2.min()]), np.max([c1.max(), c2.max()]), 10)
            plt.plot(yx,yx,linestyle="--",color="red",label=r"$y=x$")

        return fig, ax