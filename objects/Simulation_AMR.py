import sys
import os
import yt
import h5py
import yt.frontends
import yt.frontends.stream
import yt.frontends.stream.data_structures
from POLARIScore.utils.utils import *
from POLARIScore.config import *
import json
from astropy.constants import m_p
from astropy import units as u
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from POLARIScore.utils.physics_utils import PC_TO_CM
from POLARIScore.objects.Simulation_DC import Simulation_DC
from scipy.spatial import cKDTree
from scipy import ndimage
import glob
from typing import *

KEY_H5_TO_SIM = {
    "RHO": "scalars/density"
}

def fill_zeros_nearest(arr):
    mask = arr != 0
    idx = ndimage.distance_transform_edt(~mask, return_indices=True)
    out = arr[*idx[1]]
    return out

def fill_zeros_slice(arr, method=fill_zeros_nearest, axis=-1):
    out = np.empty_like(arr)
    moved = np.moveaxis(arr, axis, 0)
    moved_out = np.empty_like(moved)

    for i in range(moved.shape[0]):
        moved_out[i] = method(moved[i])

    out = np.moveaxis(moved_out, 0, axis)

    return out


def fill_zeros_idw(arr, k=8, power=2):
    arr = np.asarray(arr)

    valid_mask = arr > 0

    if valid_mask.all():
        return arr.copy()

    if not np.any(valid_mask):
        return arr.copy()

    valid_coords = np.array(np.nonzero(valid_mask)).T
    valid_values = arr[valid_mask]

    empty_mask = ~valid_mask
    empty_coords = np.array(np.nonzero(empty_mask)).T

    k = min(k, len(valid_coords))

    tree = cKDTree(valid_coords)

    distances, indices = tree.query(empty_coords, k=k)

    if k == 1:
        distances = distances[:, None]
        indices = indices[:, None]

    distances = np.maximum(distances, 1e-12)

    weights = 1.0 / distances**power
    weights /= weights.sum(axis=1, keepdims=True)

    interpolated = np.sum(
        weights * valid_values[indices],
        axis=1
    )

    out = arr.copy()
    out[empty_mask] = interpolated

    return out

class Simulation_AMR(Simulation_DC):
    
    def __init__(self, name, global_size, init=True, init_datacubes:bool=True):

        super().__init__(name, global_size, init=False)
        self.yt:yt.frontends.stream.data_structures.StreamParticlesDataset = None
        if init:
            self.init(init_datacubes=init_datacubes)

    
    def init(self, init_datacubes:bool=True):
        LOGGER.log(f"Initing simulation {self.name}")
        paths = glob.glob(os.path.join(self.folder,"*.h5"))

        if os.path.exists(os.path.join(self.folder,"processing_config.json")):
            with open(os.path.join(self.folder,"processing_config.json"), "r") as file:
                self.header = json.load(file)

        assert len(paths) > 0, LOGGER.error(f"Can't init AMR simulation {self.name} because no h5 file was found.")
        if len(paths) == 1:
            assert self.load_h5(key="RHO", path=paths[0], unit=(1/(1.673e-24 *1.4)))
        else:
            assert self.load_h5(key="RHO", path="datatree", unit=(1/(1.673e-24 *1.4)))
            self.load_h5(key="TEMP", path="datatree_temp", unit=1, str_unit="K")
            self.load_h5(key="VX1", path="datatree_velx", unit=1e5)
            self.load_h5(key="VX2", path="datatree_vely", unit=1e5)
            self.load_h5(key="VX3", path="datatree_velz", unit=1e5)

        self.bbox = self.bbox*self.global_size
        self.init_yt()
        self.nres = AMR_BASE_RESOLUTION
        self.cell_size = self.size/self.nres*PC_TO_CM
        if init_datacubes:
            self.init_datacubes()

    def init_yt(self):
        LOGGER.log("Initing stream particles dataset for amr simulation.")
        if self.yt is not None:
            LOGGER.warn("YT Dataset was not null, the previous dataset was removed.") 
            self.yt = None
        keys = self.data.keys()
        p_keys = []
        for k in keys:
            k = k.lower()
            if not(isinstance(k, str)):
                continue
            if k[0]=="p" and k[1]=="_":
                p_keys.append(k)
        assert len(p_keys) > 0, LOGGER.error("Can't initialize YT Dataset -> missing data in self.data")
        assert "points" in self.data, LOGGER.error("Can't initialize YT Dataset -> missing points/positions in self.data")
        data = {
            "particle_position_x": (self.data["points"][:,2], "cm"),
            "particle_position_y": (self.data["points"][:,1], "cm"),
            "particle_position_z": (self.data["points"][:,0], "cm"),
        }
        for k in p_keys:
            data["particle_"+k[2:]] = self.data["p_"+k[2:].upper()]
        self.yt = yt.load_particles(data, bbox=self.bbox*PC_TO_CM, length_unit=1)
        for k in p_keys:
            self.yt.add_deposited_particle_field(("all", "particle_"+k[2:]), "sum")
        return self.yt

    def load_h5(self, key:str="RHO", path:str="datatree", unit=1, str_unit="", volume_weighted:bool=True):
        path = os.path.join(self.folder, path)
        if not(".h5" in path):
            path += ".h5"
        if not(os.path.exists(path)):
            LOGGER.error(f"Data {key} not loaded in simulation {self.name} -> File not found")
            return False
        with h5py.File(path, "r") as f:
            points = f['points'][:]
            if "points" in self.data:
                points = points*PC_TO_CM*self.global_size
                if len(points) != len(self.data["points"]):
                    LOGGER.error(f"Data {key} not loaded in simulation {self.name} -> The points are not the same that the previous file loaded.")
                    return False
                if any([points[0][i] != self.data["points"][0][i] for i in range(len(points))]):
                    LOGGER.error(f"Data {key} not loaded in simulation {self.name} -> The points are not the same that the previous file loaded.")
                    return False
            else:
                x0,x1 = np.min(points[:,2]),np.max(points[:,2])
                y0,y1 = np.min(points[:,1]),np.max(points[:,1])
                z0,z1 = np.min(points[:,0]),np.max(points[:,0])

                self.data["points"] = points*PC_TO_CM*self.global_size
                
                eps = 1e-10
                self.bbox = np.array([[x0-eps,x1+eps],[y0-eps,y1+eps],[z0-eps,z1+eps]])
                if x1-x0 != y1-y0 or x1-x0 != z1-z0 or z1-z0 != y1-y0:
                   LOGGER.warn("The simulation is not a cube, self.size is the maximum length.") 
                self.size = np.max([x1-x0,y1-y0,z1-z0])*self.global_size
                self.relative_size = self.size/self.global_size
            if "p_SIZE" not in self.data:
                LOGGER.log(f"AMR Simulation {self.name} resolution goes from {int(1/np.max(f["scalars/size"][:]))} to {int(1/np.min(f["scalars/size"][:]))}")
                self.data["p_SIZE"] = (f["scalars/size"][:]*self.size*PC_TO_CM, "cm")
                self.data["p_VOLUME"] = ((f["scalars/size"][:]*self.size*PC_TO_CM)**3, "cm**3")
            query_key = "scalars/density"
            if key in KEY_H5_TO_SIM:
                query_key = KEY_H5_TO_SIM[key]
                factor = unit
                if str_unit:
                    str_unit += "*cm**3"
                else:
                    str_unit = "cm**3"
                if volume_weighted:
                    assert "p_SIZE" in self.data, LOGGER.error("Can't volume weight the particle volumes if there is no size data.")
                    factor *= (self.data["p_SIZE"][0])**3
                self.data["p_"+key] = (f[query_key][:]*factor, str_unit)
        if self.data["p_"+key] is None:
            LOGGER.warn(f"Data {key} not loaded in simulation {self.name} -> file empty")
            return False
        return True
    
    def init_datacubes(self, res=AMR_BASE_RESOLUTION, force:bool=True):
        keys = list(self.data.keys())
        for k in keys:
            k = k.lower()
            if not(isinstance(k, str)):
                continue
            if k[0]=="p" and k[1]=="_":
                self.to_datacube(key=k, res=res, store=True, force=force)
        self.nres = AMR_BASE_RESOLUTION
        self.cell_size = self.size/self.nres*PC_TO_CM
        
    def to_datacube(self, key, filling_method:Optional[Callable]=fill_zeros_nearest, res:Union[List[int],int]=128, smoothing:float=0, bbox:Optional[List[float]]=None, store:bool=False, force:bool=False):
        bbox_was_none = bbox is None
        if bbox is not None:
            force=True
        bbox = self.bbox.copy() if bbox is None else bbox            
        if len(bbox) < 3:
            for _ in range(3-len(bbox)):
                bbox.append(None)
        for i in range(len(bbox)):
            if bbox[i] is None:
                bbox[i] = [*self.bbox[i]]
        bbox = np.array(bbox)
        bbox *= PC_TO_CM
        t = [*bbox[1]]
        bbox[1] = bbox[0]
        bbox[0] = t
        if (key[0] == "p" and key[1] == "_"):
            query_key = "p_"+key[2:].upper()
            key = key[2:]
        else:
            query_key = "p_"+key.upper()
        LOGGER.log(f"Making a datacube of {query_key} with a resolution of {res}")

        assert query_key in self.data, LOGGER.error(f"Can't construct datacube from {query_key} -> No such key loaded in data.")
        assert self.yt is not None, LOGGER.error(f"Mising yt streaming dataset.")
        if key in self.data and not(force) and self.data[key].shape[0] == res:
            return self.data[key]
        
        if isinstance(res, (int, float)):
            res = [int(res),int(res),int(res)]

        bbox[0,:] = self.global_size*PC_TO_CM-bbox[0,:] #dont know why but this is needed (y flipped and matplotlib plot y,x)
        bbox[0,:] = np.sort(bbox[0,:])

        cg = self.yt.arbitrary_grid(
            left_edge=bbox[:,0],
            right_edge=bbox[:,1],
            dims=(res[0], res[1], res[2])
        )

        if 'VOLUME' in self.data and self.data['VOLUME'].shape[0] == res and bbox_was_none:
            volume_tensor = self.data['VOLUME']
        else:
            volume_tensor = cg["deposit", "all_sum_volume"].to_ndarray()
            if (bbox_was_none):
                self.data['VOLUME'] = volume_tensor

        if key.upper() == "VOLUME":
            return volume_tensor

        datacube = cg["deposit", "all_sum_"+key.lower()].to_ndarray()

        if key.upper() != "SIZE":
            datacube = np.divide(datacube,volume_tensor,out=np.zeros_like(datacube),
                where=(volume_tensor > 0) & (datacube > 0)
            )
        if filling_method is not None:
            datacube = filling_method(datacube)
        if smoothing > 0:
            datacube = gaussian_filter(datacube, sigma=smoothing)

        if store:
            self.data[key.upper()] = datacube

        return datacube
    