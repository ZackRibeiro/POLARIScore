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
from copy import deepcopy
import heapq

KEY_H5_TO_SIM = {
    "RHO": "scalars/density",
    "TEMP": "scalars/temperature",
    "VEL": "vectors/velocity"
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

    valid_mask = arr != 0

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


def amr_leaf_to_datacube(points,sizes,values,bbox,res,dtype=np.float64,log:bool=True, out_file:Optional[str]="cached_amr_cube.mem"):
    if isinstance(res, int):
        res = (res, res, res)

    nx, ny, nz = res

    cube = np.zeros((nx, ny, nz), dtype=dtype) if out_file is None else np.memmap(out_file,dtype=dtype,mode="w+",shape=(nx, ny, nz),order="C")

    if out_file is not None:
        cube[:] = 0
        cube.flush()

    bbox = np.asarray(bbox, dtype=dtype)

    xmin, ymin, zmin = bbox[:,0]
    xmax, ymax, zmax = bbox[:,1]

    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny
    dz = (zmax - zmin) / nz

    indexes_to_keep = []

    for i, (p, h) in enumerate(zip(points, sizes)):
        half = h / 2
        x0 = p[2] - half
        x1 = p[2] + half
        y0 = p[1] - half
        y1 = p[1] + half
        z0 = p[0] - half
        z1 = p[0] + half

        inside = ((x1 >= xmin) and (x0 <= xmax) and(y1 >= ymin) and (y0 <= ymax) and(z1 >= zmin) and (z0 <= zmax))

        if inside:
            indexes_to_keep.append(i)

    sorted_indexes = np.argsort(sizes[indexes_to_keep])[::-1]
    kept_indexes = np.asarray(indexes_to_keep)[sorted_indexes]

    for i, idx in enumerate(kept_indexes):

        p = points[idx]
        h = sizes[idx]
        val = values[idx]

        if log and (i + 1) % int(len(values)/1e3) == 0:            
            printProgressBar(i, len(values), "AMR Leaves to Datacube", length=30)

        half = h / 2

        x0 = p[2] - half
        x1 = p[2] + half

        y0 = p[1] - half
        y1 = p[1] + half

        z0 = p[0] - half
        z1 = p[0] + half

        ix0 = int(np.floor((x0 - xmin) / dx))
        ix1 = int(np.ceil((x1 - xmin) / dx))


        iy0 = int(np.floor((y0 - ymin) / dy))
        iy1 = int(np.ceil((y1 - ymin) / dy))

        iz0 = int(np.floor((z0 - zmin) / dz))
        iz1 = int(np.ceil((z1 - zmin) / dz))

        if iz1==iz0:
            iz1+=1
        if ix1==ix0:
            ix1+=1
        if iy1==iy0:
            iy1+=1

        ix0 = max(0, min(nx, ix0))
        iy0 = max(0, min(ny, iy0))
        iz0 = max(0, min(nz, iz0))

        ix1 = max(0, min(nx, ix1))
        iy1 = max(0, min(ny, iy1))
        iz1 = max(0, min(nz, iz1))

        cube[ix0:ix1, iy0:iy1, iz0:iz1] = val / (h**3)

    return cube

class Simulation_AMR(Simulation_DC):
    
    def __init__(self, name, global_size, init=True, init_datacubes:bool=True):

        super().__init__(name, global_size, init=False)
        self.yt:yt.frontends.stream.data_structures.StreamParticlesDataset = None
        if init:
            self.init(init_datacubes=init_datacubes)

    
    def init(self, init_datacubes:bool=True, init_yt:bool=True):
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
            self.load_h5(key="VEL", path="datatree_vel", unit=1, str_unit="cm/s")
            #self.load_h5(key="GRAV", path="datatree_grav", unit=1, str_unit="cm**2/s**2")
            #self.load_h5(key="B", path="datatree_mag", unit=1, str_unit="G")
            #self.load_h5(key="M", path="datatree_mom", unit=1, str_unit="g/cm**2/s")

        self.bbox = self.bbox*self.global_size
        if init_yt:
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
            LOGGER.log(f"Adding {k} to particle fields.")
            self.yt.add_deposited_particle_field(("all", "particle_"+k[2:]), "sum")
        return self.yt

    def load_h5(self, key:str="RHO", path:str="datatree", unit=1, str_unit="", volume_weighted:bool=True):
        path = os.path.join(self.folder, path)
        if not(".h5" in path):
            path += ".h5"
        if not(os.path.exists(path)):
            LOGGER.error(f"Data {key} not loaded in simulation {self.name} -> File not found")
            return False
        key_to_verify = key
        with h5py.File(path, "r") as f:
            points = f['points'][:]
            if "points" in self.data:
                points = points*PC_TO_CM*self.global_size
                if len(points) != len(self.data["points"]):
                    LOGGER.error(f"Data {key} not loaded in simulation {self.name} -> The points are not the same that the previous file loaded.")
                    return False
                if any([points[i][0] != self.data["points"][i][0] for i in range(len(points))]):
                    LOGGER.error(f"Data {key} not loaded in simulation {self.name} -> The points are not the same that the previous file loaded.")
                    return False
            else:
                x0,x1 = np.min(points[:,2]),np.max(points[:,2])
                y0,y1 = np.min(points[:,1]),np.max(points[:,1])
                z0,z1 = np.min(points[:,0]),np.max(points[:,0])

                self.data["points"] = points*PC_TO_CM*self.global_size
                
                eps = 1e-10
                if x1-x0 != y1-y0 or x1-x0 != z1-z0 or z1-z0 != y1-y0:
                    LOGGER.warn(f"The simulation is not a cube, self.size is the maximum length. ({x0},{x1};{y0},{y1};{z0},{z1}) -> Cube forced") 
                    x0,y0,z0 = np.min([x0,y0,z0]),np.min([x0,y0,z0]),np.min([x0,y0,z0])
                    x1,y1,z1 = np.max([x1,y1,z1]),np.max([x1,y1,z1]),np.max([x1,y1,z1])
                self.bbox = np.array([[x0-eps,x1+eps],[y0-eps,y1+eps],[z0-eps,z1+eps]])
                self.size = np.max([x1-x0,y1-y0,z1-z0])*self.global_size
                self.relative_size = self.size/self.global_size
            if "p_SIZE" not in self.data:
                LOGGER.log(f"AMR Simulation {self.name} resolution goes from {int(1/np.max(f["scalars/size"][:]))} to {int(1/np.min(f["scalars/size"][:]))}")
                self.data["p_SIZE"] = (f["scalars/size"][:]*self.global_size*PC_TO_CM, "cm")
                self.data["p_VOLUME"] = ((f["scalars/size"][:]*self.global_size*PC_TO_CM)**3, "cm**3")
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
                if "vectors" in query_key:
                    key_to_verify = []
                    for i in range(f[query_key][:].shape[-1]):
                        #v_key = key[0]+"X"+str(f[query_key][:].shape[-1]-i)
                        v_key = key[0]+"X"+str(i+1)
                        self.data["p_"+v_key] = (f[query_key][:,i]*factor, str_unit)
                        LOGGER.log(f"{key}: vmin={np.min(f[query_key][:,i])}, vmax={np.max(f[query_key][:,i])}")
                        key_to_verify.append(v_key)
                else:
                    self.data["p_"+key] = (f[query_key][:]*factor, str_unit)
                    key_to_verify = key
        if not(isinstance(key_to_verify, list)):
            key_to_verify = [key_to_verify]
        for k in key_to_verify:
            if self.data["p_"+k] is None:
                LOGGER.warn(f"Data {k} not loaded in simulation {self.name} -> file empty")
                return False
        return True
    
    def init_datacubes(self, res=AMR_BASE_RESOLUTION, force:bool=True, filling_method=amr_leaf_to_datacube, keys:Optional[List[str]]=None, use_mem_cache:bool=False):
        keys = list(self.data.keys()) if keys is None else keys
        for k in keys:
            k = k.lower()
            if not(isinstance(k, str)):
                continue
            if k[0]=="p" and k[1]=="_":
                self.to_datacube(key=k, res=res, store=True, cache=True, force=force, filling_method=filling_method, out_file="cached_amr_cube.mem" if use_mem_cache else None)
        self.nres = res
        self.cell_size = self.size/self.nres*PC_TO_CM
        
    def to_datacube(self, key:Union[str,List[str]], filling_method:Optional[Callable]=amr_leaf_to_datacube, res:Union[List[int],int]=128, smoothing:float=0, bbox:Optional[List[float]]=None,
                    store:bool=False, force:bool=False, log:bool=True, cache:bool=False, dtype=np.float64, out_file:Optional[str]="cached_amr_cube.mem"):
        
        if isinstance(out_file, bool):
            if out_file:
                out_file = "cached_amr_cube.mem"
            else:
                out_file = None
        if isinstance(key, list):
            result_list = []
            for k in key:
                result_list.append(self.to_datacube(
                    k, filling_method=filling_method, res=res, smoothing=smoothing, bbox=bbox, force=True, log=log, dtype=dtype, out_file=None
                ))
            return result_list

        bbox_was_none = bbox is None
        if bbox is not None:
            force=True
        init_bbox = deepcopy(list(self.bbox)) if bbox is None else deepcopy(list(bbox))        
        bbox = deepcopy(init_bbox)    
        if len(bbox) < 3:
            for _ in range(3-len(bbox)):
                bbox.append(None)
        for i in range(len(bbox)):
            if bbox[i] is None:
                bbox[i] = [*self.bbox[i]]
        bbox = np.array(bbox)
        bbox *= PC_TO_CM
        #t = [*bbox[1]]
        #bbox[1] = bbox[0]
        #bbox[0] = t

        if isinstance(res, (int, float)):
            res = [int(res),int(res),int(res)]
        else:
            pass
            #tr = res[1]
            #res[1] = res[0]
            #res[0] = tr

        if (key[0] == "p" and key[1] == "_"):
            query_key = "p_"+key[2:].upper()
            key = key[2:]
        else:
            query_key = "p_"+key.upper()
        if log:
            LOGGER.log(f"Making a datacube of {query_key} with a resolution of {res}")

        if cache and bbox_was_none:
            path_key = os.path.join(self.folder, "datacube_"+key+"_"+str(res[0])+"_"+filling_method.__name__+".npy")
            if os.path.exists(path_key):
                LOGGER.log(f"Datacube {key} loaded using cache.")
                if store:
                    self.data[key.upper()] = np.load(path_key, mmap_mode="r+")
                return self.data[key.upper()]

        assert query_key in self.data, LOGGER.error(f"Can't construct datacube from {query_key} -> No such key loaded in data.")
        #bbox[0,:] = self.global_size*PC_TO_CM-bbox[0,:]
        #bbox[0,:] = np.sort(bbox[0,:])

        if key in self.data and not(force) and self.data[key].shape[0] == res[0]:
            return self.data[key]
        if filling_method.__name__ == amr_leaf_to_datacube.__name__:
            points = self.data["points"]
            sizes = self.data["p_SIZE"][0]
            values = self.data[query_key][0]
            datacube = amr_leaf_to_datacube(points=points,sizes=sizes,values=values,bbox=bbox,res=res, dtype=dtype, out_file=out_file)
            #datacube = fill_zeros_nearest(datacube)
        else:
            assert self.yt is not None, LOGGER.error(f"Mising yt streaming dataset.")
            
            if key.upper() != "SIZE":
                if 'VOLUME' in self.data and self.data['VOLUME'].shape[0] == res[0] and bbox_was_none:
                    volume_tensor = self.data['VOLUME']
                else:
                    if key.upper() != "VOLUME":
                        volume_tensor = self.to_datacube("VOLUME", filling_method=filling_method, res=res, smoothing=smoothing, bbox=init_bbox, store=store, force=False, log=log, cache=cache, dtype=dtype, out_file="cached_amr_cube_volume.mem" if out_file is not None else None)

            cg = self.yt.arbitrary_grid(left_edge=bbox[:,0],right_edge=bbox[:,1],dims=res)
            datacube = cg["deposit", "all_sum_"+key.lower()].to_ndarray()

            if key.upper() != "SIZE" and key.upper() != "VOLUME":
                datacube = np.divide(datacube,volume_tensor,out=datacube,
                    where=(volume_tensor > 0) & (datacube != 0)
                )
        if filling_method.__name__ != amr_leaf_to_datacube.__name__ and filling_method is not None:
            datacube = filling_method(datacube)
        if smoothing > 0:
            datacube = gaussian_filter(datacube, sigma=smoothing)

        if cache and bbox_was_none:
            path_key = os.path.join(self.folder, "datacube_"+key+"_"+str(res[0])+"_"+filling_method.__name__+".npy")
            np.save(path_key, datacube)
            LOGGER.log(f"Datacube {key} saved at {path_key}.")
            del datacube
            datacube = np.load(path_key, mmap_mode="r+")
        if store:
            self.data[key.upper()] = datacube

        return datacube
    
    def get_region_datacube(self, key, axis, bbox, res, use_datacube:bool=False):
        if use_datacube:
            return super().get_region_datacube(key, axis, bbox, res)
        if axis == -1:
            insert_axis = len(bbox)
        else:
            insert_axis = axis

        grid_axis = axis
        if axis == 0:
            grid_axis = 1
        elif axis == 1:
            grid_axis = 0

        is_a_cube=False
        if len(bbox) == 4:
            bbox=[[bbox[0], bbox[1]], [bbox[2], bbox[3]]]
            bbox.insert(insert_axis, None)
            rres = [res,res]
            rres.insert(insert_axis, 1024)
        elif len(bbox) == 6:
            bbox=[[bbox[0], bbox[1]], [bbox[2], bbox[3]], [bbox[4], bbox[5]]]
            rres = [res,res,res]
            is_a_cube = True

        #data = self.to_datacube(key.upper(), res=rres, bbox=bbox,
        #                        filling_method=lambda t: fill_zeros_slice(t, method=fill_zeros_nearest, axis=grid_axis), smoothing=0.5, force=True, log=False)
        data = self.to_datacube(key.upper(), res=rres, bbox=bbox,
                                filling_method=amr_leaf_to_datacube, smoothing=0., force=True, log=False, out_file=None)
        
        if not(is_a_cube):
            print(data.shape)
            data = np.moveaxis(data, insert_axis, -1)
            print(data.shape)
        data += 1e-1
        return data
    
    def _get_core_volume(self, core:Dict, use_datacube:bool=False, res=256, amr_method=amr_leaf_to_datacube, box_size=8.):
        if use_datacube:
            return super()._get_core_volume(core)
        
        pos = np.array([core['pos_x'],core['pos_y'],core['pos_z']])
        x0, x1 = pos-box_size/2, pos+box_size/2
        bbox = np.array([x0,x1]).T
        rho = self.get_region_datacube("RHO",res=res+1, bbox=bbox.flatten()[:4], axis=0)
        return rho, 0.


        data = self.to_datacube(["RHO","VX1","VX2","VX3"],
                                               res=[res+1,res+1,res+1], bbox=bbox, filling_method=amr_method, force=True, out_file=None,
                                               log=False)
        rho = data[0]
        vx = data[1]
        vy = data[2]
        vz = data[3]

        shape = rho.shape

        pos = np.floor(res//2, res//2, res//2)
        dx = box_size/res
        peak = tuple(pos)

        neighbors = [(1, 0, 0), (-1, 0, 0),(0, 1, 0), (0, -1, 0),(0, 0, 1), (0, 0, -1)]
        visited = {peak}
        heap = [(-rho[peak], peak)]

        region = []

        best_alpha = np.inf
        best_region = None

        peak_xyz = np.array(peak)

        while heap:

            _, cell = heapq.heappop(heap)
            region.append(cell)

            idx = np.array(region)

            x, y, z = idx.T

            m = rho[x, y, z]*(1.673e-24 *1.4) * dx**3
            M = np.sum(m)

            if M <= 0:
                continue

            pos = idx.astype(float)
            r2 = np.sum((pos - peak_xyz[None, :])**2,axis=1) * dx**2
            R = np.sqrt(np.sum(m * r2) / M)

            vx_reg = vx[x, y, z]
            vy_reg = vy[x, y, z]
            vz_reg = vz[x, y, z]

            vx_cm = np.sum(m * vx_reg) / M
            vy_cm = np.sum(m * vy_reg) / M
            vz_cm = np.sum(m * vz_reg) / M

            sigma2 = np.sum(m * ((vx_reg - vx_cm)**2 +(vy_reg - vy_cm)**2 +(vz_reg - vz_cm)**2)) / (3 * M)

            alpha = 5.0 * sigma2 * R / (6.67430e-8 * M)
            if len(region) > 5:
                if alpha < best_alpha:
                    best_alpha = alpha
                    best_region = region.copy()
                else:
                    break

            for dx_, dy_, dz_ in neighbors:
                nx = cell[0] + dx_
                ny = cell[1] + dy_
                nz = cell[2] + dz_

                if not (0 <= nx < shape[0] and 0 <= ny < shape[1] and 0 <= nz < shape[2]):
                    continue

                p = (nx, ny, nz)

                if p in visited:
                    continue

                visited.add(p)

                heapq.heappush(heap,(-rho[p], p))

        return np.array(best_region), best_alpha

    