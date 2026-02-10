from POLARIScore.utils import *
from POLARIScore.config import *
import numpy as np
from typing import Dict, Tuple, Callable, Any

class Raycaster():
    """
    A raycaster travel through the simulation in a given direction/face and cast for each cell traveled a method and propagate the result to next cells. 
    This allows raytracing methods like simulation of CO spectrum. (work for Datacube simulation)

    TODO: get all the data instead of needing simulation data cube
    """
    def __init__(self, cube_dimension, method:Callable[[Any,Tuple[int,int,int],Tuple[int,int,int],Dict],Dict], starting_position:Tuple[int,int,int], axis:int=0, stop_pos:Tuple[int,int,int]=[0,0,0],
                 extra_args=None):
        self.axis:Tuple[int,int,int] = axis
        self.method:Callable[[Any,Tuple[int,int,int],Tuple[int,int,int],Dict],Dict] = method
        self.ray_position:Tuple[int,int,int] = starting_position
        self.start_pos:Tuple[int,int,int] = starting_position
        self.position:Tuple[int,int,int] = self.start_pos
        self.stop_pos:Tuple[int,int,int] = stop_pos
        self.ray_direction:Tuple[int,int,int] = self._get_direction()
        self.ray_direction = self.ray_direction/np.linalg.norm(self.ray_direction)
        self.dimension:Tuple[int,int,int] = cube_dimension
        self.result:Dict = None
        self.extra_args = extra_args

    def start(self)->Dict:
        """
        Starts casting the ray through the simulation, applying the method to each cell the ray crosses.
        """
        current_pos = np.array(list(self.position))
        #previous step result
        result = {}
        steps = 0
        while self._in_bounds(current_pos) and steps < self.dimension*2:
            result = self.method(self.dimension,current_pos,self.ray_direction,result,self.extra_args)
            current_pos = (current_pos + self.ray_direction).astype(int)
            self.position = current_pos
            steps += 1
        self.result = result
        return result

    def _get_direction(self)->np.ndarray:
        direction = [0, 0, 0]
        direction[self.axis] = -1
        return np.array(direction)
    
    def _in_bounds(self, position:Tuple[int,int,int])->bool:
        """Returns if the position is still in the cube."""
        return all(self.stop_pos[i] <= position[i] <= self.start_pos[i] for i in range(len(position))) and all(0 <= position[i] < self.dimension for i in range(len(position)))

def _ray_worker(args):
    cube_dimension, method, axis, starting_pos, i, j, extra_args = args
    raycaster = Raycaster(cube_dimension, method, starting_pos, axis, extra_args=extra_args)
    result = raycaster.start()

    return i, j, result

import multiprocessing as mp
def ray_mapping(simulation,method:Callable[[Any,Tuple[int,int,int],Tuple[int,int,int],Dict],Dict],axis:int,region:Tuple[int,int,int,int]=[0,-1,0,-1], 
                used_cpu:float=1., extra_args=None):
    """Launch raycasters on a cube face."""
    if used_cpu > 0.:
        used_cpu = max(used_cpu,1.)
    nproc=max(int(np.ceil(mp.cpu_count()*used_cpu)),1)

    cube_dimension = simulation.nres
    irange = range(region[0],region[1] if region[1] > 0 else cube_dimension)
    jrange = range(region[2],region[3] if region[3] > 0 else cube_dimension)

    tasks = []
    data_keys = ['RHO','VX1','VX2','VX3','TEMP']
    direction=np.eye(3)[axis]*-1
    ite = 0
    for i in irange:
        for j in jrange:
            ite += 1
            printProgressBar(ite, total=len(irange)*len(jrange), length=30, prefix="Init ray tasks")
            if axis == 0:
                starting_pos = np.array([cube_dimension - 1, i, j])
            elif axis == 1:
                starting_pos = np.array([i, cube_dimension - 1, j])
            elif axis == 2:
                starting_pos = np.array([i, j, cube_dimension - 1])
            else:
                raise ValueError("Axis must be 0, 1, or 2.")
            data_args = {key: simulation.project_data(key,i,j,axis=axis) for key in data_keys}
            tasks.append((cube_dimension, method, axis, starting_pos, i, j, {**extra_args, **data_args}))
    results = {(i, j): None for i in irange for j in jrange}

    with mp.Pool(processes=nproc or mp.cpu_count()) as pool:
        for k, (i, j, res) in enumerate(pool.imap_unordered(_ray_worker, tasks)):
            results[(i, j)] = res
            printProgressBar(k + 1,len(tasks),prefix="Ray Mapping",length=20,)

    output = []
    for i in irange:
        row = []
        for j in jrange:
            row.append(results[(i, j)])
        output.append(row)

    print("")
    return output
    