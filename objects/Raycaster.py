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
    def __init__(self, simulation, method:Callable[[Any,Tuple[int,int,int],Tuple[int,int,int],Dict],Dict], starting_position:Tuple[int,int,int], axis:int=0, stop_pos:Tuple[int,int,int]=[0,0,0]):
        self.simulation = simulation
        self.axis:Tuple[int,int,int] = axis
        self.method:Callable[[Any,Tuple[int,int,int],Tuple[int,int,int],Dict],Dict] = method
        self.ray_position:Tuple[int,int,int] = starting_position
        self.start_pos:Tuple[int,int,int] = starting_position
        self.position:Tuple[int,int,int] = self.start_pos
        self.stop_pos:Tuple[int,int,int] = stop_pos
        self.ray_direction:Tuple[int,int,int] = np.array(self._get_direction())
        self.ray_direction = self.ray_direction/np.linalg.norm(self.ray_direction)
        self.dimensions:Tuple[int,int,int] = self.simulation.data['RHO'].shape
        self.result:Dict = None

    def start(self)->Dict:
        """
        Starts casting the ray through the simulation, applying the method to each cell the ray crosses.
        """
        current_pos = np.array(list(self.position))
        #previous step result
        result = {}
        steps = 0
        while self._in_bounds(current_pos) and steps < self.simulation.nres*2:
            result = self.method(self.simulation,current_pos,self.ray_direction,result)
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
        return all(self.stop_pos[i] <= position[i] <= self.start_pos[i] for i in range(len(position))) and all(0 <= position[i] < self.dimensions[i] for i in range(len(position)))
    
def ray_mapping(simulation,method:Callable[[Any,Tuple[int,int,int],Tuple[int,int,int],Dict],Dict],axis:int,region:Tuple[int,int,int,int]=[0,-1,0,-1]):
    """Launch raycasters on a cube face."""
    results = []
    irange = range(region[0],region[1] if region[1] > 0 else simulation.nres)
    jrange = range(region[2],region[3] if region[3] > 0 else simulation.nres)
    for ir,i in enumerate(irange):
        results.append([])
        for jr,j in enumerate(jrange):
            printProgressBar(len(jrange)*ir+jr,len(irange)*len(jrange),prefix="Ray Mapping", length=20)

            if axis == 0: 
                starting_pos = [simulation.nres-1, i, j]
            elif axis == 1: 
                starting_pos = [i, simulation.nres-1, j]
            elif axis == 2:
                starting_pos = [i, j, simulation.nres-1]
            else:
                raise ValueError("Axis must be 0 (X), 1 (Y), or 2 (Z).")
            starting_pos = np.array(starting_pos)
            
            raycaster = Raycaster(simulation, method, starting_pos, axis)
            result = raycaster.start()
            results[ir].append(result)
    print("")
    return results
    