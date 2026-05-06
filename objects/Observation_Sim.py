from POLARIScore.config import LOGGER
from POLARIScore.objects.Simulation_DC import Simulation_DC
from POLARIScore.objects.Observation import Observation
from POLARIScore.objects.DenseCore import DenseCore
from POLARIScore.utils.utils import compute_column_density, convert_pc_to_index

import numpy as np

class Observation_Sim(Observation):
    def __init__(self,simulation:Simulation_DC, axis=0):
        super().__init__(simulation.name, None, 0., 'pc_map')
        LOGGER.log(f"Creating observation based on simulation {simulation.name}")
        LOGGER.warn("Certain methods of the original Observation class are not (yet) supported by this subclass.")

        self.axis = axis
        self.simulation = simulation
        self.folder = simulation.folder
        self.data = compute_column_density(simulation.data['RHO'], simulation.cell_size, axis=axis)
        self.get_cores()

    def skycoord_to_pixel(self,coords, _=None):
        x_pc, y_pc = coords
        x_idx = convert_pc_to_index(x_pc, len(self.data), self.simulation.size, start=self.simulation.axis[0][0], clip=False)
        y_idx = convert_pc_to_index(y_pc, len(self.data), self.simulation.size, start=self.simulation.axis[0][0], clip=False)
        return (x_idx, y_idx)

    def get_cores(self):
        try:
            self.cores, coords = self.simulation.get_cores(axis=self.axis, flip_y=False)
            cores = []
            for i,c in enumerate(self.cores):
                new_core = DenseCore(self,{
                    **c
                })
                new_core.data['pos_x'] = coords[0][i]
                new_core.data['pos_y'] = coords[1][i]
                cores.append(new_core)
            self.cores = cores

        except:
            LOGGER.error("Can't load cores in observation simulation")
        return self.cores
    
    def pc_to_pixels(self, pc):
        result = pc*self.simulation.nres/self.simulation.size
        return result
    