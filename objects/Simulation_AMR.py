import sys
import os
import yt
import h5py
from POLARIScore.utils.utils import *
from POLARIScore.config import *
import json
from astropy.constants import m_p
from astropy import units as u
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
# Create a slice plot of density
#slc = yt.SlicePlot(ds, "z", "density")
#slc.show()

from scipy import ndimage

def fill_zeros_nearest(arr):

    mask = arr != 0
    idx = ndimage.distance_transform_edt(~mask, return_indices=True)

    ix, iy, iz = idx[1]

    out = arr[ix, iy, iz]

    return out

#def fill_zeros_nearest()


class Simulation_ARM():
    def __init__(self, name, global_size, init=True):
        self.name = name
        """Simulatio name, name of the folder where the sim is in"""
        self.folder = os.path.join(os.path.dirname(os.path.abspath(__file__)),"../data/sims/"+name+"/")
        """Path to the folder where the simulation is stored"""
        self.file = os.path.join(self.folder,"cellsToPoints_cgs.h5")
        """Path to the simulation data"""
        self.data = None
        """Raw simulation data"""
        self.global_size = global_size

        if init:
            self.init()

    def init(self):
        """
        Load files and data in self variables
        """

        with h5py.File(self.file, "r") as f:
            points = f["points"][:] 
            density = f["scalars/density"][:] 
            size = f["scalars/size"][:]

        parsec_to_cm = 3.0857e18
        mean_molecular_weight = 1.4
        number_density = density / (mean_molecular_weight * m_p.value)  # cm⁻³

        max_size = np.min(size)
        print(max_size, 1/max_size)

        min_pos = 0.3
        max_pos = 0.7
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        data = {
            "particle_position_x": (x, "code_length"),
            "particle_position_y": (y, "code_length"),
            "particle_position_z": (z, "code_length"),
            "particle_density": (number_density, "cm**-3"),
            "particle_size": (size, "code_length"),
            "particle_mass": (density*(size*parsec_to_cm*self.global_size)**3, "g"),
            "particle_volume": (size**3, "code_length**3"),
        }
        #particle_smoothing_length

        print(np.min(x), np.max(x))
        print(np.min(y), np.max(y))
        print(np.min(z), np.max(z))

        # Define domain bounding box (min/max values of coordinates)
        #bbox = np.array([[min_pos,max_pos], [min_pos,max_pos], [min_pos,max_pos]])
        bbox = np.array([[np.min(x),np.max(x)], [np.min(y),np.max(y)], [np.min(z),np.max(z)]])


        # Load as a particle dataset in yt
        self.data = yt.load_particles(data, bbox=bbox, length_unit=parsec_to_cm*self.global_size)
        field = "particle_density"
        n1 = self.data.add_deposited_particle_field(("all", field), "sum")
        n2 = self.data.add_deposited_particle_field(("all", field), "count")

        print(n1, n2)
        

if __name__ == "__main__":
    sim = Simulation_ARM("orionMHD_lowB_AMR", 66.0948)

    ds = sim.data
    
    size = 8

    wanted_nres = 1024
    level = np.log2(wanted_nres)
    #print(level)
    #cg = ds.covering_grid(
    #    level=level,
    #    left_edge=[.5,.5,.5],
    #    dims=ds.domain_dimensions * size
    #)

    cg = ds.arbitrary_grid(
        #left_edge=[0.48,0.48,ds.domain_left_edge[-1]],
        left_edge=[0.48,0.48,0.48],
        #right_edge=[0.52,0.52,ds.domain_right_edge[-1]],
        right_edge=[0.52,0.52,0.52],

        dims=(512, 512, 512)
    )

    sum = cg["deposit", "all_sum_density"].to_ndarray()
    count = cg["deposit", "all_count"].to_ndarray()
    tensor = np.divide(
        sum,
        count,
        out=np.zeros_like(sum),
        where=(count > 0) & (sum > 0)
    )
    tensor = fill_zeros_nearest(tensor)
    
    #p = yt.ProjectionPlot(ds, "z", ("deposit", "all_nn_density"))
    #p.set_cmap(("deposit", "all_nn_density"), "inferno")
    #p.set_zlim(("deposit", "all_nn_density"), 1e-3, 1e3)
    #p.show()

    #tensor = cg[field]
    plt.imshow(tensor[:,:,100], cmap="jet", norm=LogNorm())
    plt.colorbar()
    plt.show()
    