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
from POLARIScore.objects.math.BinaryTree import OcTree
from POLARIScore.objects.math.QRegion import QRegion
from POLARIScore.objects.math.QNode import QNode

class Simulation_ARM():
    def __init__(self, name, global_size, init=True):
        self.name = name
        """Simulatio name, name of the folder where the sim is in"""
        self.folder = os.path.join(os.path.dirname(os.path.abspath(__file__)),"../data/sims/"+name+"/")
        """Path to the folder where the simulation is stored"""
        self.file = os.path.join(self.folder,"cellsToPoints_cgs.h5")
        """Path to the simulation data"""
        self.data:OcTree = None
        """Raw simulation data"""
        self.global_size = global_size

        if init:
            self.init()

    def init(self):
        """
        Load files and data in self variables
        """
        LOGGER.log("Initializing the AMR Simulation  ")

        with h5py.File(self.file, "r") as f:
            points = f["points"][:] 
            density = f["scalars/density"][:] 
            size = f["scalars/size"][:]
            LOGGER.log(f"Particle sizes go from {min(size):.4f} to {max(size):.4f}")

        x, y, z = points[:, 0], points[:, 1], points[:, 2]

        parsec_to_cm = 3.0857e18
        mean_molecular_weight = 1.4
        number_density = density / (mean_molecular_weight * m_p.value)  # cm⁻³

        x, y, z = points[:, 0], points[:, 1], points[:, 2]

        LOGGER.log(f"Number of points in the simulation: {len(x)}")
        center = np.array([0.5,0.5,0.5])
        half_size = 0.2
        region = QRegion(center, half_size)
        oct_tree = OcTree(region)
        oct_tree.print_logs = False

        for i in range(len(x)):
            printProgressBar(i, len(x), "Creating OcTree")
            node = QNode(np.array([x[i],y[i],z[i]]),{"density":number_density,"size":size,"mass":number_density})
            oct_tree.insert_leaves(node)
        
        self.data = oct_tree
        

if __name__ == "__main__":
    sim = Simulation_ARM("orionMHD_lowB_AMR_low", 66.0948)
    data = sim.data.to_datacube()
    plt.imshow(np.sum(data,axis=0),norm=LogNorm())
    plt.colorbar()