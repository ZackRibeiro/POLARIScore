from POLARIScore.config import LOGGER
from astropy import units as u
import json, os
import numpy as np
from POLARIScore.objects.Simulation_DC import Simulation_DC
import glob

def init_ramses(simulation:'Simulation_DC', loadTemp=False, loadVel=False):
    """
    Init a simulation made with ramses

    Args:
        loadTemp (bool): try to load temperature ?
        loadVel (bool): try to load velocity ?
    """
    LOGGER.log(f"Loading simulation {simulation.name} using RAMSES init")
    
    assert simulation.load_fit(key="RHO", path="datacube")
    if loadTemp:
        LOGGER.log(f"Loading temperature of simulation {simulation.name}")
        simulation.load_fit(key="TEMP", path="datacube_temp")
    if loadVel:
        LOGGER.log(f"Loading velocity of simulation {simulation.name}")
        simulation.load_fit(key="VX1", path="datacube_velx")
        simulation.load_fit(key="VX2", path="datacube_vely")
        simulation.load_fit(key="VX3", path="datacube_velz")

    simulation.nres = simulation.data['RHO'].shape[0]
    if os.path.exists(os.path.join(simulation.folder,"processing_config.json")):
        with open(os.path.join(simulation.folder,"processing_config.json"), "r") as file:
            simulation.header = json.load(file)
        simulation.relative_size = simulation.header["run_parameters"]["size"]
        simulation.center = np.array([simulation.header["run_parameters"]["xcenter"],simulation.header["run_parameters"]["ycenter"],simulation.header["run_parameters"]["zcenter"]])
        simulation.cell_size = (simulation.global_size*simulation.relative_size/simulation.nres) * u.parsec
        simulation.cell_size = simulation.cell_size.to(u.cm).value
        simulation.size = simulation.global_size*simulation.relative_size
        simulation.axis = ([simulation.center[0]*simulation.global_size-simulation.size/2,simulation.center[0]*simulation.global_size+simulation.size/2],[simulation.center[1]*simulation.global_size-simulation.size/2,simulation.center[1]*simulation.global_size+simulation.size/2],[simulation.center[2]*simulation.global_size-simulation.size/2,simulation.center[2]*simulation.global_size+simulation.size/2])    
    LOGGER.log(f"Simulation {simulation.name} loaded.")

from POLARIScore.utils.vtk_io import readVTKCart
def init_idefix(simulation:'Simulation_DC', blacklist=[], invert_axes=False):
    """
    Init a simulation made with idefix
    """
    #Add idefix.ini for units bcs for now this is units code

    LOGGER.log(f"Loading simulation {simulation.name} using IDEFIX init")
    vtk = readVTKCart(glob.glob(os.path.join(simulation.folder, "*.vtk"))[0])
    for key in vtk.data:
        if key in blacklist:
            continue
        d = vtk.data[key]
        if type(d) is list:
            d = np.array(d)
        simulation.data[key] = d
        LOGGER.log(f"Key {key} added to simulation data.")

    if invert_axes:
        simulation.data['RHO'] = simulation.data['RHO'].transpose()
        for s in ["BX", "VX"]:
            if s+"1" in simulation.data:
                simulation.data["vel_cache"] = simulation.data[s+"1"]
                simulation.data[s+"1"] = simulation.data[s+"3"]
                simulation.data[s+"3"] = simulation.data["vel_cache"]
                del simulation.data["vel_cache"]

    simulation.nres = simulation.data['RHO'].shape[0]
    simulation.cell_size = (simulation.global_size*simulation.relative_size/simulation.nres) * u.parsec
    simulation.cell_size = simulation.cell_size.to(u.cm).value
    LOGGER.log(f"Simulation {simulation.name} loaded.")
    
    