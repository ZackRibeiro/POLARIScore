from POLARIScore.config import LOGGER
from astropy import units as u
import json, os
import numpy as np
import glob
from typing import Tuple, List, Union, Callable, Any, Optional

def init_ramses(simulation, loadTemp=False, loadVel=False):
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
        simulation.load_fit(key="VX1", path="datacube_velx", unit=1e4)
        simulation.load_fit(key="VX2", path="datacube_vely", unit=1e4)
        simulation.load_fit(key="VX3", path="datacube_velz", unit=1e4)

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
def init_idefix(simulation, blacklist=[], invert_axes=False):
    """
    Init a simulation made with idefix
    """
    #Add idefix.ini for units bcs for now this is units code

    LOGGER.log(f"Loading simulation {simulation.name} using IDEFIX init")
    vtk = readVTKCart(glob.glob(os.path.join(simulation.folder, "*.vtk"))[0])

    ini_path = os.path.join(simulation.folder, "idefix.ini")
    dens_unit = 1.
    vel_unit = 1.
    length_unit = 1.
    if os.path.exists(ini_path):
        with open(ini_path, "r", encoding="utf-8") as file:
            lines = file.readlines()
            units_index = len(lines)
            for i, line in enumerate(lines):
                if i > units_index+3:
                    break
                props = line.strip().split()
                if "[Units]" in line:
                    units_index = i
                if i > units_index:
                    if "length" in props[0]:
                        length_unit = float(props[1])
                    elif "density" in props[0]:
                        dens_unit = float(props[1])
                    elif "velocity" in props[0]:
                        vel_unit = float(props[1])
            if units_index == len(lines):
                LOGGER.warn("When reading idefix.ini, no units block was found -> Data is maybe in code units.")
    else:
        LOGGER.warn("No idefix.ini found in simulation folder -> Data is maybe in code units.")
    #!! RHO IS HYDROGEN NUMBER DENSITY
    key_match_unit = {
        "RHO": dens_unit / (1.673e-24 *1.4),
        "VX1": vel_unit, "VX2": vel_unit, "VX3": vel_unit,
        "BX1": dens_unit**(1/2)*vel_unit, "BX2": dens_unit**(1/2)*vel_unit, "BX3": dens_unit**(1/2)*vel_unit
    }

    for key in vtk.data:
        if key in blacklist:
            continue
        d = vtk.data[key]
        if type(d) is list:
            d = np.array(d)
        if key in key_match_unit:
            d = d * key_match_unit[key]
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

from POLARIScore.utils.utils import compute_column_density, compute_mass_weighted_density
from POLARIScore.networks.utils.nn_utils import predict_map
def predict_average_density(simulation, model_trainer, average_method=compute_mass_weighted_density, patch_size:Tuple[int,int]=(128, 128), nan_value:float=-1.0, overlap:float=0.5, downsample_factor:float=1., apply_baseline:bool=True):
    
    column_density = [compute_column_density(simulation.data['RHO'], simulation.cell_size, axis=i) for i in range(3)]
    volume_density = [simulation._compute_v_density(average_method, axis=i) for i in range(3)]
    
    return [predict_map(compute_column_density(simulation.data['RHO'], simulation.cell_size, axis=i), model_trainer, patch_size, nan_value, overlap, downsample_factor, apply_baseline) for i in range(3)]