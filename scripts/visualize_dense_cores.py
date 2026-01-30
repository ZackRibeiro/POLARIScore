import os
import argparse
import matplotlib.pyplot as plt
from POLARIScore.utils.utils import printProgressBar, plot_rect_bg, dictsToString

"""

Plot and save zoom on dense cores

Usage:
python -m POLARIScore.scripts.visualize_dense_cores --obs_name OrionB --obs_dist 400 --obs_suffixes unet cinn ddpm

"""

parser = argparse.ArgumentParser()
parser.add_argument("--obs_name", required=True, help="Molecular cloud name / Folder name of the observation")
parser.add_argument("--obs_dist", required=False, default=400., help="Distance to the molecular cloud")
parser.add_argument("--obs_suffixes", required=False, nargs="+", default=None, help="Suffixes of the .npy predictions")

parser.add_argument("--plot_details",required=False,default='no',help="If yes, plot cores details with slices.")

parser.add_argument("--folder_name", required=False, default="dense_cores")

args = parser.parse_args()

if str.lower(args.plot_details) in ["yes","y","true"]:
    args.plot_details = True
else:
    args.plot_details = False

import shutil
from POLARIScore.config import EXPORT_FOLDER, LOGGER
folder_path = os.path.join(EXPORT_FOLDER, args.folder_name)
if os.path.exists(folder_path):
    LOGGER.warn(f"There was a previous folder named this way ({args.folder_name}), the folder was removed.")
    shutil.rmtree(folder_path)
os.mkdir(folder_path)
global_path = os.path.join(folder_path, "comparison")
os.mkdir(global_path)
if args.plot_details:
    for s in args.obs_suffixes:
        os.mkdir(os.path.join(folder_path,s))

from POLARIScore.objects.Observation import Observation
observation:'Observation' = None
if args.obs_name is not None:
    observation = Observation(args.obs_name,"column_density_map")
    observation.get_cores(use_deconvolved_values=False)
    observation.distance = float(args.obs_dist)
    assert args.obs_suffixes is not None, LOGGER.error("You must give corresponding prediction suffixes.")
else:
    LOGGER.error("No observation name given.")

core_properties = []
for i,c in enumerate(observation.get_cores()):
    printProgressBar(i, len(observation.get_cores()), prefix="Plotting...", length=30)
    
    fig = plt.figure(figsize=(2*(len(args.obs_suffixes)+1),4),dpi=200.)
    cdens_ax = plt.subplot2grid((1, 1+len(args.obs_suffixes)),(0, 0), fig=fig, projection=observation.wcs)
    c.plot(ax=cdens_ax, env_size=None, cdens=True, cbar=True, contour=True, nearby_cores=False, show_marker=False, show_title=False, show_legend=False, show_ticks=False, contour_levels=10
           ,cbar_settings={'fraction':0.05, 'location':"bottom", 'orientation':"horizontal", 'pad':0.05})

    axes = [None for _ in range(len(args.obs_suffixes))]
    continue_flag = False
    c_props = c.get_properties()
    del c_props['mass']
    del c_props['vdens']

    for j,s in enumerate(args.obs_suffixes):
        observation.load("_"+s)
        vdens_ax = plt.subplot2grid((1, 1+len(args.obs_suffixes)),(0, j+1), fig=fig, projection=observation.wcs)
        try:
            cbar_settings = {'fraction':0.05, 'location':"bottom", 'orientation':"horizontal", 'pad':0.05}
            if j != int(len(args.obs_suffixes)/2):
                cbar_settings['label'] = ""
            c.plot(ax=vdens_ax, env_size=None, cdens=False, cbar=True, contour=True, nearby_cores=False, show_marker=False, toplabel=s, show_title=False, show_legend=False, show_ticks=False, contour_levels=10
                   ,cbar_settings=cbar_settings)
        except:
            continue_flag = True
            continue
        axes[j] = vdens_ax

        if args.plot_details:
            fig_details, _ = c.plot(save_path=folder_path + "/"+s+"/")
            plt.close(fig_details)
        
        c_props['mass_'+s] = c.compute_mass()
        c_props['vdens_'+s] = c.get_center_density()
    
    if continue_flag:
        plt.close(fig)
        continue


    core_properties.append(c_props)
    

    plot_rect_bg(fig, axes=axes, color="tab:orange", pad=0.0)
    plot_rect_bg(fig, axes=[cdens_ax], color="tab:blue", pad=0.0)
    
    fig.savefig(os.path.join(global_path, f"core_{c.data['name']}.jpg"))
    plt.close(fig)

string = dictsToString(core_properties)
with open(os.path.join(folder_path, "core_properties.txt"), "w") as file:
    file.write(f"Core properties \n")
    file.write(f"Maps used: "+str(args.obs_suffixes)+"\n")
    file.write(f"---------------------------------------------"+"\n")
    file.write(string)

LOGGER.log("Plotting cores done")