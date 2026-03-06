from typing import Optional, List, Tuple, Any, Callable, Literal
from POLARIScore.objects.Simulation_DC import Simulation_DC
from POLARIScore.objects.Dataset import Dataset, getDataset
from POLARIScore.config import LOGGER
from POLARIScore.utils import longest_common_substring
from POLARIScore.utils.sim_utils import init_idefix

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.widgets import Slider
import os, glob
import numpy as np
from matplotlib import cm
import inspect
import uuid


class SimulationArray():

    def __init__(self, simulations:List['Simulation_DC']=[], indexes:Optional[List[float]]=None ,name:Optional[str]=None):
        assert type(simulations) is list, LOGGER.error("SimulationArray need to be initialized with a list of simulations")
        indexes_was_none = False
        if indexes is None:
            indexes_was_none = True
            indexes = range(len(simulations)) if len(simulations) > 0 else []
        else:
            assert len(indexes) == len(simulations), LOGGER.error("Length of indexes need to be the same of the length of simulations for init a SimulationArray")

        self.simulations:List['Simulation_DC'] = simulations
        self.indexes:List[float] = indexes
        """indexes ~ timesteps"""

        if name is not None and len(simulations) == 0:
            LOGGER.log("Will try to auto-init simulation array, For now only available for IDEFIX simulations.")
            sim_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)),"../data/sims/"+name+"/")
            if os.path.exists(sim_folder):
                vtk_files = glob.glob(os.path.join(sim_folder,"*.vtk"))
                if len(vtk_files) > 0:
                    indexes = [int(os.path.basename(vtk_files[i]).split(".")[1]) for i in range(len(vtk_files))]
                    simulations = [Simulation_DC(name, init=False) for i,v in enumerate(vtk_files)]
                    for i,s in enumerate(simulations):
                        init_idefix(s, vtk_path=vtk_files[i])
                        if 'TIME' in s.data and 'OUT_VTK' in s.data:
                            s.data["T_INDEX"] = indexes[i]
                            indexes[i] = indexes[i]*s.data['OUT_VTK']
                    if indexes_was_none:
                        self.indexes = indexes
                    self.simulations = simulations
                else:
                    LOGGER.warn(f"Can't auto initialize simulation array -> there is no vtk files.")
            else:
                LOGGER.warn(f"Can't auto initialize simulation array -> there is no folder with such name: {name}.")

        if name is None:
            name = longest_common_substring([s.name for s in simulations])
            if len(name) <= 3:
                name = simulations[0].name

        sorted_sims = []
        sorted_indexes = []
        for i in np.argsort(indexes):
            sorted_sims.append(self.simulations[i])
            sorted_indexes.append(self.indexes[i])
        self.indexes = sorted_indexes
        self.simulations = sorted_sims

        self.name:str = name

    def plot(self, plot_method:Callable, ax:Optional["matplotlib.axes.Axes"]=None, mode:Literal["slider","mixed"]="mixed",
             colors=None, linestyles=None,
             **kwargs):
        """
        Plot any Simulation plotting method over the array.

        Args:
            method: Simulation_DC plotting method (e.g. Simulation_DC.plot_power_spectrum)
            ax: matplotlib axis
            mode: "mixed" or "slider"
            kwargs: forwarded to the plotting method
        """

        sig = inspect.signature(plot_method)
        accepts_ax = "ax" in sig.parameters
        accepts_fig = "fig" in sig.parameters

        if colors is not None and type(colors) is str:
            cmap = cm.get_cmap(colors)
            colors = [cmap(i/len(self.simulations)) for i in range(len(self.simulations))]
        if mode == "mixed":
            assert accepts_ax, LOGGER.error(f"{plot_method.__name__} creates its own figure and cannot be used in 'mixed' mode")
            if ax is None:
                fig, ax = plt.subplots()
            else:
                fig = ax.figure

            for i, sim in enumerate(self.simulations):
                if colors is not None:
                    kwargs["color"] = colors[i]
                try:
                    plot_method(sim, ax=ax, label=str(self.indexes[i]), **kwargs)
                except TypeError:
                    plot_method(sim, ax=ax, **kwargs)

            return fig, ax

        elif mode == "slider":
            assert accepts_fig, LOGGER.error(f"{plot_method.__name__} don't take fig in args.")
            fig, axes = plot_method(self.simulations[0], **kwargs)
            fig.suptitle(str(self.indexes[0]))
            plt.subplots_adjust(bottom=0.2)

            shared_obj = {"axes":axes}
            def draw(i, shared_obj):
                content_axes = shared_obj["axes"]
                for ax in content_axes:
                    if type(ax) is list or type(ax) is np.ndarray:
                        for a in ax:
                            a.remove()
                    else:
                        ax.remove()
                _, axes = plot_method(self.simulations[i], fig=fig, **kwargs)
                shared_obj["axes"] = axes
                fig.suptitle(str(self.indexes[i]))
                fig.canvas.draw_idle()

            slider_ax = fig.add_axes([0.2, 0.05, 0.6, 0.03])
            slider = Slider(slider_ax, "Index", 0, len(self.simulations)-1,
                            valinit=0, valstep=1)

            slider.on_changed(lambda v: draw(int(v),shared_obj))
            fig._array_slider = slider


            return None, None

        else:
            raise ValueError(f"Unknown mode: {mode}")

    def generate_dataset(self, name="merged_dataset", **kwargs):
        LOGGER.log("Generating a dataset using the simulation array.")
        dataset_names = [str(uuid.uuid4()) for _ in range(len(self.simulations))]
        for i,s in enumerate(self.simulations):
            s.generate_dataset(name=dataset_names[i], **kwargs)
        datasets = [getDataset("batch_"+n) for n in dataset_names]
        merged_dataset = datasets[0].merge(datasets[1:])
        merged_dataset.save(name=name)
        for d in datasets:
            d.delete()        
        return merged_dataset
        
    
    