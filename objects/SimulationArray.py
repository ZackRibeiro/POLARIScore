from typing import Optional, List, Tuple, Any
from POLARIScore.objects.Simulation_DC import Simulation_DC
from POLARIScore.config import LOGGER
from POLARIScore.utils import longest_common_substring

class SimulationArray():

    def __init__(self, simulations:List['Simulation_DC'], indexes:Optional[List[float]]=None ,name:Optional[str]=None):
        assert type(simulations) is list, LOGGER.error("SimulationArray need to be initialized with a list of simulations")
        if indexes is None:
            indexes = range(len(simulations)) if len(simulations) > 0 else []
        else:
            assert len(indexes) == len(simulations), LOGGER.error("Length of indexes need to be the same of the length of simulations for init a SimulationArray")

        self.simulations:List['Simulation_DC'] = simulations
        self.indexes:List[float] = indexes
        """indexes ~ timesteps"""

        if name is None:
            name = longest_common_substring([s.name for s in simulations])
            if len(name) <= 3:
                name = simulations[0].name

        self.name:str = name

    
    