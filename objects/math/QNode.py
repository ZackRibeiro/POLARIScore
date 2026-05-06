import copy
import numpy as np
from typing import *

class QNode():
    def __init__(self,position:np.ndarray,data:Dict):
        self.position=position
        self.data:Dict = {}
        self.host:'QuadTree' = None
    def dist_to(self,node_2:'QNode'):
        diff = node_2
        return np.linalg.norm(node_2.position-self.position)
    def equals(self,node_2:'QNode'):
        if any(self.position[i] != node_2.position[i] for i in range(len(self.position))):
            return False
        for key in self.data.keys():
            if not(key in node_2.data):
                return False
            if node_2.data[key] != self.data[key]:
                return False
        return True
    def clone(self):
        clone = QNode(self.position.copy(),{})
        clone.data = copy.deepcopy(self.data)
        clone.host = self.host
        return clone