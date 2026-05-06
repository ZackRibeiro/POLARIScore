from POLARIScore.utils.utils import is_vector_in_box
from POLARIScore.objects.math.QNode import QNode
import numpy as np

class QRegion():

    def __init__(self, center:np.ndarray, half_length:float):
        self.center = center
        self.half_length = half_length

    def contains(self,node:QNode):
        return is_vector_in_box(node.position, self.center, self.half_length)