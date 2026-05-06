
QT_PARTICLE_CAPACITY = 1000
"""How many particles a quadtree can contain"""

import numpy as np
from POLARIScore.utils.utils import is_vector_in_box_2, printProgressBar
from POLARIScore.objects.math.QRegion import QRegion
from POLARIScore.objects.math.QNode import QNode
from POLARIScore.config import LOGGER, CACHES_FOLDER
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple, Union, Dict, Optional
import json
import os

class OcTree():
    def __init__(self, region:QRegion):
        self.region:QRegion = region
        self.subqt:List[Optional['OcTree']] = [None,None,None,None,None,None,None,None]
        self.parent:Optional[Tuple['OcTree',int]] = None
        self.nodes:List[QNode] = []
        self.masscenter:Optional[Tuple[float, np.ndarray]] = None

        self.minsize = None

        self.print_logs = False

    def insert(self,node:QNode)->bool:

        if(not(self.region.contains(node))):
            return False
        if(len(self.nodes) < QT_PARTICLE_CAPACITY and self.subqt[0] == None):
            node.host = self
            self.nodes.append(node)
            return True
        
        if(self.subqt[0] == None):
            self.subdivide()

        for qt in self.subqt:
            if(qt.insert(node)):
                return True
        
        return False
    
    def insert_leaves(self, node: QNode) -> bool:
        if not self.region.contains(node):
            return False

        cell_size = node.data.get("size", 0)
        node_size = self.region.half_length * 2

        if abs(node_size - cell_size) < 1e-10:
            self.nodes.append(node)
            node.host = self
            return True

        if self.subqt[0] is None:
            self.subdivide()

        for qt in self.subqt:
            if qt.insert_leaves(node):
                return True

        self.nodes.append(node)
        return True
    
    def subdivide(self):
        """Subdivide the octree into 8 smaller trees"""


        h = self.region.half_length / 2
        cx = self.region.center
        i = 0

        if(self.print_logs):
            LOGGER.log(f"Subdividing OT {cx,h}.")

        for dx in (-h, h):
            for dy in (-h, h):
                for dz in (-h, h):
                    offset = np.array([dx, dy, dz])
                    self.subqt[i] = OcTree(QRegion(cx + offset, h))
                    self.subqt[i].print_logs = self.print_logs
                    i += 1
        
        for i in range(len(self.subqt)):
            self.subqt[i].parent = (self,i)

        qt_particles = []
        qt_particles.extend(self.nodes)
        for p in qt_particles:
            for qt in self.subqt:
                if(qt.region.contains(p)):
                    qt.insert(p)
                    self.nodes.remove(p)
                    break
        if(len(self.nodes) > 0):
            LOGGER.error("QT node not in subtree")

    def query(self, min, max)->List[QNode]:
        """Get all the particles in a demarcated area"""

        result = []

        if(any([self.region.center[i]+self.region.half_length < min[i] or self.region.center[i]-self.region.half_length > max[i] for i in range(len(min))])):
            return result
        
        for p in self.nodes:
            if(is_vector_in_box_2(p.position, min, max)):
                result.append(p)

        if(self.subqt[0] == None):
            return result
        
        for qt in self.subqt:
            result.extend(qt.query(min,max))

        return result
    
    def compute_mass_center(self):
        masscenter = (0.,np.array([0,0,0]))
        if(self.subqt[0] != None):
            for qt in self.subqt:
                m_c = qt.masscenter
                if(m_c == None):
                    m_c = qt.compute_mass_center()
                masscenter[0] += m_c[0]
                masscenter[1] = masscenter[1] + m_c[1] * m_c[0]
            masscenter[1] = masscenter[1] /masscenter[0]
        
        if(len(self.nodes) > 0 and self.subqt[0] == None):
            p_masscenter = [0, np.array([0,0,0])]
            for p in self.nodes:
                mass = p.data['mass'] if 'mass' in p.data else 1
                p_masscenter[0] += mass
                p_masscenter[1] = p_masscenter[1] + (p.position * mass)
            p_masscenter[1] = p_masscenter[1] /p_masscenter[0]

            masscenter[0] = masscenter[0] + p_masscenter[0]
            masscenter[1] = (masscenter[1]+p_masscenter[1]*p_masscenter[0])/masscenter[0]

        self.masscenter = masscenter
        return masscenter
     
    def get_all_nodes(self)->List[QNode]:
        """Return a flatten list of all the particles in the QuadTree and his children"""
        part = []
        part.extend(self.nodes)

        if(self.subqt[0] == None):
            return part

        for qt in self.subqt:
            part.extend(qt.get_all_nodes())

        return part
    
    def get_min_size(self):
        min_size = self.region.half_length*2.
        if(self.subqt[0] != None):
            for qt in self.subqt:
                min_size = min(min_size,qt.get_min_size())
        return min_size
    
    def get_closest_distance(self, pos:np.ndarray)->float:
        c_distance = np.linalg.norm(pos-self.region.center)
        if(self.subqt[0] != None):
            for qt in self.subqt:
                c_distance = min(c_distance,qt.get_closest_distance(pos))
        return c_distance 

    def to_datacube(self, resolution:int=64, key:str="density", sigma:Optional[float]=None, cache:Optional[str]="octree_cached"):
        center = self.region.center
        size = self.region.half_length * 2
        dx = size / resolution

        if sigma is None:
            sigma = dx

        grid = np.zeros((resolution, resolution, resolution), dtype=np.float32)
        weight = np.zeros_like(grid)

        nodes = self.get_all_nodes()
        if len(nodes) == 0:
            LOGGER.warn("Can't make a datacube because Octree is empty")
            return grid

        def kernel(r2):
            return np.exp(-r2 / (2 * sigma**2))

        for p in nodes:
            if key not in p.data:
                continue

            pos = p.position
            val = p.data[key]

            rel = (pos - (center - size/2)) / dx
            ix, iy, iz = rel.astype(int)

            r_vox = int(np.ceil(3 * sigma / dx))

            for i in range(ix - r_vox, ix + r_vox + 1):
                if i < 0 or i >= resolution: continue
                for j in range(iy - r_vox, iy + r_vox + 1):
                    if j < 0 or j >= resolution: continue
                    for k in range(iz - r_vox, iz + r_vox + 1):
                        if k < 0 or k >= resolution: continue

                        voxel_pos = np.array([
                            center[0] - size/2 + i*dx,
                            center[1] - size/2 + j*dx,
                            center[2] - size/2 + k*dx
                        ])

                        d2 = np.sum((pos - voxel_pos)**2)
                        w = kernel(d2)

                        grid[i, j, k] += val * w
                        weight[i, j, k] += w

        mask = weight > 0
        grid[mask] /= weight[mask]

        path = os.path.join(CACHES_FOLDER, cache+f"_{resolution}.npy")
        if os.path.exists(path):
             LOGGER.warn(f"A previous cache named {cache}_{resolution}.npy was replaced(removed).")
             os.remove(path)
             np.save(path, grid)

        return grid

    def save(self):
        pass

    @staticmethod
    def load(name:str)->'OcTree':
        pass
    
    def to_dict(self):
        return {
            "center": self.region.center.tolist(),
            "half_length": self.region.half_length,
            "nodes": [
                {
                    "pos": n.position.tolist(),
                    "data": n.data
                } for n in self.nodes
            ],
            "children": [qt.to_dict() if qt else None for qt in self.subqt]
        }
    
    @staticmethod
    def from_dict(data):
        region = QRegion(np.array(data["center"]), data["half_length"])
        tree = OcTree(region)

        tree.nodes = [
            QNode(np.array(n["pos"]), n["data"])
            for n in data["nodes"]
        ]

        if data["children"]:
            tree.subqt = [
                OcTree.from_dict(child) if child else None
                for child in data["children"]
            ]
            for i, child in enumerate(tree.subqt):
                if child:
                    child.parent = (tree, i)

        return tree