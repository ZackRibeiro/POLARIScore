import numpy as np
import matplotlib.pyplot as plt
from typing import List, Union, Tuple, Optional, Literal, Dict
from POLARIScore.config import LOGGER

class Node():
    def __init__(self,position:np.ndarray):
        self.position = np.array(position)
        self.properties = {}

    def is_same_as(self, n:'Node'):
        is_same_node = True

        is_same_node = is_same_node and np.allclose(n.position, self.position)
        if not(is_same_node):
            return False
        for p_key in n.properties.keys():
            if p_key not in self.properties:
                is_same_node = False
                break
            if self.properties[p_key] != n.properties[p_key]:
                is_same_node = False
                break
        return is_same_node
    
    def share_prop(self, obj:Union[Dict, np.ndarray])->bool:
        if isinstance(obj, dict):
            for k in obj.keys():
                if k not in self.properties:
                    return False
                if self.properties[k] == obj[k]:
                    return True
        elif isinstance(obj, np.ndarray):
            return self.position == obj
        return False

class Graph():
    def __init__(self):
        self.nodes:List[Node] = []
        self.edges:List[Tuple[int,int]] = []
        self.properties = {}

    def contains_node(self, node:Node)->bool:
        for n in self.nodes:
            if node.is_same_as(n):
                return True
        return False
    
    def index_of_node(self, node:Node)->Union[int, None]:
        for i,n in enumerate(self.nodes):
            if node.is_same_as(n):
                return i
        return None
    
    def add_edge(self, node1:Union[Node, int], node2:Union[Node, int])->Union[int, None]:
        if isinstance(node1, Node):
            node1 = self.index_of_node(node1)
        if isinstance(node2, Node):
            node2 = self.index_of_node(node2)
        if node1 is None or node2 is None:
            LOGGER.warn("An edge can't be created because one node is not in the graph.")
            return None 
        
        self.edges.append((node1, node2))
        return len(self.edges)-1
    
    def remove_edges(self, object:Union[int, Node])->int:
        if isinstance(object, Node):
            object = self.index_of_node(object)
        was_removed = 0
        for i, edge in enumerate(self.edges):
            if object in edge:
                self.edges.pop(i-was_removed)
                was_removed += 1
        return was_removed
    
    def remove_node(self, object:Union[Dict, Node, np.ndarray])->int:
        """Remove all nodes that:
        - are the same as object if object is Node
        - share the properties contained in object if object is a Dict
        - have the same position=object if object is a np.ndarray

        Returns:
            int: how many nodes was removed
        """
        was_removed = 0
        for i, n in enumerate(self.nodes):
            index = i-was_removed
            remove = False
            if isinstance(object, Node):
                if object.is_same_as(n):
                    remove = True
            else:
                if n.share_prop(object):
                    remove = True
            if remove:
                self.remove_edges(i)
                self.nodes.pop(index)
                was_removed += 1

        if was_removed > 0:
            for i,edge in enumerate(self.edges):
                edge[0] -= was_removed
                edge[1] -= was_removed
                if edge[0] < 0 or edge[1] < 0:
                    self.edges.pop(i)
                
        return was_removed

    def add_node(self, node:Union[Node,np.ndarray,list], force:bool=False)->Node:
        if isinstance(node, np.ndarray) or isinstance(node, list):
            node = Node(node)
        contains_node = self.contains_node(node)
        if not(contains_node) or force:
            if force and not(contains_node):
                LOGGER.warn("A node was already present in the graph, but it was forcibly added.")
            self.nodes.append(node)
        return node

    def get_nodes(self, node:Union[Node, np.ndarray, list, Dict])->Tuple[List[int],List[Node]]:
        nodes = []
        indexes = []
        for i, n in enumerate(self.nodes):
            if isinstance(object, Node):
                if object.is_same_as(n):
                    indexes.append(i)
                    nodes.append(n)
            else:
                if n.share_prop(object):
                    indexes.append(i)
                    nodes.append(n)
        return indexes, nodes
    
    def plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        for node in self.nodes:
            ax.scatter(node.position[0], node.position[1], color="red")
        for edge in self.edges:
            pos1 = self.nodes[edge[0]].position
            pos2 = self.nodes[edge[1]].position
            ax.plot([pos1[0],pos2[0]],[pos1[1],pos2[1]], color="red")

        return fig, ax