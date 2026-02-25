from POLARIScore.objects.tools.Graph import Graph, Node 
import matplotlib.pyplot as plt
from matplotlib import rcParams

class Dendrogram(Graph):
    def __init__(self):
        super().__init__()

    def get_leaves(self):
        return self.get_nodes({"is_leaf":True})
    
    def get_roots(self):
        return self.get_nodes({"is_root":True})
    
    def plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        for edge in self.edges:
            pos1 = self.nodes[edge[0]].position
            pos2 = self.nodes[edge[1]].position
            ax.plot([pos1[0],pos2[0]],[pos1[1],pos2[1]], color="black")
        for node in self.nodes:
            color = "black"
            size = 0.5*rcParams['lines.markersize'] ** 2
            if "is_leaf" in node.properties and node.properties["is_leaf"]:
                color = "green"
                size = size*2
            if "is_root" in node.properties and node.properties["is_root"]:
                color = "orange"
                size = size*2
            ax.scatter(node.position[0], node.position[1], color=color, s=size)


        return fig, ax