import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class Node:

    def __init__(self, out_dims):
        self.out_dims = np.array(out_dims, dtype=int)

    def reassign_links(self, old, new):
        self.out_dims[self.out_dims == old] = new
        

class TensorNetwork:

    def __init__(self, nodes: list[Node]):
        self.nodes = nodes
        self.active = np.full(len(nodes), True, dtype=bool)

    def links_of(self, i):
        return self.nodes[i].out_dims

    def merge_nodes(self, i: int, j: int) -> int:

        if i >= len(self.nodes) or j >= len(self.nodes):
            raise ValueError(
                f"Cannot contract: node {i} or {j} not in network")

        if not self.active[i] or not self.active[j]:
            raise ValueError(
                f"Either {i} or {j} is inactive")

        if i == j:
            return 0

        x = self.links_of(i)
        y = self.links_of(j)
        n_mutual = np.sum(x == j)

        if n_mutual != np.sum(y == i):
            raise ValueError(f"Mismatch in links between nodes {i} and {j}")

        cost = (len(x) - n_mutual) * (len(y) - n_mutual) * n_mutual

        # For every neighbor connected to node i,
        # replace their connection to i with a connection to j
        for neighbor in x:
            if neighbor not in [-1, j]:
                self.nodes[neighbor].reassign_links(old=i, new=j)

        self.nodes[j].out_dims = np.concatenate([x[x != j], y[y != i]])

        self.active[i] = False

        return cost

    def draw_network(self, title='Tensor Network'):
        
        G = nx.MultiDiGraph()
        edge_index_map = {}  # (src, dst) -> list of indices

        for i, node in enumerate(self.nodes):
            if not self.active[i]:
                continue
            G.add_node(i)
            for j in node.out_dims:
                if j != -1:
                    G.add_edge(i, j)
                    edge_index_map[(i, j)] = np.where(self.links_of(i)==j)

        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True)

        # Custom edge label rendering for MultiDiGraph
        ax = plt.gca()
        for (u, v), indices in edge_index_map.items():
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            alpha = 0.15  # position near the start of the edge
            x, y = (1 - alpha) * x1 + alpha * x2, (1 - alpha) * y1 + alpha * y2
            label = str(list(*indices))
            ax.text(x, y, label, color = 'red')

        plt.title(title)
        plt.show()



