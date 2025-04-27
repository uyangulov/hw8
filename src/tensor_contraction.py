import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class Node:

    def __init__(self, links, dim_ranges):
        '''
        links[i] is index of neighbor of node connected to i-th dimension
        dim_ranges[i] is size of dimension i
        '''
        self.links = np.array(links, dtype=int)
        self.dim_ranges = np.array(dim_ranges)

    def reassign_links(self, old, new):
        '''
        If any connections to node with index 'old', redirect to index 'new'
        '''
        self.links[self.links == old] = new


class TensorNetwork:

    def __init__(self, nodes: list[Node]):
        self.nodes = nodes
        self.active = np.full(len(self), True, dtype=bool)

    def __len__(self) -> int:
        return len(self.nodes)

    def links_of(self, i):
        return self.nodes[i].links

    def ranges_of(self, i):
        return self.nodes[i].dim_ranges

    def mutual_ranges(self, i, j):
        '''
        Return sizes of dimensions to be merged with node j
        (or empty array, if no such dimensions)
        '''
        return self.ranges_of(i)[self.links_of(i) == j]

    def validate_merge(self, i: int, j: int) -> None:

        if i == j:
            raise ValueError(f"Attempt to merge node {i} with itself")

        if i >= len(self) or j >= len(self):
            raise ValueError(
                f"Cannot contract: node {i} or {j} not in network")

        if not self.active[i] or not self.active[j]:
            raise ValueError(f"Either {i} or {j} is inactive")

        print(self.mutual_ranges(i, j))
        print(self.mutual_ranges(j, i))
        if np.any(self.mutual_ranges(i, j) != self.mutual_ranges(j, i)):
            raise ValueError(
                f"Cannot merge nodes {i}, {j} over dims of different size")

    def merge_cost(self, i,  j):
        '''
        Cost of merge, including case of no common indices
        '''
        cost = np.prod(self.ranges_of(i)) * np.prod(self.ranges_of(j))
        mutual_ranges = self.mutual_ranges(i, j)
        if mutual_ranges.size != 0:
            cost /= np.prod(mutual_ranges)
        return cost

    def merge_nodes(self, i: int, j: int) -> int:
        '''
        1) Merge nodes i and j
        2) Save result at index j
        3) Return cost of the merge
        '''
        self.validate_merge(i, j)
        cost = self.merge_cost(i, j)

        # For every neighbor connected to node i,
        # replace their connection to i with a connection to j
        for neighbor in self.links_of(i):
            if neighbor not in [-1, j]:
                self.nodes[neighbor].reassign_links(old=i, new=j)

        # Save result of merge to j according to rules of np.tensordot
        x = self.links_of(i)
        y = self.links_of(j)
        rg_x = self.ranges_of(i)
        rg_y = self.ranges_of(j)
        self.nodes[j].dim_ranges = np.concatenate([rg_x[x != j], rg_y[y != i]])
        self.nodes[j].links = np.concatenate([x[x != j], y[y != i]])
        self.active[i] = False
        return cost

    def draw_network(self, title='Tensor Network'):

        G = nx.MultiDiGraph()
        edge_index_map = {}  # (src, dst) -> list of indices

        for i, node in enumerate(self.nodes):
            if not self.active[i]:
                continue
            G.add_node(i)
            for j in node.links:
                if j != -1:
                    G.add_edge(i, j)
                    edge_index_map[(i, j)] = np.where(self.links_of(i) == j)

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
            ax.text(x, y, label, color='red')

        plt.title(title)
        plt.show()
