import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class Edge:
    def __init__(self, nodes,)

class Node:
    
    def __init__(self, links, link_dims, dim_ranges):
        '''
        links[i] is index of neighbor of node connected to i-th dimension
        link_dims[i] is neighbor's dimension connected to dimension i 
        dim_ranges[i] is size of dimension i
        '''
        self.links = np.array(links, dtype=int)
        self.link_dims = np.array(link_dims, dtype=int)
        self.dim_ranges = np.array(dim_ranges)

    @property
    def ndims(self):
        return self.links.size

    def redirect(self, old, new, mapping, offset):
        '''
        If any connections to node with index 'old', redirect to index 'new'
        '''
        # mask = self.links == old
        # self.links[mask] = new
        # self.link_dims[mask] = mapping[self.link_dims[mask]] + offset
        print(old, new)
        for i, n in enumerate(self.links):
            if n == old:
                print(" ", i, self.link_dims[i])
                self.links[i] = new
                self.link_dims[i] = mapping[self.link_dims[i]] + offset


class TensorNetwork:

    @classmethod
    def validate_nodes(cls, nodes):
        for my_index, node in enumerate(nodes):
            for my_dim, neighbor_index in enumerate(node.links):
                their_dim = node.link_dims[my_dim]
                neighbor = nodes[neighbor_index]
                print(my_index, neighbor_index)
                print(their_dim)
                print(neighbor.links[their_dim])
                assert neighbor.links[their_dim] == my_index

    def __init__(self, nodes: list[Node]):
        TensorNetwork.validate_nodes(nodes)
        self.nodes = nodes
        self.active = np.full(len(self), True, dtype=bool)

    def __len__(self) -> int:
        return len(self.nodes)

    def links_of(self, i):
        return self.nodes[i].links

    def ranges_of(self, i):
        return self.nodes[i].dim_ranges

    def ndims_of(self, i):
        return self.nodes[i].ndims

    def link_dims_of(self, i):
        return self.nodes[i].link_dims

    def neighbors_of(self, i):
        return np.unique(self.links_of(i))

    @property
    def active_nodes(self):
        return np.flatnonzero(self.active)

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

    def merge_cost(self, i,  j):
        '''
        Cost of merge, including case of no common indices
        '''
        cost = np.prod(self.ranges_of(i)) * np.prod(self.ranges_of(j))
        mutual_ranges = self.mutual_ranges(i, j)
        if mutual_ranges.size != 0:
            cost /= np.prod(mutual_ranges)
        return cost

    def map_after_drop(self, i, j):
        orig_len = self.ndims_of(i)
        drop_pos = np.flatnonzero(self.links_of(i) == j)
        mask = np.ones(orig_len, dtype=bool)
        mask[drop_pos] = False
        mapping = np.full(orig_len, -1, dtype=int)
        mapping[mask] = np.cumsum(mask)[mask] - 1
        offset = np.sum(mask)
        print(mask)
        print(mapping, offset)
        return mapping, offset

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
        map_a, offset = self.map_after_drop(i, j)
        map_b, _ = self.map_after_drop(j, i)
        for neighbor in self.neighbors_of(i):
            if neighbor not in [-1, j]:
                print('neighbor = ', neighbor)
                self.nodes[neighbor].redirect(j, j,  # j, j is not a typo
                                              mapping=map_b, offset=offset)
                self.nodes[neighbor].redirect(i, j, mapping=map_a, offset=0)

        # Save result of merge to j according to rules of np.tensordot
        x = self.links_of(i)
        y = self.links_of(j)
        rg_x = self.ranges_of(i)
        rg_y = self.ranges_of(j)
        ld_x = self.link_dims_of(i)
        ld_y = self.link_dims_of(j)
        self.nodes[j].dim_ranges = np.concatenate([rg_x[x != j], rg_y[y != i]])
        self.nodes[j].link_dims = np.concatenate([ld_x[x != j], ld_y[y != i]])
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
