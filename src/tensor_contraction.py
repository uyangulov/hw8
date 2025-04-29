import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class Node:

    def __init__(self, links, link_dims):
        '''
        links[i] is index of neighbor of node connected to i-th dimension
        link_dims[i] is neighbor's dimension connected to dimension i 
        '''
        self.links = np.array(links, dtype=int)
        self.link_dims = np.array(link_dims, dtype=int)

    @property
    def ndims(self):
        return self.links.size

    def redirect(self, old, new, mapping, offset):
        '''
        If any connections to node with index 'old', redirect to index 'new'
        '''
        mask = self.links == old
        self.links[mask] = new
        self.link_dims[mask] = mapping[self.link_dims[mask]] + offset


class Edge:
    def __init__(self, nodes, dims):
        self.nodes = nodes
        self.dims = dims


class TensorNetwork:

    def __init__(self, nodes: list[Node]):
        self.nodes = nodes
        self.active = np.full(len(self), True, dtype=bool)

    @classmethod
    def from_edges_and_nodes(cls, nodes, edges):

        tn = TensorNetwork(nodes)

        for edge in edges:
            node1, node2 = edge.nodes
            dim1, dim2 = edge.dims

            # dim1 of node1 is connected to dim2 of node2
            tn.nodes[node1].links[dim1] = node2
            tn.nodes[node1].link_dims[dim1] = dim2

            # dim1 of node1 is connected to dim2 of node2
            tn.nodes[node2].links[dim2] = node1
            tn.nodes[node2].link_dims[dim2] = dim1
        return tn

    def __len__(self) -> int:
        return len(self.nodes)

    def links_of(self, i):
        return self.nodes[i].links

    def ndims_of(self, i):
        return self.nodes[i].ndims

    def link_dims_of(self, i):
        return self.nodes[i].link_dims

    def neighbors_of(self, i):
        return np.unique(self.links_of(i))

    @property
    def active_nodes(self):
        return np.flatnonzero(self.active)

    def n_mutual_dims(self, i, j):
        '''
        Return number of dimensions to be merged with node j
        (0 if no such dimensions)
        '''
        return np.sum(self.links_of(i) == j)

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
        exponent = self.ndims_of(i) + self.ndims_of(j)
        # if any common dims, they were accounted twise, so subtract
        exponent -= self.n_mutual_dims(i, j)
        return (1 << exponent)

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
        2) Save result at index j (modifies state of network)
        3) Return cost of the merge
        '''
        self.validate_merge(i, j)
        cost = self.merge_cost(i, j)

        # map from indices of nodes i and j to nodes of contracted tensors
        map_a, offset = self.map_after_drop(i, j)
        map_b, _ = self.map_after_drop(j, i)

        # update connections of j-th node neighbors
        for neighbor in self.neighbors_of(j):
            if neighbor not in [-1, i]:
                self.nodes[neighbor].redirect(j, j,  # j, j is not a typo
                                              mapping=map_b, offset=offset)

        # update connections of i-th node neighbors and reassign them to j
        for neighbor in self.neighbors_of(i):
            if neighbor not in [-1, j]:
                self.nodes[neighbor].redirect(i, j, mapping=map_a, offset=0)

        # Save result of merge to j according to rules of np.tensordot
        x, y = self.links_of(i), self.links_of(j)
        lx, ly = self.link_dims_of(i), self.link_dims_of(j)
        self.nodes[j].link_dims = np.concatenate([lx[x != j], ly[y != i]])
        self.nodes[j].links = np.concatenate([x[x != j], y[y != i]])
        self.active[i] = False
        return cost

    def draw_network(self, title='Tensor Network'):

        G = nx.MultiDiGraph()
        edge_index_map = {}

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
