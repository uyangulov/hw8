import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from copy import deepcopy


def RZZ(theta):
    exponents = np.array([1, -1, -1, 1], dtype=np.complex128)
    diagonal = np.diag(np.exp(-1j * theta * exponents / 2))
    return diagonal.reshape((2, 2, 2, 2))


def RX(theta):
    X = np.array([
        [0, 1],
        [1, 0]
    ])
    return np.cos(theta / 2) * np.eye(2) - 1j * np.sin(theta/2) * X


class State:
    def __init__(self, n_qubits):
        self.state = np.ones((2 ** n_qubits), dtype=complex)  # |+> state
        self.state /= np.sqrt(2 ** n_qubits)
        self.state = np.reshape(self.state, [2]*n_qubits)

    def apply_1qubit(self, u, i):
        a = np.tensordot(u, self.state, axes=((1), (i)))
        self.state = np.moveaxis(a, 0, i)

    def apply_2qubit(self, u, i, j):
        a = np.tensordot(np.reshape(u, [2]*4),
                         self.state, axes=((2, 3), (i, j)))
        self.state = np.moveaxis(a, (0, 1), (i, j))

    def get_probs(self):
        return np.abs(self.state.reshape(-1))**2

    def run_qaoa(self, lambd, gammas, betas):
        n = lambd.shape[0]
        operations = []
        for layer, (gamma, beta) in enumerate(zip(gammas, betas)):
            rzz = RZZ(gamma)
            rx = RX(beta)
            for i in range(n):
                for j in range(i):
                    if lambd[i, j] == 1:
                        self.apply_2qubit(rzz, i, j)
            for i in range(n):
                self.apply_1qubit(rx, i)


class Node:

    def __init__(self, ndims, tensor=None, color="blue"):
        self.links = np.full(ndims, -1, dtype=int)
        self.link_dims = np.full(ndims, -1, dtype=int)
        self.tensor = tensor
        self.color = color

    def connect_to(self, neighbor, our_dim, their_dim):
        self.links[our_dim] = neighbor
        self.link_dims[our_dim] = their_dim

    def redirect(self, old, new, mapping, offset):
        '''
        If any connections to node with index 'old', redirect to index 'new'.
        Remap indices according to given index mapping and index offset
        '''
        mask = self.links == old
        self.links[mask] = new
        self.link_dims[mask] = mapping[self.link_dims[mask]] + offset


class TensorNetwork:

    def __init__(self, nodes: list[Node]):
        self.nodes = nodes
        self.active = np.full(len(self), True, dtype=bool)

    def __len__(self) -> int:
        return len(self.nodes)

    def links_of(self, i):
        return self.nodes[i].links

    def ndims_of(self, i):
        return self.nodes[i].links.size

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
        (zero if no such dimensions)
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
        '''
        Mapping between indices of i to indices of new tensor
        (After contracting with j)
        '''
        drop = self.links_of(i) == j
        mapping = np.cumsum(~drop) - 1
        mapping[drop] = -1
        offset = np.sum(~drop)
        return mapping, offset

    def merge_nodes(self, i: int, j: int, actually_contract=False) -> int:
        '''
        1) Merge nodes i and j
        2) Save result at index j (modifies state of network)
        3) Return cost of the merge
        '''
        # self.validate_merge(i, j) uncomment if checks necessary
        cost = self.merge_cost(i, j)

        if actually_contract:
            dims_ii = np.flatnonzero(self.links_of(i) == j)
            dims_jj = self.link_dims_of(i)[dims_ii]
            self.nodes[j].tensor = np.tensordot(
                self.nodes[i].tensor,
                self.nodes[j].tensor,
                axes=(dims_ii, dims_jj)
            )

        # map from indices of nodes i and j to nodes of contracted tensor
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

    def draw_network(self, h=5, w=7):
        plt.figure(figsize=(w, h))

        G = nx.MultiDiGraph()
        edge_labels = {}

        for i, node in enumerate(self.nodes):
            if not self.active[i]:
                continue
            G.add_node(i, color=node.color)
            for dim, j in enumerate(node.links):
                if j != -1:
                    G.add_edge(i, j)
                    edge_labels[(i, j)] = edge_labels.get((i, j), []) + [dim]

        pos = nx.spring_layout(G)
        node_colors = [G.nodes[n]['color'] for n in G.nodes]

        nx.draw(G, pos, with_labels=True,
                node_color=node_colors, node_size=800)
        ax = plt.gca()

        for (u, v), dims in edge_labels.items():
            x = (1 - 0.15) * pos[u][0] + 0.15 * pos[v][0]
            y = (1 - 0.15) * pos[u][1] + 0.15 * pos[v][1]
            ax.text(x, y, str(dims), color='red')

        plt.show()

    @classmethod
    def qaoa_like_from_params(cls,
                              lambd,
                              gammas,
                              betas,
                              generate_meas_layer=True):
        """
        Create QAOA-like tensor network with given parameters.

        Args:
            lambd: Coupling matrix.
            gammas: Array of gamma parameters.
            betas: Array of beta parameters.
            generate_meas_layer (bool, optional): Whether to include measurement layer. Defaults to True.
        """
        return cls._qaoa_like_generic(lambd,
                                      gammas.size,
                                      gammas,
                                      betas,
                                      generate_meas_layer)

    @classmethod
    def qaoa_like_tree_only(cls,
                            lambd,
                            n_layers,
                            generate_meas_layer=True):
        """
        Create QAOA-like tensor network without acutal tensors,
        but with info about tensor dims
        (used in contraction-order-finding algorithms)
        Args:
            lambd: Coupling matrix.
            n_layers (int): Number of layers.
            generate_meas_layer (bool, optional): Whether to include measurement layer. Defaults to True.
            TensorNetwork: Constructed QAOA-like tensor network structure.
        """
        return cls._qaoa_like_generic(lambd,
                                      n_layers,
                                      gammas=None,
                                      betas=None,
                                      generate_meas_layer=generate_meas_layer)

    @classmethod
    def _qaoa_like_generic(cls, lambd,
                           n_layers,
                           gammas,
                           betas,
                           generate_meas_layer):
        '''
        build qaoa like tensor net with tetris algorithm
        '''
        generate_tensors = (gammas is not None)
        n_qubits = lambd.shape[0]
        nodes = []
        # for every wire in quantum circuit, track last node on it
        last_node_on_wire = np.full(n_qubits, -1, dtype=int)
        # for every wire in quantum circuit, track last OUTPUT dim on it
        last_dim_on_wire = np.full(n_qubits, -1, dtype=int)

        def connect_to_wire(wire, dim):
            last = len(nodes) - 1
            nodes[last_node_on_wire[wire]].connect_to(
                our_dim=last_dim_on_wire[wire],
                neighbor=last,
                their_dim=dim
            )
            nodes[last].connect_to(
                our_dim=dim,
                neighbor=last_node_on_wire[wire],
                their_dim=last_dim_on_wire[wire],
            )

        # initial state tensors with |+> state
        plus = np.array([1, 1]) / np.sqrt(2) if generate_tensors else None
        for i in range(n_qubits):
            nodes.append(Node(1, color="red", tensor=plus))
            last_node_on_wire[i] = len(nodes) - 1
            last_dim_on_wire[i] = 0

        for layer in range(n_layers):
            rzz = RZZ(gammas[layer]) if generate_tensors else None
            rx = RX(betas[layer]) if generate_tensors else None
            # rzz layer
            for i in range(n_qubits):
                for j in range(i):
                    if lambd[i, j]:
                        nodes.append(Node(4, tensor=rzz))
                        connect_to_wire(wire=i, dim=0)
                        connect_to_wire(wire=j, dim=1)
                        last_node_on_wire[i] = len(nodes) - 1
                        last_node_on_wire[j] = len(nodes) - 1
                        last_dim_on_wire[i] = 2
                        last_dim_on_wire[j] = 3
            # rx layer
            for i in range(n_qubits):
                nodes.append(Node(2, color="green", tensor=rx))
                connect_to_wire(wire=i, dim=0)
                last_node_on_wire[i] = len(nodes) - 1
                last_dim_on_wire[i] = 1

        # Measure
        if generate_meas_layer:
            proj = tensor = np.array([1, 0]) if generate_tensors else None
            for i in range(n_qubits):
                nodes.append(Node(1, color="pink", tensor=proj))
                connect_to_wire(wire=i, dim=0)

        return TensorNetwork(nodes)


class TensorNetworkOptimizer:

    def __init__(self, network: TensorNetwork, log_file='tensor_optimization.log'):
        '''
        initialize optimizer with network (copy is made)
        '''
        self.network = deepcopy(network)
        self.current_sequence = []
        self.current_cost = 0

    def optimize_network(self, k: int):
        '''
        1) find the best k-step sequence
        2) apply its first step until fully contracted.
        3) repeat until fully contracted
        '''
        while len(self.network.active_nodes) > 1:

            best_sequence, best_cost = self.dfs(self.network,
                                                best_cost=float('inf'),
                                                sequence=[],
                                                cost=0,
                                                steps_left=k)

            first_step = best_sequence[0]
            node_i, node_j = first_step
            step_cost = self.network.merge_nodes(node_i, node_j)
            self.current_cost += step_cost
            self.current_sequence.append((node_i, node_j))
        return self.current_sequence, self.current_cost

    def dfs(self, network: TensorNetwork,
            best_cost: float,
            sequence: list,
            cost: float,
            steps_left: int):
        """
        dfs to find the best k-step contraction sequence.
        returns a tuple (best_sequence, best_cost)
        """
        best_sequence = None
        active = network.active_nodes
        for i, node_i in enumerate(active):
            for j in range(i):
                node_j = active[j]
                if steps_left == 1 or len(network.active_nodes) == 2:
                    step_cost = network.merge_cost(node_i, node_j)
                    child_seq = sequence + [(node_i, node_j)]
                    child_cost = cost + step_cost
                else:
                    network_copy = deepcopy(network)
                    step_cost = network_copy.merge_nodes(node_i, node_j)
                    child_seq, child_cost = self.dfs(network_copy,
                                                     best_cost,
                                                     sequence +
                                                     [(node_i, node_j)],
                                                     cost + step_cost,
                                                     steps_left - 1)
                if child_cost < best_cost:
                    best_cost = child_cost
                    best_sequence = child_seq
        return best_sequence, best_cost
