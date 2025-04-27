from tensor_contraction import TensorNetwork, Node
import numpy as np


class TensorNetworkOptimizer:

    def __init__(self, network: TensorNetwork):
        self.initial_network = network
        self.best_sequence = None
        self.best_cost = float('inf')

    def find_optimal_sequence(self, k: int):
        self.dfs_search(self.initial_network,
                        current_sequence=[],
                        current_cost=0,
                        steps_left=k)
        return self.best_sequence, self.best_cost

    def dfs_search(self, network: TensorNetwork,
                   current_sequence: list,
                   current_cost: int,
                   steps_left: int):

        if steps_left == 0:
            if current_cost < self.best_cost:
                self.best_cost = current_cost
                self.best_sequence = current_sequence.copy()
            return

        active_nodes = np.where(network.active)

        for i, node_i in enumerate(active_nodes):
            for j, node_j in enumerate(active_nodes[i+1:]):
                network_copy = self._copy_network(network)
                try:
                    step_cost = network_copy.merge_nodes(node_i, node_j)
                    new_sequence = current_sequence + [(node_i, node_j)]
                    new_cost = current_cost + step_cost
                    self._dfs_search(network_copy, new_sequence,
                                     new_cost, steps_left-1)
                except ValueError:
                    continue

    def _copy_network(self, network: TensorNetwork) -> TensorNetwork:
        new_nodes = []
        for node in network.nodes:
            new_node = Node(node.out_dims.copy())
            new_nodes.append(new_node)

        new_network = TensorNetwork(new_nodes)
        new_network.active = network.active.copy()
        return new_network
