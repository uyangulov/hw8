from tensor_contraction import TensorNetwork
from copy import deepcopy


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
        if steps_left == 0 or len(network.active_nodes) <= 1:
            return sequence, cost

        best_sequence = None
        active = network.active_nodes

        for i, node_i in enumerate(active):
            for j in range(i):
                node_j = active[j]
                network_copy = deepcopy(network)
                step_cost = network_copy.merge_nodes(node_i, node_j)
                child_seq, child_cost = self.dfs(network_copy,
                                                 best_cost,
                                                 sequence + [(node_i, node_j)],
                                                 cost + step_cost,
                                                 steps_left - 1)

                if child_cost < best_cost:
                    best_cost = child_cost
                    best_sequence = child_seq
        return best_sequence, best_cost
