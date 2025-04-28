from src.tensor_contraction import TensorNetwork, Node
import numpy as np
import logging


class TensorNetworkOptimizer:

    def __init__(self, network: TensorNetwork, log_file='tensor_optimization.log'):
        self.initial_network = network
        self.best_sequence = None
        self.best_cost = float('inf')

        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            filemode='w'
        )
        self.logger = logging.getLogger()

    def find_optimal_sequence(self, k: int):
        self.logger.info("Starting optimization search...")
        self.dfs_search(self.initial_network,
                        current_sequence=[],
                        current_cost=0,
                        steps_left=k)

        self.logger.info("\nOptimization complete!")
        self.logger.info(
            f"Best sequence found: {self._pretty_sequence(self.best_sequence)}")
        self.logger.info(f"Total cost: {self.best_cost}")
        return self.best_sequence, self.best_cost

    def dfs_search(self, network: TensorNetwork,
                   current_sequence: list,
                   current_cost: int,
                   steps_left: int):

        if steps_left == 0:
            if current_cost < self.best_cost:
                self.logger.info(
                    f"New best sequence found! Cost: {current_cost} (previous best: {self.best_cost})")
                self.best_cost = current_cost
                self.best_sequence = current_sequence.copy()
            return

        active_nodes = network.active_nodes

        self.logger.info(
            f"\nStep {len(current_sequence) + 1}/{len(current_sequence) + steps_left}:")
        self.logger.info(
            f"Current sequence: {self._pretty_sequence(current_sequence)}")
        self.logger.info(f"Current cost: {current_cost}")
        self.logger.info(f"Active nodes: {active_nodes.tolist()}")

        for i, node_i in enumerate(active_nodes):
            for j, node_j in enumerate(active_nodes[i+1:]):

                self.logger.info(
                    f"  Attempting merge: ({node_i}, {node_j})")

                network_copy = self._copy_network(network)
                try:
                    step_cost = network_copy.merge_nodes(node_i, node_j)
                    new_sequence = current_sequence + [(node_i, node_j)]
                    new_cost = current_cost + step_cost

                    self.logger.info(f"  Cost: {step_cost}")
                    self.dfs_search(network_copy, new_sequence,
                                    new_cost, steps_left-1)
                except ValueError as e:
                    self.logger.info(f"  Failed: {str(e)}")
                    continue

    def _copy_network(self, network: TensorNetwork) -> TensorNetwork:
        """Create a deep copy of the network"""
        new_nodes = []
        for node in network.nodes:
            new_node = Node(links=node.links.copy(),
                            link_dims=node.link_dims.copy())
            new_nodes.append(new_node)

        new_network = TensorNetwork(new_nodes)
        new_network.active = network.active.copy()
        return new_network

    def _pretty_sequence(self, sequence):
        if not sequence:
            return "[]"
        return ", ".join(f"({a},{b})" for a, b in sequence)
