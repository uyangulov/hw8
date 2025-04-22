import numpy as np
from __future__ import annotations
from typing import List, Dict


class Node:
    def __init__(self, out_edges: Dict[int, List[int]], tensor: np.ndarray):
        self.tensor = tensor
        self.out_edges = out_edges  # maps neighbor node ID -> list of connected tensor indices


class TensorNetwork:
    def __init__(self, nodes: Dict[Node]):
        self.nodes = nodes

    def _indices_after_drop(self, node_id: int, neighbor_id: int) -> np.ndarray:
        tensor_rank = self.nodes[node_id].tensor.ndim
        to_drop = self.nodes[node_id].out_edges[neighbor_id]
        remaining = np.setdiff1d(
            np.arange(tensor_rank), to_drop, assume_unique=True)
        reverse_map = -np.ones(tensor_rank, dtype=int)
        reverse_map[remaining] = np.arange(len(remaining))
        return reverse_map

    def _remap_edges(self, edges: Dict[int, List[int]], reverse_map: np.ndarray) -> Dict[int, List[int]]:
        remapped = {}
        for neighbor, indices in edges.items():
            remapped_indices = [reverse_map[i]
                                for i in indices if reverse_map[i] != -1]
            if remapped_indices:
                remapped[neighbor] = remapped_indices
        return remapped

    def _merge_edges(self,
                     a: int,
                     b: int,
                     edges_a: Dict[int, List[int]],
                     edges_b: Dict[int, List[int]],
                     map_a: np.ndarray,
                     map_b: np.ndarray,
                     offset: int) -> Dict[int, List[int]]:
        merged = {}
        for neighbor, indices in edges_a.items():
            if neighbor != b:
                merged[neighbor] = [map_a[i]
                                    for i in indices if map_a[i] != -1]

        for neighbor, indices in edges_b.items():
            if neighbor != a:
                remapped = [map_b[i] +
                            offset for i in indices if map_b[i] != -1]
                if neighbor in merged:
                    merged[neighbor].extend(remapped)
                else:
                    merged[neighbor] = remapped
        return merged

    def _update_neighbors(self, i: int, j: int, new_node_id: int):

        affected_neighbors = set(self.nodes[i].out_edges.keys()).union(
            self.nodes[j].out_edges.keys())

        for neighbor_id in affected_neighbors:
            if neighbor_id in [i, j]:
                continue

            neighbor = self.nodes[neighbor_id]
            updated_edges = []

            if i in neighbor.out_edges:
                updated_edges.extend(neighbor.out_edges.pop(i))
            if j in neighbor.out_edges:
                updated_edges.extend(neighbor.out_edges.pop(j))

            if updated_edges:
                if new_node_id in neighbor.out_edges:
                    neighbor.out_edges[new_node_id].extend(updated_edges)
                else:
                    neighbor.out_edges[new_node_id] = updated_edges

    def contract_tensors(self, i: int, j: int) -> int:
        if i > j:
            i, j = j, i

        node_i = self.nodes[i]
        node_j = self.nodes[j]

        contracted_i = node_i.out_edges[j]
        contracted_j = node_j.out_edges[i]

        map_i = self._indices_after_drop(i, j)
        map_j = self._indices_after_drop(j, i)
        offset = np.count_nonzero(map_i != -1)

        new_edges = self._merge_edges(
            i, j, node_i.out_edges, node_j.out_edges, map_i, map_j, offset)
        new_tensor = np.tensordot(
            node_i.tensor, node_j.tensor, axes=(contracted_i, contracted_j))
        
        self._update_neighbors(i, j, new_node_id=i)
        new_node = Node(out_edges=new_edges, tensor=new_tensor)
        self.nodes[i] = new_node
        del self.nodes[j]
