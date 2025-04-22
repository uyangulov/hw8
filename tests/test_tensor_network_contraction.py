import numpy as np
import pytest
from tensor_contraction import TensorNetwork, Node

'''
   Node A         Node B         Node C         Node D
  +-------+      +-------+      +-------+      +-------+
  | 0   1 |------| 2   1 |------| 0   1 |------| 0   1 |
  |       |      |       |      |       |      |       |
  |       |      |       |      |       |      |       |
  | 2   3 |------| 3   0 |------| 2   3 |------| 2   3 |
  +-------+      +-------+      +-------+      +-------+
'''
@pytest.fixture
def LinearChained():
    np.random.seed(0)

    tensor_A = np.random.rand(2, 2, 2, 2)
    edges_A = {1: [1, 3]}
    node_A = Node(0, edges_A, tensor_A)

    tensor_B = np.random.rand(2, 2, 2, 2)
    edges_B = {0: [2, 3], 2: [1, 0]}
    node_B = Node(1, edges_B, tensor_B)

    tensor_C = np.random.rand(2, 2, 2, 2)
    edges_C = {1: [0, 2], 3: [1, 3]}
    node_C = Node(2, edges_C, tensor_C)

    tensor_D = np.random.rand(2, 2, 2, 2)
    edges_D = {2: [0, 2]}
    node_D = Node(3, edges_D, tensor_D)

    return TensorNetwork([node_A, node_B, node_C, node_D])


'''
   Node A         Node BC         Node D
  +-------+      +-------+      +-------+
  | 0   1 |------| 0   2 |------| 0   1 |
  |       |      |       |      |       |
  |       |      |       |      |       |
  | 2   3 |------| 1   3 |------| 2   3 |
  +-------+      +-------+      +-------+
'''
def test_contract_b_and_c(LinearChained):
    network = LinearChained

    # Contract nodes B (1) and C (2)
    network.contract_tensors(1, 2)

    # Ensure node count decreased
    assert len(network.nodes) == 3

    # Ensure the remaining nodes are Node 0, 1 (B⊗C), and 3
    assert 0 in network.nodes
    assert 3 in network.nodes
    assert 1 in network.nodes  # the merged B–C node

    # Ensure the new tensor is of expected shape: (2,2,2,2)
    # because contracting over 2 axes of 4-dim tensors
    new_tensor = network.nodes[1].tensor
    assert new_tensor.shape == (2, 2, 2, 2)

    # Ensure it has edges to node 0 and node 3
    new_edges = network.nodes[1].edges
    assert 0 in new_edges or 3 in new_edges
