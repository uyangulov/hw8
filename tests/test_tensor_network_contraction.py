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
    node_A = Node(edges_A, tensor_A)

    tensor_B = np.random.rand(2, 2, 2, 2)
    edges_B = {0: [2, 3], 2: [1, 0]}
    node_B = Node(edges_B, tensor_B)

    tensor_C = np.random.rand(2, 2, 2, 2)
    edges_C = {1: [0, 2], 3: [1, 3]}
    node_C = Node(edges_C, tensor_C)

    tensor_D = np.random.rand(2, 2, 2, 2)
    edges_D = {2: [0, 2]}
    node_D = Node(edges_D, tensor_D)

    return TensorNetwork({0: node_A, 1: node_B, 2: node_C, 3: node_D})


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
    net = LinearChained

    net.contract_tensors(1, 2)

    assert 1 in net.nodes
    assert 2 not in net.nodes
    new_tensor = net.nodes[1].tensor
    assert isinstance(new_tensor, np.ndarray)
    assert new_tensor.ndim == 4 

    edges = net.nodes[1].out_edges
    assert 0 in edges  # connection to A should still exist
    assert 3 in edges  # connection to D should still exist
    assert len(edges[0]) > 0
    assert len(edges[3]) > 0
    
    
def test_contract_single_node_fails():
    tensor = np.random.rand(2, 2)
    node = Node({}, tensor)
    net = TensorNetwork({0: node})
    with pytest.raises(ValueError):
        net.contract_tensors(0, 1)  # node 1 does not exist

def test_contract_single_node_self():
    tensor = np.random.rand(2, 2)
    node = Node({}, tensor)
    net = TensorNetwork({0: node})
    result = net.contract_tensors(0, 0)
    assert result == 0
    assert 0 in net.nodes
    assert len(net.nodes) == 1

def test_contract_two_nodes_invalid_ids():
    tensor = np.random.rand(2, 2)
    node_a = Node({}, tensor)
    node_b = Node({}, tensor)
    net = TensorNetwork({0: node_a, 1: node_b})

    with pytest.raises(ValueError):
        net.contract_tensors(0, 2)

def test_contract_two_nodes_success():
    t = np.random.rand(2, 2)
    node_a = Node({1: [1]}, t)
    node_b = Node({0: [0]}, t)
    net = TensorNetwork({0: node_a, 1: node_b})

    result = net.contract_tensors(0, 1)
    assert result == 0
    assert 1 not in net.nodes
    assert isinstance(net.nodes[0].tensor, np.ndarray)
