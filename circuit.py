import numpy as np
from definitions import RX, RZZ


class State:
    def __init__(self, n_qubits):
        self.state = np.ones((2 ** n_qubits), dtype=complex)  # |+>
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
