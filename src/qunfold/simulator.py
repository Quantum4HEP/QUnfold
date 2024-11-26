import functools
import numpy as np
from scipy import sparse


identity = np.array([[1, 0], [0, 1]])
sigma_x = np.array([[0, 1], [1, 0]])
sigma_z = np.array([[1, 0], [0, -1]])


class IsingHamiltonianSimulator:
    def __init__(self, anneal_schedule=None, time_steps=100, dtype=np.float64):
        self.anneal_schedule = self.default_anneal_schedule if anneal_schedule is None else anneal_schedule
        self.time_steps = time_steps
        self.dtype = dtype

    @staticmethod
    def default_anneal_schedule(s):
        a = 1 - s
        b = s
        return a, b

    def get_H_init(self, num_qubits):
        size = 2**num_qubits
        H_init = sparse.csr_array((size, size), dtype=self.dtype)
        i_sparse = sparse.csc_array(identity, dtype=self.dtype)
        x_sparse = sparse.csc_array(sigma_x, dtype=self.dtype)
        for i in range(num_qubits):
            terms = [i_sparse] * num_qubits
            terms[i] = x_sparse
            H_init -= functools.reduce(lambda A, B: sparse.kron(A, B, format="csr"), terms)
        return H_init

    def get_H_final(self, num_qubits, h, J):
        size = 2**num_qubits
        diag = np.zeros(size, dtype=self.dtype)
        i_diag = np.diag(identity)
        z_diag = np.diag(sigma_z)
        for i in range(num_qubits):
            terms = [i_diag] * num_qubits
            terms[i] = z_diag
            diag += h[i] * functools.reduce(np.outer, terms).ravel()
        for i in range(num_qubits):
            for j in range(i + 1, num_qubits):
                terms = [i_diag] * num_qubits
                terms[i] = z_diag
                terms[j] = z_diag
                diag += J[i, j] * functools.reduce(np.outer, terms).ravel()
        H_final = sparse.dia_array((diag, [0]), shape=(size, size))
        return H_final

    def run(self, bqm):
        num_qubits = bqm.num_variables
        linear, quadratic, _ = bqm.to_ising()
        h = np.array([linear.get(i, 0) for i in range(num_qubits)])
        J = np.array([[quadratic.get((i, j), 0) for i in range(num_qubits)] for j in range(num_qubits)])
        H_init = self.get_H_init(num_qubits=num_qubits)
        H_final = self.get_H_final(num_qubits=num_qubits, h=h, J=J)
        time = np.linspace(start=0, stop=1, num=self.time_steps)
        H_ising_list = []
        for s in time:
            a, b = self.anneal_schedule(s=s)
            H_ising = -(a / 2) * H_init + (b / 2) * H_final
            H_ising_list.append(H_ising)
        return H_ising_list
