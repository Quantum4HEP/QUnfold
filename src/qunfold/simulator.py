import functools
import numpy as np


identity = np.array([[1, 0], [0, 1]])
sigma_x = np.array([[0, 1], [1, 0]])
sigma_z = np.array([[1, 0], [0, -1]])


class IsingHamiltonianSimulator:
    def __init__(self, anneal_schedule=None, time_steps=100):
        self.anneal_schedule = self.default_anneal_schedule if anneal_schedule is None else anneal_schedule
        self.time_steps = time_steps

    @staticmethod
    def default_anneal_schedule(s):
        a = 1 - s
        b = s
        return a, b

    @staticmethod
    def get_H_init(num_qubits):
        H_init = np.zeros(shape=(2**num_qubits, 2**num_qubits))
        for i in range(num_qubits):
            terms = [identity] * num_qubits
            terms[i] = sigma_x
            H_init -= functools.reduce(np.kron, terms)
        return H_init

    @staticmethod
    def get_H_final(num_qubits, h, J):
        H_final = np.zeros(shape=(2**num_qubits, 2**num_qubits))
        for i in range(num_qubits):
            terms = [identity] * num_qubits
            terms[i] = sigma_z
            H_final += h[i] * functools.reduce(np.kron, terms)
        for i in range(num_qubits):
            for j in range(i + 1, num_qubits):
                terms = [identity] * num_qubits
                terms[i] = sigma_z
                terms[j] = sigma_z
                H_final += J[i, j] * functools.reduce(np.kron, terms)
        return H_final

    def run(self, bqm):
        num_qubits = bqm.num_variables
        linear, quadratic, _ = bqm.to_ising()
        h = np.array([linear.get(i, 0) for i in range(num_qubits)])
        J = np.array([[quadratic.get((i, j), 0) for i in range(num_qubits)] for j in range(num_qubits)])
        H_init = self.get_H_init(num_qubits=num_qubits)
        H_final = self.get_H_final(num_qubits=num_qubits, h=h, J=J)
        time = np.linspace(start=0, stop=1, num=self.time_steps)
        for s in time:
            a, b = self.anneal_schedule(s=s)
            H_ising = (a / 2) * H_init + (b / 2) * H_final
        # What should we return here?
