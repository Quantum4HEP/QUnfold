#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ---------------------- Metadata ----------------------
#
# File name:  QUnfoldQUBO.py
# Author:     Gianluca Bianco (biancogianluca9@gmail.com) & Simone Gasperini (simone.gasperini4@unibo.it)
# Date:       2023-06-16
# Copyright:  (c) 2023 Gianluca Bianco under the MIT license.

import numpy as np
from pyqubo import LogEncInteger
from dwave.samplers import SimulatedAnnealingSampler
from dwave.system import LeapHybridSampler


class QUnfoldQUBO:
    """
    Class used to perform the unfolding formulated as a QUBO problem.
    """

    def __init__(self, response, measured, lam=0.0):
        """
        Initialize the QUnfoldQUBO object.

        Parameters:
            response (numpy.ndarray): input response matrix.
            measured (numpy.ndarray): input measured distribution.
            lam (float, optional): regularization parameter (default is 0.0).
        """
        self.R = response
        self.d = measured
        self.lam = lam

    @staticmethod
    def _get_laplacian(dim):
        """
        Build the Laplacian matrix.

        Parameters:
            dim (int): dimension of the matrix.

        Returns:
            numpy.ndarray: Laplacian matrix.
        """
        diag = np.array([-1] + [-2] * (dim - 2) + [-1])
        D = np.diag(diag).astype(float)
        diag1 = np.ones(dim - 1)
        D += np.diag(diag1, k=1) + np.diag(diag1, k=-1)
        return D

    def _define_variables(self):
        """
        Define the list of variables for the QUBO problem.

        Returns:
            list: encoded integer variables.
        """
        # Get largest power of 2 integer below the total number of entries
        n = int(2 ** np.floor(np.log2(sum(self.d)))) - 1

        # Encode integer variables using logarithmic binary encoding
        vars = [LogEncInteger(f"x{i}", value_range=(0, n)) for i in range(len(self.d))]
        return vars

    def _define_hamiltonian(self, x):
        """
        Define the Hamiltonian expression for the QUBO problem.

        Parameters:
            x (list): encoded integer variables.

        Returns:
            pyqubo.Expression: Hamiltonian expression.
        """
        hamiltonian = 0
        dim = len(x)

        # Add linear terms
        a = -2 * (self.R.T @ self.d)
        for i in range(dim):
            hamiltonian += a[i] * x[i]

        # Add quadratic terms:
        G = self._get_laplacian(dim)
        B = (self.R.T @ self.R) + self.lam * (G.T @ G)
        for i in range(dim):
            for j in range(dim):
                hamiltonian += B[i, j] * x[i] * x[j]
        return hamiltonian

    def _define_pyqubo_model(self):
        """
        Define the PyQUBO model instance for the QUBO problem.

        Returns:
            tuple: labels for the variables and PyQUBO model instance.
        """
        x = self._define_variables()
        h = self._define_hamiltonian(x)
        labels = [x[i].label for i in range(len(x))]
        model = h.compile()
        return labels, model

    def solve_simulated_annealing(self, num_reads=100, seed=None):
        """
        Solve the unfolding QUBO problem using the Simulated Annealing sampler.

        Parameters:
            num_reads (int, optional): number of reads for the sampler (default is 100).
            seed (int, optional): seed for random number generation (default is None).

        Returns:
            numpy.ndarray: unfolded histogram.
        """
        labels, model = self._define_pyqubo_model()
        sampler = SimulatedAnnealingSampler()
        sampleset = sampler.sample(model.to_bqm(), num_reads=num_reads, seed=seed)
        decoded_sampleset = model.decode_sampleset(sampleset)
        best_sample = min(decoded_sampleset, key=lambda s: s.energy)
        return np.array([best_sample.subh[label] for label in labels])

    def solve_hybrid_sampler(self):
        """
        Solve the unfolding QUBO problem using the Leap Hybrid sampler.

        Returns:
            numpy.ndarray: unfolded histogram.
        """
        labels, model = self._define_pyqubo_model()
        sampler = LeapHybridSampler()
        sampleset = sampler.sample(model.to_bqm())
        decoded_sampleset = model.decode_sampleset(sampleset)
        best_sample = min(decoded_sampleset, key=lambda s: s.energy)
        return np.array([best_sample.subh[label] for label in labels])

    def compute_energy(self, x):
        """
        Compute the energy of the Hamiltoninan for the given input histogram.

        Args:
            x (numpy.ndarray): input histogram.

        Returns:
            float: energy for the given input.
        """
        num_bits = int(np.floor(np.log2(sum(self.d))))
        x_binary = {}
        for i, entry in enumerate(x):
            bitstr = np.binary_repr(int(entry), width=num_bits)
            for j, bit in enumerate(bitstr[::-1]):
                x_binary[f"x{i}[{j}]"] = int(bit)
        _, model = self._define_pyqubo_model()
        return model.to_bqm().energy(x_binary)
