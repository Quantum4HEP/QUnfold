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
    Class used to perform the unfolding using QUBO problems solution.
    """

    def __init__(self, response, meas, lam=0.0):
        """
        Initialize the QUnfoldQUBO object.

        Parameters:
            response (numpy.ndarray): The response matrix.
            meas (numpy.ndarray): The measured distribution.
            lam (float or list/array, optional): The regularization parameter(s).
                If float, it represents a single regularization parameter (default is 0.0).
                If list/array, it represents a set of regularization parameters to be optimized.
        """

        self.R = response
        self.d = meas
        self.lam = lam

    @staticmethod
    def _get_laplacian(dim):
        """
        Calculate the Laplacian matrix.

        Parameters:
            dim (int): The dimension of the matrix.

        Returns:
            numpy.ndarray: The Laplacian matrix.
        """
        diag = np.array([-1] + [-2] * (dim - 2) + [-1])
        D = np.diag(diag).astype(float)
        diag1 = np.ones(dim - 1)
        D += np.diag(diag1, k=1) + np.diag(diag1, k=-1)
        return D

    def _define_variables(self):
        """
        Define the variables for the QUBO problem.

        Returns:
            list: List of encoded integer variables.
        """

        # Get largest power of 2 integer below the total number of entries
        n = int(2 ** np.floor(np.log2(sum(self.d)))) - 1

        # Encode integer variables using logarithmic binary encoding
        vars = [LogEncInteger(f"x{i}", value_range=(0, n)) for i in range(len(self.d))]
        return vars

    def _define_hamiltonian(self, x):
        """
        Define the Hamiltonian for the QUBO problem.

        Parameters:
            x (list): List of encoded integer variables.

        Returns:
            pyqubo.Expression: The Hamiltonian expression.
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
        Define the PyQUBO model for the QUBO problem.

        Returns:
            tuple: Labels for the variables and the PyQUBO model.
        """

        x = self._define_variables()
        h = self._define_hamiltonian(x)
        labels = [x[i].label for i in range(len(x))]
        model = h.compile()
        return labels, model

    def _lambda_optimization(self, annealer, *args):
        """
        Perform optimization of the regularization parameter. Still work in progress.

        Parameters:
            annealer (function): The function used to evaluate the unfolded distribution.
            *args: Additional arguments to be passed to the annealer function.

        Returns:
            numpy.ndarray: The best choice of the unfolded distribution.
        """

        if isinstance(self.lam, float):
            return annealer(*args)

        results = {}
        for regularization_param in self.lam:
            list_of_lam = self.lam
            self.lam = regularization_param
            result_temp = annealer(*args)
            energy = self.compute_energy(result_temp)
            results[tuple(result_temp)] = (self.lam, energy)
            self.lam = list_of_lam

        best_result = min(results, key=lambda k: results[k][1])
        return np.array(best_result)

    def solve_simulated_annealing(self, num_reads=100, seed=None):
        """
        Solve the QUBO problem using the Simulated Annealing sampler.

        Parameters:
            num_reads (int, optional): Number of reads for the sampler (default is 100).
            seed (int, optional): Seed for random number generation (default is None).

        Returns:
            numpy.ndarray: Array of solutions.
        """

        def simulated_solver(num_reads, seed):
            labels, model = self._define_pyqubo_model()
            sampler = SimulatedAnnealingSampler()
            sampleset = sampler.sample(model.to_bqm(), num_reads=num_reads, seed=seed)
            decoded_sampleset = model.decode_sampleset(sampleset)
            best_sample = min(decoded_sampleset, key=lambda s: s.energy)
            return np.array([best_sample.subh[label] for label in labels])

        result = self._lambda_optimization(simulated_solver, num_reads, seed)
        return result

    def solve_hybrid_sampler(self):
        """
        Solve the QUBO problem using the Leap Hybrid sampler.
        If the self.lam parameter is provided as a list, an optimization is performed in order to find the best result.

        Returns:
            numpy.ndarray: Array of solutions.
        """

        def hybrid_solver():
            labels, model = self._define_pyqubo_model()
            sampler = LeapHybridSampler()
            sampleset = sampler.sample(model.to_bqm())
            decoded_sampleset = model.decode_sampleset(sampleset)
            best_sample = min(decoded_sampleset, key=lambda s: s.energy)
            return np.array([best_sample.subh[label] for label in labels])

        result = self._lambda_optimization(hybrid_solver)
        return result

    def compute_energy(self, x):
        """
        Computes the energy of the system for the given solution.

        Args:
            x (numpy.ndarray): An array containing the solution to the QUBO problem.

        Returns:
            float: The computed energy for the given solution.
        """
        num_bits = int(np.floor(np.log2(sum(self.d))))
        x_binary = {}
        for i, entry in enumerate(x):
            bitstr = np.binary_repr(int(entry), width=num_bits)
            for j, bit in enumerate(bitstr[::-1]):
                x_binary[f"x{i}[{j}]"] = int(bit)
        _, model = self._define_pyqubo_model()
        return model.to_bqm().energy(x_binary)
