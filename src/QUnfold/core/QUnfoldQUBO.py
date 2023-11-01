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

    def __init__(self, response, meas, lam=0.0, reg="laplace"):
        """
        Initialize the QUnfoldQUBO object.

        Parameters:
            response (numpy.ndarray): The response matrix.
            meas (numpy.ndarray): The measured distribution.
            lam (float, optional): The regularization parameter (default is 0.0).
            reg (string, optional): The regularization operator (default is "laplace"). Possible choices: "laplace", "cowan".
        """

        self.R = response
        self.d = meas
        self.lam = lam
        self.reg = reg

        # Normalize the response
        if not self._is_normalized(self.R):
            self.R = self._normalize(self.R)

    @staticmethod
    def _is_normalized(matrix):
        """
        Check if the matrix is normalized.

        Parameters:
            matrix (numpy.ndarray): The matrix to check.

        Returns:
            bool: True if the matrix is normalized, False otherwise.
        """

        return np.allclose(np.sum(matrix, axis=1), 1)

    @staticmethod
    def _normalize(matrix):
        """
        Normalize the matrix.

        Parameters:
            matrix (numpy.ndarray): The matrix to normalize.

        Returns:
            numpy.ndarray: The normalized matrix.
        """

        row_sums = np.sum(matrix, axis=1)
        mask = np.nonzero(row_sums)
        norm_matrix = np.copy(matrix)
        norm_matrix[mask] /= row_sums[mask][:, np.newaxis]
        return norm_matrix

    @staticmethod
    def _get_laplacian(dim):
        """
        Calculate the Laplacian matrix.

        Parameters:
            dim (int): The dimension of the matrix.

        Returns:
            numpy.ndarray: The Laplacian matrix.
        """

        diag = np.ones(dim) * -2
        ones = np.ones(dim - 1)
        D = np.diag(diag) + np.diag(ones, k=1) + np.diag(ones, k=-1)
        return D

    @staticmethod
    def _get_cowan_matrix(dim):
        """
        Generates the Cowan matrix used in statistical data analysis. The Cowan matrix is constructed according to formula (11.48) in Glen Cowan's
        book "Statistical Data Analysis".

        Parameters:
            dim (int): The dimension of the matrix.

        Returns:
            numpy.ndarray: A 2-dimensional array representing the Cowan matrix.
        """

        diag = np.array([1, 5] + [6] * (dim - 4) + [5, 1])
        G = np.diag(diag).astype(float)
        diag1 = np.array([-2] + [-4] * (dim - 3) + [-2])
        G += np.diag(diag1, k=1) + np.diag(diag1, k=-1)
        diag2 = np.ones(dim - 2)
        G += np.diag(diag2, k=2) + np.diag(diag2, k=-2)
        return G

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

        # Add quadratic terms
        if self.reg == "laplace":
            G = self._get_laplacian(dim)
        elif self.reg == "cowan":
            G = self._get_cowan_matrix(dim)
        else:
            raise ValueError(
                'The inserted regularization matrix "{}" is not valid!'.format(self.reg)
            )
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

    def solve_simulated_annealing(self, num_reads=100, seed=None):
        """
        Solve the QUBO problem using the Simulated Annealing sampler.

        Parameters:
            num_reads (int, optional): Number of reads for the sampler (default is 100).
            seed (int, optional): Seed for random number generation (default is None).

        Returns:
            numpy.ndarray: Array of solutions.
        """

        labels, model = self._define_pyqubo_model()
        sampler = SimulatedAnnealingSampler()
        sampleset = sampler.sample(model.to_bqm(), num_reads=num_reads, seed=seed)
        decoded_sampleset = model.decode_sampleset(sampleset)
        best_sample = min(decoded_sampleset, key=lambda s: s.energy)
        return np.array([best_sample.subh[label] for label in labels])

    def solve_hybrid_sampler(self):
        """
        Solve the QUBO problem using the Leap Hybrid sampler.

        Returns:
            numpy.ndarray: Array of solutions.
        """

        labels, model = self._define_pyqubo_model()
        sampler = LeapHybridSampler()
        sampleset = sampler.sample(model.to_bqm())
        decoded_sampleset = model.decode_sampleset(sampleset)
        best_sample = min(decoded_sampleset, key=lambda s: s.energy)
        return np.array([best_sample.subh[label] for label in labels])
