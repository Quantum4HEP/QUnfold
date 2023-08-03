#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from pyqubo import LogEncInteger
from dwave.samplers import SimulatedAnnealingSampler


class QUnfoldQUBO:
    def __init__(self, response, meas):
        self.R = response
        self.d = meas

    @staticmethod
    def _get_laplacian(dim):
        diag = np.ones(dim) * -2
        ones = np.ones(dim - 1)
        D = np.diag(diag) + np.diag(ones, k=1) + np.diag(ones, k=-1)
        return D

    def _compute_linear(self):
        a = -2.0 * (self.R.T @ self.d)
        return a

    def _compute_quadratic(self, G, lam):
        B = (self.R.T @ self.R) + lam * (G.T @ G)
        return B

    def _get_pyqubo_model(self, lam):
        bins = len(self.d)
        n = int(sum(self.d))

        # Define integer variables ane perform binary encoding
        labels = [f"x{i}" for i in range(bins)]
        x = [LogEncInteger(label, value_range=(0, n)) for label in labels]

        hamiltonian = 0
        # Compute linear terms
        a = self._compute_linear()
        for i in range(len(x)):
            hamiltonian += a[i] * x[i]

        # Compute quadratic terms
        G = self._get_laplacian(dim=bins)
        B = self._compute_quadratic(G, lam)
        for i in range(len(x)):
            for j in range(len(x)):
                hamiltonian += B[i, j] * x[i] * x[j]

        model = hamiltonian.compile()
        return labels, model

    def solve_simulated_annealing(self, lam=0.1, num_reads=100):
        labels, model = self._get_pyqubo_model(lam)
        sampler = SimulatedAnnealingSampler()
        sampleset = sampler.sample(model.to_bqm(), num_reads=num_reads)
        decoded_sampleset = model.decode_sampleset(sampleset)
        best_sample = min(decoded_sampleset, key=lambda s: s.energy)
        return np.array([best_sample.subh[label] for label in labels])
