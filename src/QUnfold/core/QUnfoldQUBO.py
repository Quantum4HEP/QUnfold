#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ---------------------- Metadata ----------------------
#
# File name:  QUnfoldQUBO.py
# Author:     Gianluca Bianco (biancogianluca9@gmail.com)
# Date:       2023-06-16
# Copyright:  (c) 2023 Gianluca Bianco under the MIT license.

import numpy as np
from pyqubo import LogEncInteger
from dwave.samplers import SimulatedAnnealingSampler


class QUnfoldQUBO:

    def __init__(self):
        pass

    @staticmethod
    def _get_laplacian(dim):
        diag = np.ones(dim, dtype=int) * -2
        ones = np.ones(dim-1, dtype=int)
        return np.diag(diag) + np.diag(ones, k=1) + np.diag(ones, k=-1)

    @staticmethod
    def _compute_linear(R, d):
        return -2. * np.matmul(np.transpose(R), d)

    @staticmethod
    def _compute_quadratic(R, G, lam):
        return np.matmul(np.transpose(R), R) + lam * np.matmul(np.transpose(G), G)

    def _get_pyqubo_model(self, response, data, lam):
        num_bins = len(data)
        num_entries = int(sum(data))
        labels = [f'x{i}' for i in range(num_bins)]
        # variables binary encoding
        x = [LogEncInteger(label=label, value_range=(0, num_entries))
             for label in labels]
        hamiltonian = 0
        # linear terms
        a = self._compute_linear(R=response, d=data)
        for i in range(len(x)):
            hamiltonian += a[i] * x[i]
        # quadratic terms
        G = self._get_laplacian(dim=num_bins)
        B = self._compute_quadratic(R=response, G=G, lam=lam)
        for i in range(len(x)):
            for j in range(len(x)):
                hamiltonian += B[i, j] * x[i] * x[j]
        model = hamiltonian.compile()
        return labels, model

    def run_simulated_annealing(self, response, data, lam=0.1, num_reads=100):
        labels, model = self._get_pyqubo_model(response, data, lam)
        sampler = SimulatedAnnealingSampler()
        sampleset = sampler.sample(model.to_bqm(), num_reads=num_reads)
        decoded_sampleset = self._model.decode_sampleset(sampleset)
        best_sample = min(decoded_sampleset, key=lambda s: s.energy)
        solution = np.array([best_sample.subh[label] for label in labels])
        return solution.astype(int)
