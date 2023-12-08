#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ---------------------- Metadata ----------------------
#
# File name:  simneal_example.py
# Author:     Gianluca Bianco (biancogianluca9@gmail.com) & Simone Gasperini (simone.gasperini4@unibo.it)
# Date:       2023-11-02
# Copyright:  (c) 2023 Gianluca Bianco under the MIT license.

import numpy as np
from QUnfold import QUnfoldQUBO, QUnfoldPlotter
from QUnfold.utility import normalize_response


# Set parameters for data generation
num_entries = 40000
num_bins = 16
min_bin = -8.0
max_bin = 8.0
bins = np.linspace(min_bin, max_bin, num_bins + 1)

seed = 42
np.random.seed(seed)
mean = 0.0
std = 2.6
mean_smear = -0.3
std_smear = 0.5

# Generate and normalize response matrix
mc_data = np.random.normal(loc=mean, scale=std, size=num_entries)
print(mc_data)
reco_data = mc_data + np.random.normal(
    loc=mean_smear, scale=std_smear, size=num_entries
)
response, _, _ = np.histogram2d(reco_data, mc_data, bins=bins)
mc_truth, _ = np.histogram(mc_data, bins=bins)
print(mc_truth)
response = normalize_response(response, mc_truth)

# Generate random true data
true_data = np.random.normal(loc=mean, scale=std, size=num_entries)

# Apply gaussian smearing to get measured data
meas_data = true_data + np.random.normal(
    loc=mean_smear, scale=std_smear, size=num_entries
)

# Generate truth and measured histograms
true, _ = np.histogram(true_data, bins=bins)
measured, _ = np.histogram(meas_data, bins=bins)

# Run simulated annealing to solve QUBO problem
unfolder = QUnfoldQUBO(response, measured, lam=0.05)
unfolder.initialize_qubo_model(optimize_vars_range=False)
unfolded = unfolder.solve_simulated_annealing(num_reads=100, seed=seed)

# Plot unfolding result
plotter = QUnfoldPlotter(
    response=response, measured=measured, truth=true, unfolded=unfolded, binning=bins
)
plotter.plotResponse()
plotter.plot()

plotter.saveResponse("examples/simneal_response.png")
plotter.savePlot("examples/simneal_result.png", method="SA")
