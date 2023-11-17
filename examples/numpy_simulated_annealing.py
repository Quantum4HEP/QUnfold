#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ---------------------- Metadata ----------------------
#
# File name:  numpy_simulated_annealing.py
# Author:     Gianluca Bianco (biancogianluca9@gmail.com)
# Date:       2023-11-02
# Copyright:  (c) 2023 Gianluca Bianco under the MIT license.

# STD modules
import os

# Data science modules
import numpy as np

# QUnfold modules
from QUnfold import QUnfoldQUBO
from QUnfold import QUnfoldPlotter


def main():
    # Data variables
    num_entries = 4000
    num_bins = 16
    min_bin = -8.0
    max_bin = 8.0

    # Generate random true data
    true_data = np.random.normal(loc=0.0, scale=1.8, size=num_entries)

    # Apply gaussian smearing to get measured data
    smearing = np.random.normal(loc=-0.3, scale=0.5, size=num_entries)
    meas_data = true_data + smearing

    # Generate truth and measured histograms
    bins = np.linspace(min_bin, max_bin, num_bins + 1)
    truth, _ = np.histogram(true_data, bins=bins)
    measured, _ = np.histogram(meas_data, bins=bins)

    # Generate and normalize response matrix
    response, _, _ = np.histogram2d(meas_data, true_data, bins=bins)
    response /= truth + 1e-6

    # Unfold with simulated annealing
    unfolder = QUnfoldQUBO(response=response, measured=measured, lam=0.01)
    unfolder.initialize_qubo_model()
    unfolded_SA = unfolder.solve_simulated_annealing(num_reads=100)

    # Create results dir
    if not os.path.exists("img/examples/numpy_simulated_annealing"):
        os.makedirs("img/examples/numpy_simulated_annealing")

    # Plot information
    plotter = QUnfoldPlotter(
        response=response,
        measured=measured,
        truth=truth,
        unfolded=unfolded_SA,
        binning=bins,
    )
    plotter.saveResponse("img/examples/numpy_simulated_annealing/response.png")
    plotter.savePlot("img/examples/numpy_simulated_annealing/comparison.png", "SA")


if __name__ == "__main__":
    main()
