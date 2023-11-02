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
    max_bin = 5
    min_bin = 0
    bins = 5

    # Generic data
    truth = np.array([5, 8, 12, 6, 2])
    meas = np.array([6, 9, 13, 5, 3])
    response = np.array(
        [
            [1, 1, 0, 0, 0],
            [1, 2, 1, 0, 0],
            [0, 1, 3, 1, 0],
            [0, 0, 1, 3, 1],
            [0, 0, 0, 1, 2],
        ]
    )

    # Unfold with simulated annealing
    unfolder = QUnfoldQUBO(response=response, meas=meas, lam=0.1)
    unfolded_SA = unfolder.solve_simulated_annealing(num_reads=200)

    # Create results dir
    if not os.path.exists("img/examples/numpy_simulated_annealing"):
        os.makedirs("img/examples/numpy_simulated_annealing")

    # Plot information
    plotter = QUnfoldPlotter(
        response=response,
        measured=meas,
        truth=truth,
        unfolded=unfolded_SA,
        binning=np.linspace(min_bin, max_bin, bins + 1),
    )
    plotter.saveResponse("img/examples/numpy_simulated_annealing/response.png")
    plotter.savePlot("img/examples/numpy_simulated_annealing/comparison.png", "SA")


if __name__ == "__main__":
    main()
