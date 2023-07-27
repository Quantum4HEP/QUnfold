#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ---------------------- Metadata ----------------------
#
# File name:  standard.py
# Author:     Gianluca Bianco (biancogianluca9@gmail.com)
# Date:       2023-06-16
# Copyright:  (c) 2023 Gianluca Bianco under the MIT license.

# Data science modules
import numpy as np

# QUnfold modules
from QUnfold.core import QUnfoldQUBO
from QUnfold.plot import QUnfoldPlotter


def main():

    # Load normal distribution data
    truth = np.loadtxt("data/normal/truth_bin_content.txt")
    measured = np.loadtxt("data/normal/meas_bin_content.txt")
    response = np.loadtxt("data/normal/response.txt")

    # Unfold with simulated annealing
    unfolder = QUnfoldQUBO(
        response,
        measured,
    )
    unfolder.solve_simulated_annealing(lam=0.1, num_reads=100)

    # Plot information
    plotter = QUnfoldPlotter(
        unfolder=unfolder, truth=truth, binning=np.linspace(-10, 10, 41)
    )
    plotter.saveResponse("img/examples/standard/response.png")
    plotter.savePlot(
        "img/examples/standard/comparison.png", "Simulated Annealing Unfolding"
    )


if __name__ == "__main__":
    main()
