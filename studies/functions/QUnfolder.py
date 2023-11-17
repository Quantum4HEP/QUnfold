#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ---------------------- Metadata ----------------------
#
# File name:  QUnfold.py
# Author:     Gianluca Bianco (biancogianluca9@gmail.com)
# Date:       2023-07-27
# Copyright:  (c) 2023 Gianluca Bianco under the MIT license.

# Standard modules
import os

# Data science modules
import numpy as np

# QUnfold modules
from QUnfold import QUnfoldQUBO, QUnfoldPlotter


def QUnfold_unfolder_and_plot(
    unf_type, response, measured, truth, distr, bins, min_bin, max_bin
):
    """
    Unfolds the measured data using QUnfold.

    Args:
        unf_type (str): The type of unfolding method to use (e.g., "SA" for Simulated Annealing).
        response (numpy.Array): The response matrix describing the detector response.
        measured (numpy.Array): The measured data to be unfolded.
        truth (numpy.Array): The true underlying data (for comparison purposes).
        distr (str): The distribution or category name for organizing the output plots.
        bins (int): The number of bins to use in the unfolding process.
        min_bin (float): The minimum value of the first bin.
        max_bin (float): The maximum value of the last bin.

    Returns:
        numpy.Array: The unfolded data obtained from the unfolding process.
    """

    # Create dirs
    if not os.path.exists("../img/QUnfold/{}".format(distr)):
        os.makedirs("../img/QUnfold/{}".format(distr))

    # Unfolder
    unfolder = QUnfoldQUBO(response, measured, lam=0.05)
    unfolded = None

    # Unfold with simulated annealing
    if unf_type == "SA":
        unfolded = unfolder.solve_simulated_annealing(num_reads=100)
        plotter = QUnfoldPlotter(
            response=response,
            measured=measured,
            truth=truth,
            unfolded=unfolded,
            binning=np.linspace(min_bin, max_bin, bins + 1),
        )
        plotter.savePlot("../img/QUnfold/{}/unfolded_SA.png".format(distr), "SA")
        plotter.saveResponse("../img/QUnfold/{}/response.png".format(distr))
        print(
            "The png file ../img/QUnfold/{}/unfolded_SA.png has been created".format(
                distr
            )
        )

    # Unfold with hybrid solver
    elif unf_type == "HYB":
        unfolded = unfolder.solve_hybrid_sampler()
        plotter = QUnfoldPlotter(
            response=response,
            measured=measured,
            truth=truth,
            unfolded=unfolded,
            binning=np.linspace(min_bin, max_bin, bins + 1),
        )
        plotter.savePlot("../img/QUnfold/{}/unfolded_HYB.png".format(distr), "HYB")
        plotter.saveResponse("../img/QUnfold/{}/response.png".format(distr))
        print(
            "The png file ../img/QUnfold/{}/unfolded_HYB.png has been created".format(
                distr
            )
        )

    return unfolded
