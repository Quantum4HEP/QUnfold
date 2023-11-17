#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ---------------------- Metadata ----------------------
#
# File name:  analysis.py
# Author:     Gianluca Bianco (biancogianluca9@gmail.com)
# Date:       2023-08-02
# Copyright:  (c) 2023 Gianluca Bianco under the MIT license.

# STD modules
import sys

# Data science modules
import ROOT as r

# My modules
from functions.custom_logger import INFO, ERROR
from functions.generator import generate
from functions.RooUnfold import (
    RooUnfold_unfolder,
    RooUnfold_plot,
    RooUnfold_plot_response,
)
from functions.QUnfolder import QUnfold_unfolder_and_plot
from functions.comparisons import plot_comparisons

# QUnfold modules
from QUnfold.utility import TH1_to_array, TMatrix_to_array

# RooUnfold settings
loaded_RooUnfold = r.gSystem.Load("../HEP_deps/RooUnfold/libRooUnfold.so")
if not loaded_RooUnfold == 0:
    ERROR("RooUnfold not found!")
    sys.exit(0)


# Input variables
distributions = ["normal", "gamma", "exponential", "breit-wigner", "double-peaked"]
samples = 10000
max_bin = 10
min_bin = 0
bins = 20
bias = -0.13
smearing = 0.21
eff = 0.92


def main():
    # Iterate over distributions
    for distr in distributions:
        # Generate data
        INFO("Unfolding the {} distribution".format(distr))
        truth, meas, response = generate(
            distr, bins, min_bin, max_bin, samples, bias, smearing, eff
        )

        ########################## Classic ###########################

        # RooUnfold settings
        r_response = response
        RooUnfold_plot_response(r_response, distr)
        r_response.UseOverflow(False)

        # Matrix inversion (MI)
        unfolded_MI = RooUnfold_unfolder("MI", r_response, meas)
        RooUnfold_plot(truth, meas, unfolded_MI, distr)

        # Iterative Bayesian unfolding (IBU)
        unfolded_IBU = RooUnfold_unfolder("IBU", r_response, meas)
        RooUnfold_plot(truth, meas, unfolded_IBU, distr)

        # Tikhonov unfolding (SVD)
        unfolded_SVD = RooUnfold_unfolder("SVD", r_response, meas)
        RooUnfold_plot(truth, meas, unfolded_SVD, distr)

        ########################## Quantum ###########################

        # QUnfold settings
        truth = TH1_to_array(truth, overflow=False)
        meas = TH1_to_array(meas, overflow=False)
        response = TMatrix_to_array(response.Mresponse(norm=True))

        # Simulated annealing (SA)
        unfolded_SA = QUnfold_unfolder_and_plot(
            "SA", response, meas, truth, distr, bins, min_bin, max_bin
        )

        # Hybrid solver (HYB)
        unfolded_HYB = QUnfold_unfolder_and_plot(
            "HYB", response, meas, truth, distr, bins, min_bin, max_bin
        )

        ########################## Compare ###########################

        # Comparison settings
        data = {
            "IBU4": TH1_to_array(unfolded_IBU, overflow=False),
            "MI": TH1_to_array(unfolded_MI, overflow=False),
            "SVD": TH1_to_array(unfolded_SVD, overflow=False),
            "SA": unfolded_SA,
            "HYB": unfolded_HYB,
        }

        # Plot comparisons
        plot_comparisons(data, distr, truth, bins, min_bin, max_bin)
        print("Done", end="\n\n")


if __name__ == "__main__":
    main()
