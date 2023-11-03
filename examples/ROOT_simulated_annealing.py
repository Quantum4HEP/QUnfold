#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ---------------------- Metadata ----------------------
#
# File name:  ROOT_simulated_annealing.py
# Author:     Gianluca Bianco (biancogianluca9@gmail.com)
# Date:       2023-06-16
# Copyright:  (c) 2023 Gianluca Bianco under the MIT license.

# STD modules
import sys, os

# Data science modules
import numpy as np
import ROOT as r

# My modules
sys.path.append(".")
from studies.functions.generator import generate

# QUnfold modules
from QUnfold import QUnfoldQUBO
from QUnfold import QUnfoldPlotter
from QUnfold.utility import TH1_to_array, TMatrix_to_array

# RooUnfold settings
loaded_RooUnfold = r.gSystem.Load("HEP_deps/RooUnfold/libRooUnfold.so")
if not loaded_RooUnfold == 0:
    sys.exit(0)


def main():

    # Data variables
    samples = 10000
    max_bin = 10
    min_bin = 0
    bins = 20
    bias = -0.13
    smearing = 0.21
    eff = 0.92

    # Generate data in ROOT format and convert
    truth, meas, response = generate(
        "breit-wigner", bins, min_bin, max_bin, samples, bias, smearing, eff
    )
    truth = TH1_to_array(truth, overflow=False)
    meas = TH1_to_array(meas, overflow=False)
    response = TMatrix_to_array(response.Mresponse(norm=True))

    # Unfold with simulated annealing
    unfolder = QUnfoldQUBO(response=response, meas=meas, lam=0.05)
    unfolded_SA = unfolder.solve_simulated_annealing(num_reads=100)

    # Create results dir
    if not os.path.exists("img/examples/ROOT_simulated_annealing"):
        os.makedirs("img/examples/ROOT_simulated_annealing")

    # Plot information
    plotter = QUnfoldPlotter(
        response=response,
        measured=meas,
        truth=truth,
        unfolded=unfolded_SA,
        binning=np.linspace(min_bin, max_bin, bins + 1),
    )
    plotter.saveResponse("img/examples/ROOT_simulated_annealing/response.png")
    plotter.savePlot("img/examples/ROOT_simulated_annealing/comparison.png", "SA")


if __name__ == "__main__":
    main()
