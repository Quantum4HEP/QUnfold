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

# My modules
import sys

# Data generation modules
import ROOT as r

sys.path.append(".")
from studies.functions.generator import generate
from studies.functions.ROOT_converter import TH1_to_array, TH2_to_array

# QUnfold modules
from QUnfold import QUnfoldQUBO
from QUnfold import QUnfoldPlotter

# RooUnfold settings
loaded_RooUnfold = r.gSystem.Load("HEP_deps/RooUnfold/libRooUnfold.so")
if not loaded_RooUnfold == 0:
    sys.exit(0)


def main():

    # Generate data
    bins = 40
    min_bin = -10
    max_bin = 10
    truth, meas, response = generate("normal", bins, min_bin, max_bin, 10000)
    truth = TH1_to_array(truth, overflow=False)
    meas = TH1_to_array(meas, overflow=False)
    response = TH2_to_array(response.Hresponse(), overflow=False)

    # Unfold with simulated annealing
    unfolder = QUnfoldQUBO(response, meas, lam=0.1)
    unfolded_SA = unfolder.solve_simulated_annealing(num_reads=200)

    # Plot information
    plotter = QUnfoldPlotter(
        response=response,
        measured=meas,
        truth=truth,
        unfolded=unfolded_SA,
        binning=np.linspace(min_bin, max_bin, bins + 1),
    )
    plotter.saveResponse("img/examples/standard/response.png")
    plotter.savePlot("img/examples/standard/comparison.png", "SA")


if __name__ == "__main__":
    main()
