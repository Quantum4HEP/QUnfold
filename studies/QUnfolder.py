#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ---------------------- Metadata ----------------------
#
# File name:  unfolding.py
# Author:     Gianluca Bianco (biancogianluca9@gmail.com)
# Date:       2023-07-27
# Copyright:  (c) 2023 Gianluca Bianco under the MIT license.

# Input variables
distributions=["double-peaked"]
samples=10000
max_bin=10
min_bin=-10
bins=40

# Standard modules
import os, sys

# Data science modules
import ROOT as r
import numpy as np

# Utils modules
from functions.custom_logger import INFO
from functions.generator import generate_standard, generate_double_peaked, generate
from functions.ROOT_converter import (
    TH1_to_array,
    TH2_to_array
)

# QUnfold modules
sys.path.append("../src")
from QUnfold import QUnfoldQUBO, QUnfoldPlotter


def main():
    
    # Iterate over distributions
    for distr in distributions:
        INFO("Unfolding the {} distribution".format(distr))

        # Create dirs
        if not os.path.exists("../img/QUnfold/{}".format(distr)):
            os.makedirs("../img/QUnfold/{}".format(distr))
        if not os.path.exists("output/QUnfold/{}".format(distr)):
            os.makedirs("output/QUnfold/{}".format(distr))

        # Generating the distribution
        truth, meas, response = generate(distr, bins, min_bin, max_bin, samples, overflow=True)
        truth = TH1_to_array(truth, overflow=True)        
        meas = TH1_to_array(meas, overflow=True)
        response = TH2_to_array(response.Hresponse(), overflow=True)
        
        # Performing the unfolding with different methods
        for unf_type in ["simulated"]:

            # Compute the unfolding result
            print("- Unfolding with {} annealing...".format(unf_type))
            unfolder = QUnfoldQUBO(response, meas)
            unfolded = unfolder.solve_simulated_annealing(lam=0.1, num_reads=100)

            # Save unfolded data
            np.savetxt(
                "output/QUnfold/{}/unfolded_{}_bin_contents.txt".format(
                    distr, unf_type
                ),
                unfolded,
            )

            # Plot results
            plotter = QUnfoldPlotter(
                unfolder=unfolder,
                truth=truth,
                binning=np.linspace(min_bin, max_bin, bins + 1),
            )
            if unf_type == "simulated":
                title = "Simulated Annealing Unfolding"
            plotter.savePlot(
                "../img/QUnfold/{}/unfolded_{}.png".format(distr, unf_type), title
            )
            plotter.saveResponse(
                "../img/QUnfold/{}/response.png".format(distr, unf_type)
            )
            
        print()
    print("Done")

if __name__ == "__main__":
    main()
