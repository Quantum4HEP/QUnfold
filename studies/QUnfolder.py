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

# Standard modules
import os, sys

# Data science modules
import ROOT as r
import numpy as np

# Utils modules
from utils.custom_logger import INFO
from utils.helpers import load_data

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

        # Load histograms and response from file
        (
            truth,
            measured,
            response,
            binning,
        ) = load_data(distr)
        bins = int(binning[0])
        min_bin = int(binning[1])
        max_bin = int(binning[2])

        # Performing the unfolding with different methods
        for unf_type in ["simulated"]:

            # Compute the unfolding result
            print("- Unfolding with {} annealing...".format(unf_type))
            unfolder = QUnfoldQUBO(response, measured)
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
            
        print()
    print("Done")

if __name__ == "__main__":
    main()
