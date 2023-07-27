#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ---------------------- Metadata ----------------------
#
# File name:  unfolding.py
# Author:     Gianluca Bianco (biancogianluca9@gmail.com)
# Date:       2023-07-27
# Copyright:  (c) 2023 Gianluca Bianco under the MIT license.

# Standard modules
import argparse as ap
import sys, os

# Data science modules
import ROOT as r
import numpy as np
import matplotlib.pyplot as plt

# Utils modules
sys.path.extend(["src", ".."])
from studies.utils.custom_logger import RESULT
from studies.utils.helpers import load_data

# QUnfold modules
from QUnfold import QUnfoldQUBO, QUnfoldPlotter


def main():

    # Create dirs
    if not os.path.exists("../img/QUnfold/{}".format(args.distr)):
        os.makedirs("../img/QUnfold/{}".format(args.distr))
    if not os.path.exists("output/QUnfold/{}".format(args.distr)):
        os.makedirs("output/QUnfold/{}".format(args.distr))

    # Load histograms and response from file
    (
        truth,
        measured,
        response,
        binning,
    ) = load_data(args.distr)
    bins = int(binning[0])
    min_bin = int(binning[1])
    max_bin = int(binning[2])

    # Performing the unfolding with different methods
    for unf_type in ["simulated"]:

        # Compute the unfolding result
        RESULT("Unfolded with {} annealing:".format(unf_type))
        unfolder = QUnfoldQUBO(response, measured)
        unfolded = unfolder.solve_simulated_annealing(lam=0.1, num_reads=300)
        print("Bin contents: {}".format(unfolded))

        # Save unfolded data
        np.savetxt(
            "output/QUnfold/{}/unfolded_{}_bin_contents.txt".format(
                args.distr, unf_type
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
            "../img/QUnfold/{}/unfolded_{}.png".format(args.distr, unf_type), title
        )


if __name__ == "__main__":

    # Parser settings
    parser = ap.ArgumentParser(description="Parsing unfolding input variables.")
    parser.add_argument(
        "-d",
        "--distr",
        default="breit-wigner",
        type=str,
        help="Input distribution used for unfolding (used to read data).",
    )
    args = parser.parse_args()

    # Run main function
    main()
