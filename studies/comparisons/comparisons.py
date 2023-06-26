#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ---------------------- Metadata ----------------------
#
# File name:  comparisons.py
# Author:     Gianluca Bianco (biancogianluca9@gmail.com)
# Date:       2023-06-26
# Copyright:  (c) 2023 Gianluca Bianco under the MIT license.

# STD modules
import argparse as ap
import sys, os

# Data science modules
import matplotlib.pyplot as plt
import numpy as np


def plot_unfolded_distr(data):

    # Iterate over the unfolding methods
    for method, file in data.items():

        # Read data from file
        bin_contents = np.loadtxt(file)
        bin_edges = np.linspace(-10, 10, 41)

        # Plot data
        plt.bar(
            bin_edges[:-1],
            bin_contents,
            width=np.diff(bin_edges),
            align="edge",
            facecolor="#2ab0ff",
            edgecolor="#e0e0e0",
            linewidth=0.5,
        )


def main():

    # Read input data for the given distribution
    data = {
        "Truth": "../data/{}/truth_bin_content.txt".format(args.distr),
        "IBU": "output/RooUnfold/{}/unfolded_IBU_bin_contents.txt".format(args.distr),
        "B2B": "output/RooUnfold/{}/unfolded_B2B_bin_contents.txt".format(args.distr),
        "MI": "output/RooUnfold/{}/unfolded_MI_bin_contents.txt".format(args.distr),
        "SVD": "output/RooUnfold/{}/unfolded_SVD_bin_contents.txt".format(args.distr),
    }

    # Creating the output directory
    if not os.path.exists("../img/comparisons/{}".format(args.distr)):
        os.makedirs("../img/comparisons/{}".format(args.distr))

    # Plot the unfolded distributions for comparison with truth
    plot_unfolded_distr(data)


if __name__ == "__main__":

    # Parser settings
    parser = ap.ArgumentParser(description="Parsing comparisons input variables.")
    parser.add_argument(
        "-d",
        "--distr",
        default="normal",
        type=str,
        help="The type of the distribution to be used for comparison.",
    )
    args = parser.parse_args()

    # Run main function
    main()
