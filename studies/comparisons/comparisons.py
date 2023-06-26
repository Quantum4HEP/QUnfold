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
from scipy.stats import chisquare


def plot_errorbar(bin_edges, bin_contents, color, marker, method, chi2):
    """
    Plots an error bar plot with specified bin edges, bin contents, color, marker, and method.

    Args:
        bin_edges (array-like): The bin edges for the error bars.
        bin_contents (array-like): The bin contents for the error bars.
        color (str): The color of the error bars.
        marker (str): The marker style for the error bars.
        method (str): The method label for the error bars.
        chi2 (float): The chi2 of comparison about truth and unfolded distribution.
    """

    plt.errorbar(
        x=bin_edges[:-1],
        y=bin_contents,
        yerr=np.sqrt(bin_contents),
        color=color,
        marker=marker,
        ms=5,
        label=r"{} ($\chi^2 = {:.2f}$)".format(method, chi2),
        linestyle="None",
    )


def compute_chi2_dof(bin_contents, truth_bin_contents):
    """
    Compute the chi-squared per degree of freedom (chi2/dof) between two distributions.

    Args:
        bin_contents (numpy.array): The observed bin contents.
        truth_bin_contents (numpy.array): The expected bin contents.

    Returns:
        float: The chi-squared per degree of freedom.
    """

    chi2, pvalue = chisquare(
        bin_contents,
        np.sum(bin_contents) / np.sum(truth_bin_contents) * truth_bin_contents,
    )
    dof = len(bin_contents) - 1
    chi2_dof = chi2 / dof

    return chi2_dof


def plot_unfolded_distr(data):
    """
    Plots the unfolded distributions for different unfolding methods.

    Args:
        data (dict): A dictionary containing the data for each unfolding method. The keys represent the method names, and the values represent the corresponding file paths.
    """

    # Get binning information
    binning_file = "../data/{}/binning.txt".format(args.distr)
    binning = np.loadtxt(binning_file)
    bins = int(binning[0])
    min_bin = int(binning[1])
    max_bin = int(binning[2])
    bin_edges = np.linspace(min_bin, max_bin, bins + 1)

    # Plot truth distribution
    marker_offset = (bin_edges[1] - bin_edges[0]) / 2.0
    truth_bin_contents = np.loadtxt(
        "../data/{}/truth_bin_content.txt".format(args.distr)
    )
    plt.step(
        x=np.concatenate(
            ([bin_edges[0] - (bin_edges[1] - bin_edges[0])], bin_edges[:-1])
        ),
        y=np.concatenate(([truth_bin_contents[0]], truth_bin_contents)),
        label="Truth",
        color="black",
        linestyle="dashed",
    )

    # Iterate over the unfolding methods
    for method, file in data.items():

        # Plot each unfolding method
        bin_contents = np.loadtxt(file)
        truth_bin_contents = np.where(
            truth_bin_contents == 0, 1e-6, truth_bin_contents
        )  # Trick for chi2
        chi2_dof = compute_chi2_dof(bin_contents, truth_bin_contents)
        if method == "IBU4":
            plot_errorbar(
                bin_edges - marker_offset, bin_contents, "red", "o", method, chi2_dof
            )
        elif method == "B2B":
            plot_errorbar(
                bin_edges - marker_offset, bin_contents, "blue", "o", method, chi2_dof
            )
        elif method == "SVD":
            plot_errorbar(
                bin_edges - marker_offset, bin_contents, "green", "o", method, chi2_dof
            )

        # Plot settings
        plt.xlabel("Bins")
        plt.ylabel("Unfolded distribution")
        plt.legend()

        # Save plot
        plt.savefig("../img/comparisons/{}.png".format(args.distr))

    print("Done.")


def main():

    # Read input data for the given distribution
    data = {
        "IBU4": "output/RooUnfold/{}/unfolded_IBU_bin_contents.txt".format(args.distr),
        "B2B": "output/RooUnfold/{}/unfolded_B2B_bin_contents.txt".format(args.distr),
        "MI": "output/RooUnfold/{}/unfolded_MI_bin_contents.txt".format(args.distr),
        "SVD": "output/RooUnfold/{}/unfolded_SVD_bin_contents.txt".format(args.distr),
    }

    # Creating the output directory
    if not os.path.exists("../img/comparisons"):
        os.makedirs("../img/comparisons")

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
