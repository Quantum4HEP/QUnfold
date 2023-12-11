#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ---------------------- Metadata ----------------------
#
# File name:  comparisons.py
# Author:     Gianluca Bianco (biancogianluca9@gmail.com)
# Date:       2023-06-26
# Copyright:  (c) 2023 Gianluca Bianco under the MIT license.

import os
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

    # Trick for chi2 convergence
    null_indices = truth_bin_contents == 0
    truth_bin_contents[null_indices] += 1
    bin_contents[null_indices] += 1

    # Compute chi2
    chi2, _ = chisquare(
        bin_contents,
        np.sum(bin_contents) / np.sum(truth_bin_contents) * truth_bin_contents,
    )
    dof = len(bin_contents) - 1
    chi2_dof = chi2 / dof

    return chi2_dof


def plot_comparisons(data, distr, truth, bins, min_bin, max_bin):
    """
    Plots the unfolded distributions for different unfolding methods.

    Args:
        data (dict): A dictionary containing the data for each unfolding method. The keys represent the method names, and the values represent the corresponding file paths.
        distr (distr): the generated distribution.
    """

    # Plot truth distribution
    bin_edges = np.linspace(min_bin, max_bin, bins + 1)
    marker_offset = (bin_edges[1] - bin_edges[0]) / 2.0
    plt.step(
        x=np.concatenate(
            ([bin_edges[0] - (bin_edges[1] - bin_edges[0])], bin_edges[:-1])
        ),
        y=np.concatenate(([truth[0]], truth)),
        label="Truth",
        color="black",
        linestyle="dashed",
    )

    # Iterate over the unfolding methodsunfolded
    for method, unfolded in data.items():
        # Plot each unfolding method
        chi2_dof = compute_chi2_dof(unfolded, truth)
        if method == "IBU4":
            plot_errorbar(
                bin_edges - marker_offset, unfolded, "red", "o", method, chi2_dof
            )
        elif method == "SVD":
            plot_errorbar(
                bin_edges - marker_offset, unfolded, "green", "s", method, chi2_dof
            )
        elif method == "SA":
            plot_errorbar(
                bin_edges - marker_offset, unfolded, "purple", "*", method, chi2_dof
            )
        elif method == "HYB":
            plot_errorbar(
                bin_edges - marker_offset, unfolded, "orange", "*", method, chi2_dof
            )

        # Plot settings
        plt.xlabel("Bins")
        plt.ylabel("Entries")
        plt.tight_layout()
        plt.legend()

        # Save plot
        if not os.path.exists("studies/img//comparisons"):
            os.makedirs("studies/img//comparisons")
        plt.savefig("studies/img//comparisons/{}.png".format(distr))

    plt.close()
