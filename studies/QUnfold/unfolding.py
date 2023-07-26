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
from QUnfold import QUnfoldQUBO


def unfolder(type, response, h_meas):
    """
    Unfold a distribution based on a certain type of unfolding.

    Args:
        type (str): the unfolding type (simulated, quantum).
        response (np.array): the response matrix.
        h_meas (np.array): the measured pseudo-data.
        binning (np.array): binning of the measured histogram

    Returns:
        np.array: the unfolded histogram.
    """

    # Variables
    histo = None

    # Unfolding type settings
    if type == "simulated":
        RESULT("Unfolded with simulated quantum annealing:")
        unfolder = QUnfoldQUBO()
        histo = unfolder.run_simulated_annealing(
            response, h_meas, lam=0.1, num_reads=100
        )

    # Print other information
    print("Bin contents: {}".format(histo))

    # Save the unfolded histogram
    np.savetxt(
        "output/QUnfold/{}/unfolded_{}_bin_contents.txt".format(args.distr, type),
        histo,
    )

    return histo


def plot_unfolding(truth, meas, unfolded, binning, type):
    """
    Plots the unfolding results.

    Args:
        truth (np.array): True distribution histogram.
        meas (np.array): Measured distribution histogram.
        unfolded (np.array): Unfolded distribution histogram.
        binning (np.array): Binning of the distributions.
        type (str): Unfolding type (classical, quantum).
    """

    # Variables
    plot_label = ""

    # Basic properties
    plt.step(
        x=np.concatenate(([binning[0] - (binning[1] - binning[0])], binning[:-1])),
        y=np.concatenate(([truth[0]], truth)),
        label="True",
        color="red",
    )
    plt.step(
        x=np.concatenate(([binning[0] - (binning[1] - binning[0])], binning[:-1])),
        y=np.concatenate(([meas[0]], meas)),
        label="Measured",
        color="blue",
    )
    if type == "classical":
        plot_label = "Unfolded (Sim. QUBO)"
    elif type == "quantum":
        plot_label = "Unfolded (QUBO)"
    plt.step(
        x=np.concatenate(([binning[0] - (binning[1] - binning[0])], binning[:-1])),
        y=np.concatenate(([unfolded[0]], unfolded)),
        label=plot_label,
        color="green",
    )
    plt.xlabel("Bins", loc="right")
    plt.legend()
    plt.savefig("../img/QUnfold/{}/unfolded_{}.png".format(args.distr, type))


def main():

    # Create dirs
    if not os.path.exists("../img/QUnfold/{}".format(args.distr)):
        os.makedirs("../img/QUnfold/{}".format(args.distr))
    if not os.path.exists("output/QUnfold/{}".format(args.distr)):
        os.makedirs("output/QUnfold/{}".format(args.distr))

    # Load histograms and response from file
    (
        np_truth_bin_content,
        np_meas_bin_content,
        np_response,
        np_binning,
    ) = load_data(args.distr)
    bins = int(np_binning[0])
    min_bin = int(np_binning[1])
    max_bin = int(np_binning[2])

    # Performing the unfolding with different methods
    for unf_type in ["simulated"]:
        unfolded = unfolder(unf_type, np_response, np_meas_bin_content)
        plot_unfolding(
            np_truth_bin_content,
            np_meas_bin_content,
            unfolded,
            np.linspace(min_bin, max_bin, bins + 1),
            unf_type,
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
