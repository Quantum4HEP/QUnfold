#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ---------------------- Metadata ----------------------
#
# File name:  unfolding.py
# Author:     Gianluca Bianco (biancogianluca9@gmail.com)
# Date:       2023-06-13
# Copyright:  (c) 2023 Gianluca Bianco under the MIT license.

# Standard modules
import argparse as ap
import sys, os

# Data science modules
import ROOT as r
import numpy as np

# Utils modules
sys.path.extend(["../src", ".."])
from studies.utils.custom_logger import RESULT
from studies.utils.ROOT_converter import (
    array_to_TH1,
    TH1_to_array,
    array_to_TH2,
)
from studies.utils.helpers import load_RooUnfold, load_data

# ROOT settings
load_RooUnfold()
r.gROOT.SetBatch(True)


def unfolder(type, m_response, h_meas):
    """
    Unfold a distribution based on a certain type of unfolding.

    Args:
        type (str): the unfolding type (MI, SVD, IBU).
        m_response (ROOT.TH2F): the response matrix.
        h_meas (ROOT.TH1F): the measured pseudo-data.

    Returns:
        ROOT.TH1F: the unfolded histogram.
    """

    # Variables
    unfolder = ""

    # Unfolding type settings
    if type == "MI":
        RESULT("Unfolded with matrix inversion:")
        unfolder = r.RooUnfoldInvert("MI", "Matrix Inversion")
    elif type == "SVD":
        RESULT("Unfolded with SVD Tikhonov method:")
        unfolder = r.RooUnfoldSvd("SVD", "SVD Tikhonov")
        unfolder.SetKterm(3)
    elif type == "IBU":
        RESULT("Unfolded with Iterative Bayesian Unfolding method:")
        unfolder = r.RooUnfoldBayes("IBU", "Iterative Bayesian")
        unfolder.SetIterations(4)
        unfolder.SetSmoothing(0)
    elif type == "B2B":
        RESULT("Unfolded with Bin-by-Bin method:")
        unfolder = r.RooUnfoldBinByBin("B2B", "Bin-y-Bin")

    # Generic unfolding settings
    unfolder.SetVerbose(0)
    unfolder.SetResponse(m_response)
    unfolder.SetMeasured(h_meas)
    histo = unfolder.Hunfold()
    histo.SetName("unfolded_{}".format(type))
    histo_mi_bin_c = TH1_to_array(histo)

    # Print other information
    print("Bin contents: {}".format(histo_mi_bin_c))

    # Save the unfolded histogram
    bin_contents = TH1_to_array(histo)
    np.savetxt(
        "output/RooUnfold/{}/unfolded_{}_bin_contents.txt".format(args.distr, type),
        bin_contents,
    )

    return histo


def plot_unfolding(truth, meas, unfolded):
    """
    Plots the unfolding results.

    Args:
        truth (ROOT.TH1): True distribution histogram.
        meas (ROOT.TH1): Measured distribution histogram.
        unfolded (ROOT.TH1): Unfolded distribution histogram.
    """

    # Basic properties
    canvas = r.TCanvas()
    truth.SetLineColor(2)
    truth.SetStats(0)
    unfolded.SetLineColor(3)
    truth.GetXaxis().SetTitle("Bins")
    truth.GetYaxis().SetTitle("")
    truth.Draw()
    meas.Draw("same")
    unfolded.Draw("same")

    # Legend settings
    leg = r.TLegend(0.6, 0.6, 0.9, 0.9)
    leg.AddEntry(truth, "True distribution", "pl")
    leg.AddEntry(meas, "Measured distribution", "pl")
    leg.AddEntry(unfolded, "Unfolded distribution")
    leg.Draw()

    # Save canvas
    canvas.Draw()
    ext = unfolded.GetName().split("_")[-1]
    canvas.SaveAs("../img/RooUnfold/{}/unfolded_{}.png".format(args.distr, ext))


def main():

    # Create dirs
    if not os.path.exists("../img/RooUnfold/{}".format(args.distr)):
        os.makedirs("../img/RooUnfold/{}".format(args.distr))
    if not os.path.exists("output/RooUnfold/{}".format(args.distr)):
        os.makedirs("output/RooUnfold/{}".format(args.distr))

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

    # Convert to ROOT variables
    h_truth = array_to_TH1(np_truth_bin_content, bins, min_bin, max_bin, "truth")
    h_meas = array_to_TH1(np_meas_bin_content, bins, min_bin, max_bin, "meas")
    h_response = array_to_TH2(
        np_response, bins, min_bin, max_bin, bins, min_bin, max_bin, "response"
    )

    # Initialize the RooUnfold response matrix from the input data
    m_response = r.RooUnfoldResponse(h_meas, h_truth, h_response)
    m_response.UseOverflow(False)  # disable the overflow bin which takes the outliers

    # Performing the unfolding with different methods
    for unf_type in ["MI", "IBU", "SVD", "B2B"]:
        unfolded = unfolder(unf_type, m_response, h_meas)
        plot_unfolding(h_truth, h_meas, unfolded)


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
