#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Wed Jun 14 17:10:00 2023
Author: Gianluca Bianco
"""

# Followed the guide at: https://statisticalmethods.web.cern.ch/StatisticalMethods/unfolding/RooUnfold_01-Methods_PY/#Aproximating-Smearing

# STD modules
import argparse as ap
import sys, os
from tqdm import tqdm

# Data science modules
import ROOT as r
import numpy as np

# Utils modules
sys.path.extend(["../src", ".."])
from QUnfold.utils.custom_logger import INFO
from studies.utils.helpers import load_RooUnfold
from studies.utils.ROOT_converter import TH1_to_array, TH2_to_array

# ROOT settings
load_RooUnfold()
r.gROOT.SetBatch(True)


def smear(xt):
    """
    Applies a smearing effect to a given input value, xt.

    Args:
        xt (float): The input value to apply the smearing to.

    Returns:
        float or None: The resulting value after applying the smearing. Returns None if the value is filtered based on efficiency.
    """
    xeff = 0.3 + (1.0 - 0.3) / 20 * (xt + 10.0)  #  efficiency
    x = r.gRandom.Rndm()
    if x > xeff:
        return None
    xsmear = r.gRandom.Gaus(-2.5, 0.2)  #  bias and smear
    return xt + xsmear


def plot_response(response):
    """
    Plots the unfolding response matrix.

    Args:
        response (ROOT.RooUnfoldResponse): the response matrix to be plotted.
    """

    # Basic properties
    m_response_save = response.HresponseNoOverflow()
    m_response_canvas = r.TCanvas()
    m_response_save.SetStats(0)  # delete statistics box
    m_response_save.Draw("colz")  # to have heatmap

    # Save canvas
    m_response_canvas.Draw()
    m_response_canvas.SaveAs("../data/{}/response.png".format(args.distr))


def plot_truth_reco(h_truth, h_reco):
    """
    Plots truth and reco distributions.

    Args:
        h_truth (ROOT.TH1F): the truth distribution.
        h_reco (ROOT.TH1F): the reco distribution.
    """

    # Basic properties
    input_canvas = r.TCanvas()
    h_truth.SetStats(0)
    h_truth.SetFillColor(7)
    h_truth.GetXaxis().SetTitle("Bins")
    h_truth.SetTitle("")
    h_truth.Draw()
    h_reco.SetStats(0)
    h_reco.SetFillColor(42)
    h_reco.Draw("same")

    # Legend
    leg = r.TLegend(0.55, 0.7, 0.9, 0.9)
    leg.AddEntry(h_truth, "True Distribution")
    leg.AddEntry(h_reco, "Predicted Measured")
    leg.Draw()

    # Save canvas
    input_canvas.Draw()
    input_canvas.SaveAs("../data/{}/true-reco.png".format(args.distr))


def main():

    # Create directories if don't exist
    path = "../data/{}".format(args.distr)
    if not os.path.exists(path):
        os.makedirs(path)

    # Initial message
    INFO("Parameters:")
    print("- Distribution: {}".format(args.distr))
    print("- Samples: {}".format(args.samples))
    print()

    # Initialize histograms
    f0 = r.TH1F("f0", "f0", 40, -10, 10)  # truth
    g0 = r.TH1F("g0", "g0", 40, -10, 10)  # measured

    # Generate the response matrix
    response = r.RooUnfoldResponse(40, -10.0, 10.0)

    # Fill the inputs
    INFO("Filling the histograms...")
    if any(
        d in args.distr for d in ["normal", "breit-wigner"]
    ):  # case for standard distributions
        for i in tqdm(range(args.samples)):
            xt = 0
            if args.distr == "breit-wigner":
                xt = r.gRandom.BreitWigner(0.3, 2.5)
            elif args.distr == "normal":
                xt = r.gRandom.Gaus(0.0, 2.0)
            f0.Fill(xt)
            x = smear(xt)
            if x != None:
                response.Fill(x, xt)
                g0.Fill(x)
            else:
                response.Miss(xt)
    elif any(d in args.distr for d in ["double-peaked"]):  # case for double peaked
        for i in tqdm(range(5000)):
            xt = r.gRandom.Gaus(2, 1.5)
            f0.Fill(xt)
            x = r.gRandom.Gaus(xt, 1.0)
            if x != None:
                response.Fill(x, xt)
                g0.Fill(x)
            else:
                response.Miss(xt)
        for i in tqdm(range(5000)):
            xt = r.gRandom.Gaus(-2, 1.5)
            f0.Fill(xt)
            x = r.gRandom.Gaus(xt, 1.0)
            if x != None:
                response.Fill(x, xt)
                g0.Fill(x)
            else:
                response.Miss(xt)

    # Save response and histograms plots
    plot_response(response)
    plot_truth_reco(f0, g0)

    # Save the histograms as numpy arrays
    truth_bin_content, truth_bin_err = TH1_to_array(f0)
    np.savetxt("../data/{}/truth_bin_content.txt".format(args.distr), truth_bin_content)
    np.savetxt("../data/{}/truth_bin_err.txt".format(args.distr), truth_bin_err)
    meas_bin_content, meas_bin_err = TH1_to_array(g0)
    np.savetxt("../data/{}/meas_bin_content.txt".format(args.distr), meas_bin_content)
    np.savetxt("../data/{}/meas_bin_err.txt".format(args.distr), meas_bin_err)

    # Save the response matrix as numpy matrix
    np_response = TH2_to_array(response.Hresponse())
    np.savetxt("../data/{}/response.txt".format(args.distr), np_response)


if __name__ == "__main__":

    # Parser settings
    parser = ap.ArgumentParser(description="Parsing generator input variables.")
    parser.add_argument(
        "-t",
        "--distr",
        choices=["normal", "breit-wigner", "double-peaked"],
        default="normal",
        type=str,
        help="The type of the distribution to be simulated.",
    )
    parser.add_argument(
        "-s",
        "--samples",
        default=100000,
        type=int,
        help="Number of samples to be generated.",
    )
    args = parser.parse_args()

    # Run main function
    main()
