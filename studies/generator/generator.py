#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ---------------------- Metadata ----------------------
#
# File name:  generator.py
# Author:     Gianluca Bianco (biancogianluca9@gmail.com)
# Date:       2023-06-14
# Copyright:  (c) 2023 Gianluca Bianco under the MIT license.

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
    Applies a Gaussian smearing effect to a given input value, xt.

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


def generate_standard(f0, g0, response, type):
    """
    Generate data for standard distributions.

    Args:
        f0 (ROOT.TH1F): truth histogram.
        g0 (ROOT.TH1F): measured histogram.
        response (ROOT.TH2F): response matrix.
        type (str): type of data generation (data or response).

    Returns:
        ROOT.TH1F: the filled truth histogram.
        ROOT.TH1F: the filled measured histogram.
        ROOT.TH2F: the filled response matrix.
    """

    # Data generation
    if type == "data":
        r.gRandom.SetSeed(12345)
        for i in tqdm(range(args.samples)):
            xt = 0
            if args.distr == "breit-wigner":
                xt = r.gRandom.BreitWigner(0.3, 2.5)
            elif args.distr == "normal":
                xt = r.gRandom.Gaus(0.0, 2.0)
            f0.Fill(xt)
            x = smear(xt)
            if x != None:
                g0.Fill(x)

        return f0, g0

    # Response generation
    elif type == "response":
        r.gRandom.SetSeed(556)
        for i in tqdm(range(args.samples)):
            xt = 0
            if args.distr == "breit-wigner":
                xt = r.gRandom.BreitWigner(0.3, 2.5)
            elif args.distr == "normal":
                xt = r.gRandom.Gaus(0.0, 2.0)
            x = smear(xt)
            if x != None:
                response.Fill(x, xt)
            else:
                response.Miss(xt)

        return response


def generate_double_peaked(f0, g0, response, type):
    """
    Generate data for the double peaked distributions.

    Args:
        f0 (ROOT.TH1F): truth histogram.
        g0 (ROOT.TH1F): measured histogram.
        response (ROOT.TH2F): response matrix.
        type (str): type of data generation (data or response).

    Returns:
        ROOT.TH1F: the filled truth histogram.
        ROOT.TH1F: the filled measured histogram.
        ROOT.TH2F: the filled response matrix.
    """

    # Data generation
    if type == "data":
        r.gRandom.SetSeed(12345)
        for i in tqdm(range(5000)):
            xt = r.gRandom.Gaus(2, 1.5)
            f0.Fill(xt)
            x = r.gRandom.Gaus(xt, 1.0)
            if x != None:
                g0.Fill(x)
        for i in tqdm(range(5000)):
            xt = r.gRandom.Gaus(-2, 1.5)
            f0.Fill(xt)
            x = r.gRandom.Gaus(xt, 1.0)
            if x != None:
                g0.Fill(x)

        return f0, g0

    # Response generation
    elif type == "response":
        r.gRandom.SetSeed(556)
        for i in tqdm(range(5000)):
            xt = r.gRandom.Gaus(2, 1.5)
            x = r.gRandom.Gaus(xt, 1.0)
            if x != None:
                response.Fill(x, xt)
            else:
                response.Miss(xt)
        for i in tqdm(range(5000)):
            xt = r.gRandom.Gaus(-2, 1.5)
            x = r.gRandom.Gaus(xt, 1.0)
            if x != None:
                response.Fill(x, xt)
            else:
                response.Miss(xt)

        return response


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
    m_response_canvas.SaveAs("../img/data/{}/response.png".format(args.distr))


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
    input_canvas.SaveAs("../img/data/{}/true-reco.png".format(args.distr))


def main():

    # Create directories if don't exist
    path = "../data/{}".format(args.distr)
    if not os.path.exists(path):
        os.makedirs(path)
    img_path = "../img/data/{}".format(args.distr)
    if not os.path.exists(img_path):
        os.makedirs(img_path)

    # Initial message
    INFO("Parameters:")
    print("- Distribution: {}".format(args.distr))
    print("- Samples: {}".format(args.samples))
    print("- Binning: ({}, {}, {})".format(args.bins, args.min_bin, args.max_bin))
    print()

    # Initialize histograms
    f0 = r.TH1F("f0", "f0", args.bins, args.min_bin, args.max_bin)  # truth
    g0 = r.TH1F("g0", "g0", args.bins, args.min_bin, args.max_bin)  # measured

    # Generate the response matrix
    response = r.RooUnfoldResponse(args.bins, args.min_bin, args.max_bin)

    # Fill the inputs
    INFO("Filling the histograms...")

    # Case for standard distributions
    if any(d in args.distr for d in ["normal", "breit-wigner"]):
        f0, g0 = generate_standard(f0, g0, response, "data")
        response = generate_standard(f0, g0, response, "response")

    # Generate data for double peaked distributions
    elif any(d in args.distr for d in ["double-peaked"]):
        f0, g0 = generate_double_peaked(f0, g0, response, "data")
        response = generate_double_peaked(f0, g0, response, "response")

    # Save response and histograms plots
    plot_response(response)
    plot_truth_reco(f0, g0)

    # Save the histograms as numpy arrays
    truth_bin_content = TH1_to_array(f0)
    np.savetxt("../data/{}/truth_bin_content.txt".format(args.distr), truth_bin_content)
    meas_bin_content = TH1_to_array(g0)
    np.savetxt("../data/{}/meas_bin_content.txt".format(args.distr), meas_bin_content)

    # Save the response matrix as numpy matrix
    np_response = TH2_to_array(response.Hresponse())
    np.savetxt("../data/{}/response.txt".format(args.distr), np_response)

    # Save the binning
    with open("../data/{}/binning.txt".format(args.distr), "w") as f:
        f.write("{}\n".format(args.bins))
        f.write("{}\n".format(args.min_bin))
        f.write("{}\n".format(args.max_bin))


if __name__ == "__main__":

    # Parser settings
    parser = ap.ArgumentParser(description="Parsing generator input variables.")
    parser.add_argument(
        "-d",
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
    parser.add_argument(
        "-M",
        "--max_bin",
        default=10,
        type=int,
        help="The maximum bin edge.",
    )
    parser.add_argument(
        "-m",
        "--min_bin",
        default=-10,
        type=int,
        help="The minimum bin edge.",
    )
    parser.add_argument(
        "-b",
        "--bins",
        default=40,
        type=int,
        help="The number of bins.",
    )
    args = parser.parse_args()

    # Run main function
    main()
