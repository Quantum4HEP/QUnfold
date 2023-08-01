#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ---------------------- Metadata ----------------------
#
# File name:  generator.py
# Author:     Gianluca Bianco (biancogianluca9@gmail.com)
# Date:       2023-06-14
# Copyright:  (c) 2023 Gianluca Bianco under the MIT license.

# Followed the guide at: https://statisticalmethods.web.cern.ch/StatisticalMethods/unfolding/RooUnfold_01-Methods_PY/#Aproximating-Smearing

# Input variables
distributions=["breit-wigner", "normal", "double-peaked"]
samples=10000
max_bin=10
min_bin=-10
bins=40
remove_empty_bins="no"

# STD modules
import os

# Data science modules
import ROOT as r
import numpy as np

# Utils modules
from utils.custom_logger import INFO
from utils.helpers import load_RooUnfold
from utils.ROOT_converter import (
    TH1_to_array,
    TH2_to_array,
    remove_zero_entries_bins,
)

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


def generate_standard(f0, g0, response, type, distr):
    """
    Generate data for standard distributions.

    Args:
        f0 (ROOT.TH1F): truth histogram.
        g0 (ROOT.TH1F): measured histogram.
        response (ROOT.TH2F): response matrix.
        type (str): type of data generation (data or response).
        distr (distr): the distribution to be generated.

    Returns:
        ROOT.TH1F: the filled truth histogram.
        ROOT.TH1F: the filled measured histogram.
        ROOT.TH2F: the filled response matrix.
    """

    # Data generation
    if type == "data":
        r.gRandom.SetSeed(12345)
        for i in range(samples):
            xt = 0
            if distr == "breit-wigner":
                xt = r.gRandom.BreitWigner(0.3, 2.5)
            elif distr == "normal":
                xt = r.gRandom.Gaus(0.0, 2.0)
            f0.Fill(xt)
            x = smear(xt)
            if x != None:
                g0.Fill(x)

        return f0, g0

    # Response generation
    elif type == "response":
        r.gRandom.SetSeed(556)
        for i in range(samples):
            xt = 0
            if distr == "breit-wigner":
                xt = r.gRandom.BreitWigner(0.3, 2.5)
            elif distr == "normal":
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
        for i in range(samples):
            xt = r.gRandom.Gaus(2, 1.5)
            f0.Fill(xt)
            x = r.gRandom.Gaus(
                xt,
            )
            if x != None:
                g0.Fill(x)
        for i in range(samples):
            xt = r.gRandom.Gaus(-2, 1.5)
            f0.Fill(xt)
            x = r.gRandom.Gaus(xt, 1)
            if x != None:
                g0.Fill(x)

        return f0, g0

    # Response generation
    elif type == "response":
        r.gRandom.SetSeed(556)
        for i in range(samples):
            xt = r.gRandom.Gaus(2, 1.5)
            x = r.gRandom.Gaus(xt, 1)
            if x != None:
                response.Fill(x, xt)
            else:
                response.Miss(xt)
        for i in range(samples):
            xt = r.gRandom.Gaus(-2, 1.5)
            x = r.gRandom.Gaus(xt, 1)
            if x != None:
                response.Fill(x, xt)
            else:
                response.Miss(xt)

        return response


def plot_response(response, distr):
    """
    Plots the unfolding response matrix.

    Args:
        response (ROOT.RooUnfoldResponse): the response matrix to be plotted.
        distr (distr): the distribution to be generated.
    """

    # Basic properties
    m_response_save = response.HresponseNoOverflow()
    m_response_canvas = r.TCanvas()
    m_response_save.SetStats(0)  # delete statistics box
    m_response_save.GetXaxis().SetTitle("Truth")
    m_response_save.GetYaxis().SetTitle("Measured")
    m_response_save.Draw("colz")  # to have heatmap

    # Save canvas
    m_response_canvas.Draw()
    m_response_canvas.SaveAs("../img/data/{}/response.png".format(distr))


def plot_truth_reco(h_truth, h_reco, distr):
    """
    Plots truth and reco distributions.

    Args:
        h_truth (ROOT.TH1F): the truth distribution.
        h_reco (ROOT.TH1F): the reco distribution.
        distr (distr): the distribution to be generated.
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
    input_canvas.SaveAs("../img/data/{}/true-reco.png".format(distr))


def main():
    
    # Iterate over distributions
    print()
    for distr in distributions:

        # Create directories if don't exist
        path = "../data/{}".format(distr)
        if not os.path.exists(path):
            os.makedirs(path)
        img_path = "../img/data/{}".format(distr)
        if not os.path.exists(img_path):
            os.makedirs(img_path)

        # Initialize histograms
        f0 = r.TH1F("f0", "f0", bins, min_bin, max_bin)  # truth
        g0 = r.TH1F("g0", "g0", bins, min_bin, max_bin)  # measured

        # Generate the response matrix
        response = r.RooUnfoldResponse(bins, min_bin, max_bin)

        # Fill the inputs
        INFO("Generating the {} distribution:".format(distr))

        # Case for standard distributions
        if any(d in distr for d in ["normal", "breit-wigner"]):
            f0, g0 = generate_standard(f0, g0, response, "data", distr)
            response = generate_standard(f0, g0, response, "response", distr)

        # Generate data for double peaked distributions
        elif any(d in distr for d in ["double-peaked"]):
            f0, g0 = generate_double_peaked(f0, g0, response, "data")
            response = generate_double_peaked(f0, g0, response, "response")

        # Remove measured bins with 0 entries and the corresponding bins of the true
        if remove_empty_bins == "yes":
            INFO("Removing empty bins...")
            g0, f0, m_response = remove_zero_entries_bins(
                g0, f0, response.HresponseNoOverflow()
            )
            response = r.RooUnfoldResponse(g0, f0, m_response)

        # Save response and histograms plots
        plot_response(response, distr)
        plot_truth_reco(f0, g0, distr)

        # Save the histograms as numpy arrays
        truth_bin_content = TH1_to_array(f0)
        np.savetxt("../data/{}/truth_bin_content.txt".format(distr), truth_bin_content)
        meas_bin_content = TH1_to_array(g0)
        np.savetxt("../data/{}/meas_bin_content.txt".format(distr), meas_bin_content)

        # Save the response matrix as numpy matrix
        np_response = TH2_to_array(response.Hresponse())
        np.savetxt("../data/{}/response.txt".format(distr), np_response)

        # Save the binning
        with open("../data/{}/binning.txt".format(distr), "w") as f:
            f.write("{}\n".format(g0.GetNbinsX()))
            f.write("{}\n".format(int(g0.GetXaxis().GetBinLowEdge(1))))
            f.write("{}\n".format(int(g0.GetXaxis().GetBinUpEdge(g0.GetNbinsX()))))

        # Final message
        INFO("Parameters:")
        print("- Distribution: {}".format(distr))
        print("- Samples: {}".format(samples))
        print(
            "- Binning: ({}, {}, {})".format(
                g0.GetNbinsX(),
                int(g0.GetXaxis().GetBinLowEdge(1)),
                int(g0.GetXaxis().GetBinUpEdge(g0.GetNbinsX())),
            )
        )
        print()
        
        del f0, g0
    print("Done.")


if __name__ == "__main__":
    main()
