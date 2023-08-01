#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ---------------------- Metadata ----------------------
#
# File name:  unfolding.py
# Author:     Gianluca Bianco (biancogianluca9@gmail.com)
# Date:       2023-06-13
# Copyright:  (c) 2023 Gianluca Bianco under the MIT license.

# Input variables
distributions=["breit-wigner", "normal", "double-peaked"]
samples=10000
max_bin=10
min_bin=-10
bins=40

# Standard modules
import os

# Data science modules
import ROOT as r
import numpy as np

# Utils modules
from functions.custom_logger import INFO
from functions.ROOT_converter import (
    TH1_to_array
)
from functions.generator import generate_standard, generate_double_peaked

# ROOT settings
r.gROOT.SetBatch(True)

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
    m_response_canvas.SaveAs("../img/RooUnfold/{}/response.png".format(distr))


def unfolder(type, m_response, h_meas, distr):
    """
    Unfold a distribution based on a certain type of unfolding.

    Args:
        type (str): the unfolding type (MI, SVD, IBU).
        m_response (ROOT.TH2F): the response matrix.
        h_meas (ROOT.TH1F): the measured pseudo-data.
        distr (distr): the generated distribution.

    Returns:
        ROOT.TH1F: the unfolded histogram.
    """

    # Variables
    unfolder = ""

    # Unfolding type settings
    if type == "MI":
        unfolder = r.RooUnfoldInvert("MI", "Matrix Inversion")
    elif type == "SVD":
        unfolder = r.RooUnfoldSvd("SVD", "SVD Tikhonov")
        unfolder.SetKterm(2)
    elif type == "IBU":
        unfolder = r.RooUnfoldBayes("IBU", "Iterative Bayesian")
        unfolder.SetIterations(4)
        unfolder.SetSmoothing(0)
    elif type == "B2B":
        unfolder = r.RooUnfoldBinByBin("B2B", "Bin-y-Bin")

    # Generic unfolding settings
    unfolder.SetVerbose(0)
    unfolder.SetResponse(m_response)
    unfolder.SetMeasured(h_meas)
    histo = unfolder.Hunfold()
    histo.SetName("unfolded_{}".format(type))

    # Save the unfolded histogram
    bin_contents = TH1_to_array(histo)
    np.savetxt(
        "output/RooUnfold/{}/unfolded_{}_bin_contents.txt".format(distr, type),
        bin_contents,
    )

    return histo


def plot_unfolding(truth, meas, unfolded, distr):
    """
    Plots the unfolding results.

    Args:
        truth (ROOT.TH1): True distribution histogram.
        meas (ROOT.TH1): Measured distribution histogram.
        unfolded (ROOT.TH1): Unfolded distribution histogram.
        distr (distr): the generated distribution.
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
    leg = r.TLegend(0.7, 0.7, 0.9, 0.9)
    leg.AddEntry(truth, "True", "pl")
    leg.AddEntry(meas, "Measured", "pl")
    ext = unfolded.GetName().split("_")[-1]
    if ext == "MI":
        leg.AddEntry(unfolded, "Unfolded (MI)")
    elif ext == "SVD":
        leg.AddEntry(unfolded, "Unfolded (SVD)")
    elif ext == "IBU":
        leg.AddEntry(unfolded, "Unfolded (IBU)")
    elif ext == "B2B":
        leg.AddEntry(unfolded, "Unfolded (B2B)")
    leg.Draw()

    # Save canvas
    canvas.Draw()
    canvas.SaveAs("../img/RooUnfold/{}/unfolded_{}.png".format(distr, ext))


def main():
    
    # Iterate over distributions
    print()
    for distr in distributions:
        INFO("Unfolding the {} distribution".format(distr))

        # Create dirs
        if not os.path.exists("../img/RooUnfold/{}".format(distr)):
            os.makedirs("../img/RooUnfold/{}".format(distr))
        if not os.path.exists("output/RooUnfold/{}".format(distr)):
            os.makedirs("output/RooUnfold/{}".format(distr))
            
        # Generating the distribution
        truth = r.TH1F("Truth", "", bins, min_bin, max_bin)
        meas = r.TH1F("Measured", "", bins, min_bin, max_bin)
        response = r.RooUnfoldResponse(bins, min_bin, max_bin)
        
        if any(d in distr for d in ["normal", "breit-wigner"]):
            truth, meas = generate_standard(truth, meas, response, "data", distr)
            response = generate_standard(truth, meas, response, "response", distr)
        elif any(d in distr for d in ["double-peaked"]):
            truth, meas = generate_double_peaked(truth, meas, response, "data")
            response = generate_double_peaked(truth, meas, response, "response")
        response.UseOverflow(False)
            
        plot_response(response, distr)

        # Performing the unfolding with different methods
        for unf_type in ["MI", "IBU", "SVD", "B2B"]:
            unfolded = unfolder(unf_type, response, meas, distr)
            plot_unfolding(truth, meas, unfolded, distr)
            
        # Deleting histograms
        del meas, truth, response
        
        print()
    print("Done.")


if __name__ == "__main__":
    main()
