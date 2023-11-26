#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ---------------------- Metadata ----------------------
#
# File name:  RooUnfold.py
# Author:     Gianluca Bianco (biancogianluca9@gmail.com)
# Date:       2023-06-13
# Copyright:  (c) 2023 Gianluca Bianco under the MIT license.

# STD modules
import os

# Data science modules
import ROOT as r

# ROOT settings
r.gROOT.SetBatch(True)


def RooUnfold_plot_response(response, distr):
    """
    Plots the unfolding response matrix.

    Args:
        response (ROOT.RooUnfoldResponse): the response matrix to be plotted.
        distr (distr): the distribution to be generated.
    """

    # Creating the path
    if not os.path.exists("studies/img/RooUnfold/{}".format(distr)):
        os.makedirs("studies/img/RooUnfold/{}".format(distr))

    # Basic properties
    m_response_save = response.HresponseNoOverflow()
    m_response_canvas = r.TCanvas()
    m_response_save.SetStats(0)  # delete statistics box
    m_response_save.GetXaxis().SetTitle("Truth")
    m_response_save.GetYaxis().SetTitle("Measured")
    m_response_save.Draw("colz")  # to have heatmap

    # Save canvas
    m_response_canvas.Draw()
    m_response_canvas.SaveAs("studies/img/RooUnfold/{}/response.png".format(distr))


def RooUnfold_unfolder(type, m_response, h_measured):
    """
    Unfold a distribution based on a certain type of unfolding.

    Args:
        type (str): the unfolding type (MI, SVD, IBU).
        m_response (ROOT.TH2F): the response matrix.
        h_measured (ROOT.TH1F): the measured pseudo-data.

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
    unfolder.SetMeasured(h_measured)
    histo = unfolder.Hunfold()
    histo.SetName("unfolded_{}".format(type))

    return histo


def RooUnfold_plot(truth, measured, unfolded, distr):
    """
    Plots the unfolding results.

    Args:
        truth (ROOT.TH1): True distribution histogram.
        measured (ROOT.TH1): Measured distribution histogram.
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
    measured.Draw("same")
    unfolded.Draw("same")

    # Legend settings
    leg = r.TLegend(0.7, 0.7, 0.9, 0.9)
    leg.AddEntry(truth, "True", "pl")
    leg.AddEntry(measured, "Measured", "pl")
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
    canvas.SaveAs("studies/img/RooUnfold/{}/unfolded_{}.png".format(distr, ext))
