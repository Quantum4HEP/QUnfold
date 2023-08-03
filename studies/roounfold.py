#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import ROOT

path = "../results/roounfold/"


def roounfold_unfolder(response, meas, method):
    """
    Apply RooUnfold algorithm to unfold the given measured distribution.

    Args:
        response (ROOT.TH2F): response matrix.
        meas (ROOT.TH1F): measured histogram.
        method (str): classical unfolding method (RMI, SVD, IBU, B2B).

    Returns:
        ROOT.TH1F: unfolded histogram.
    """

    if method == "RMI":
        unfolder = ROOT.RooUnfoldInvert()
    elif method == "IBU":
        unfolder = ROOT.RooUnfoldBayes()
        unfolder.SetIterations(4)
        unfolder.SetSmoothing(0)
    elif method == "SVD":
        unfolder = ROOT.RooUnfoldSvd()
        unfolder.SetKterm(2)
    elif method == "B2B":
        unfolder = ROOT.RooUnfoldBinByBin()
    else:
        raise ValueError(f"Unfolding method '{method}' is unknown.")

    unfolder.SetVerbose(0)  # disable logs
    unfolder.SetResponse(response)
    unfolder.SetMeasured(meas)
    histo = unfolder.Hunfold()  # get unfolded histogram
    histo.SetName(method)

    return histo


def roounfold_plot_response(response, distr, ext="png"):
    """
    Save response matrix heatmap to local filesystem.

    Args:
        response (ROOT.RooUnfoldResponse): response matrix to be plotted.
        distr (str): name of the distribution.
        ext (str): output file extension (png, pdf)
    """
    if not os.path.exists(f"{path}{distr}"):
        os.makedirs(f"{path}{distr}")

    th2 = response.Hresponse()
    canvas = ROOT.TCanvas()
    th2.SetStats(0)  # hide stats box
    th2.SetTitle("")
    th2.GetXaxis().SetTitle("Meas")
    th2.GetYaxis().SetTitle("True")
    th2.Draw("colz")  # draw heatmap bar
    canvas.Draw()
    canvas.SaveAs(f"{path}{distr}/response.{ext}")


def roounfold_plot_results(true, meas, unfolded, distr, ext="png"):
    """
    Save unfolding results plot to local filesystem.

    Args:
        true (ROOT.TH1): true distribution histogram.
        meas (ROOT.TH1): measured distribution histogram.
        unfolded (ROOT.TH1): unfolded distribution histogram.
        distr (str): name of the distribution.
        ext (str): output file extension (png, pdf)
    """
    if not os.path.exists(f"{path}{distr}"):
        os.makedirs(f"{path}{distr}")

    canvas = ROOT.TCanvas()
    true.SetLineColor(2)
    true.SetStats(0)  # hide stats box
    unfolded.SetLineColor(3)
    true.SetTitle("")
    true.Draw()
    meas.Draw("same")
    unfolded.Draw("same")

    legend = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)
    legend.AddEntry(true, "True", "pl")
    legend.AddEntry(meas, "Meas", "pl")
    label = unfolded.GetName()
    legend.AddEntry(unfolded, f"Unfolded ({label})")
    legend.Draw()

    canvas.Draw()
    canvas.SaveAs(f"{path}{distr}/unfolded_{label}.{ext}")
