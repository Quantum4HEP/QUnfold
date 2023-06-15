#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Tue Jun 13 17:10:00 2023
Author: Gianluca Bianco
"""

# Standard modules
import argparse as ap
import sys, os

# Data science modules
import ROOT as r
import pandas as pd
import numpy as np
import scipy.stats as sc

# Utils modules
sys.path.extend(["../src", ".."])
from QUnfold.utils.custom_logger import INFO, RESULT
from studies.utils.ROOT_converter import (
    array_to_TH1,
    TH1_to_array,
    array_to_TH2,
)
from studies.utils.helpers import load_RooUnfold

# ROOT settings
load_RooUnfold()
r.gROOT.SetBatch(True)


def unfolder(type, m_response, h_pdata_reco, pdata_truth, dof):
    """
    Unfold a distribution based on a certain type of unfolding.

    Args:
        type (str): the unfolding type (MI, SVD, IBU).
        m_response (ROOT.TH2F): the response matrix.
        h_pdata_reco (ROOT.TH1F): the measured pseudo-data.
        pdata_truth (numpy.array): the truth pseudo-data.
        dof (int): the number of degrees-of-freedom used to compute chi2.

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

    # Generic unfolding settings
    unfolder.SetVerbose(0)
    unfolder.SetResponse(m_response)
    unfolder.SetMeasured(h_pdata_reco)
    histo = unfolder.Hunfold()
    histo.SetName("unfolded_{}".format(type))
    histo_mi_bin_c, histo_mi_bin_e = TH1_to_array(histo)

    # Print other information
    print("Bin contents: {}".format(histo_mi_bin_c))
    print("Bin errors: {}".format(histo_mi_bin_e))
    # chi2_mi, pval_mi = sc.chisquare(histo_mi_bin_c, pdata_truth)
    # print("chi2 / dof = {} / {} = {}".format(chi2_mi, dof, chi2_mi/float(dof)))

    return histo


def plot_unfolding(truth, reco, unfolded):

    # Basic properties
    canvas = r.TCanvas()
    unfolded.SetStats(0)
    truth.SetLineColor(2)
    unfolded.Draw()
    truth.Draw("same")
    reco.Draw("same")

    # Legend settings
    leg = r.TLegend(0.6, 0.6, 0.9, 0.9)
    leg.AddEntry(truth, "True distribution", "pl")
    leg.AddEntry(reco, "Measured distribution", "pl")
    leg.AddEntry(unfolded, "Unfolded distribution")
    leg.Draw()

    # Save canvas
    canvas.Draw()
    ext = unfolded.GetName().split("_")[-1]
    canvas.SaveAs("img/RooUnfold/unfolded_{}.png".format(ext))


def main():

    # Create dirs
    if not os.path.exists("img/RooUnfold"):
        os.makedirs("img/RooUnfold")

    # Read unfolding parameters from the input variables file
    input = pd.read_csv(args.input)
    truth = input["Truth"].to_numpy()
    pdata_truth = input["PseudoData"].to_numpy()
    INFO("Signal truth-level: {}".format(truth))
    INFO("Pseudo-data truth-level: {}".format(pdata_truth))

    # Read the response matrix
    response = np.loadtxt(args.response)
    INFO("Response matrix: \n{}".format(response))

    # Get reco variables (r = R*t)
    reco = np.dot(response, truth)
    pdata_reco = np.dot(response, pdata_truth)
    INFO("Signal reco-level: {}".format(reco))
    INFO("Pseudo-data reco-level: {}".format(pdata_reco))

    # Converting the numpy objects into ROOT histograms
    h_truth = array_to_TH1(truth, "truth")
    h_pdata_truth = array_to_TH1(pdata_truth, "data_truth")
    h_response = array_to_TH2(response, "response")
    h_reco = array_to_TH1(reco, "reco")
    h_pdata_reco = array_to_TH1(pdata_reco, "data_reco")

    # Initialize the RooUnfold response matrix
    m_response = r.RooUnfoldResponse(h_reco, h_truth, h_response)
    m_response.UseOverflow(False)  # disable the overflow bin which takes the outliers
    dof = h_truth.GetNbinsX() - 1

    # Performing the unfolding with different methods
    unfolded_MI = unfolder(
        "MI", m_response, h_pdata_reco, pdata_truth, dof
    )  # matrix inversion
    unfolded_IBU = unfolder(
        "IBU", m_response, h_pdata_reco, pdata_truth, dof
    )  # iterative Bayesian
    unfolded_SVD = unfolder(
        "SVD", m_response, h_pdata_reco, pdata_truth, dof
    )  # Tikhonov

    # Produce plots
    plot_unfolding(h_truth, h_reco, unfolded_MI)
    plot_unfolding(h_truth, h_reco, unfolded_IBU)
    plot_unfolding(h_truth, h_reco, unfolded_SVD)


if __name__ == "__main__":

    # Parser settings
    parser = ap.ArgumentParser(description="Parsing unfolding input variables.")
    parser.add_argument(
        "-i",
        "--input",
        default="../../data/distributions/peak.csv",
        help="Input data used for unfolding.",
    )
    parser.add_argument("-l", "--lreg", default=0.00, help="Regularization strength.")
    parser.add_argument(
        "-r",
        "--response",
        default="../../data/responses/nominal.txt",
        help="Regularization strength.",
    )
    args = parser.parse_args()

    # Run main function
    main()
