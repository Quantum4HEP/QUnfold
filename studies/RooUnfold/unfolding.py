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
import numpy as np

# Utils modules
sys.path.extend(["../src", ".."])
from QUnfold.utils.custom_logger import RESULT
from studies.utils.ROOT_converter import (
    array_to_TH1,
    TH1_to_array,
    array_to_TH2,
)
from studies.utils.helpers import load_RooUnfold

# ROOT settings
load_RooUnfold()
r.gROOT.SetBatch(True)


def unfolder(type, m_response, h_meas, h_truth, dof):
    """
    Unfold a distribution based on a certain type of unfolding.

    Args:
        type (str): the unfolding type (MI, SVD, IBU).
        m_response (ROOT.TH2F): the response matrix.
        h_meas (ROOT.TH1F): the measured pseudo-data.
        h_truth (ROOT.TH1F): the truth pseudo-data.
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
    unfolder.SetMeasured(h_meas)
    histo = unfolder.Hunfold()
    histo.SetName("unfolded_{}".format(type))
    histo_mi_bin_c, histo_mi_bin_e = TH1_to_array(histo)

    # Print other information
    print("Bin contents: {}".format(histo_mi_bin_c))
    print("Bin errors: {}".format(histo_mi_bin_e))
    chi2 = histo.Chi2Test(h_truth, "WW")
    print("chi2 / dof = {} / {} = {}".format(chi2, dof, chi2 / float(dof)))

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
    canvas.SaveAs("../img/studies/RooUnfold/{}/unfolded_{}.png".format(args.distr, ext))


def main():

    # Create dirs
    if not os.path.exists("../img/studies/RooUnfold/{}".format(args.distr)):
        os.makedirs("../img/studies/RooUnfold/{}".format(args.distr))

    # Variables
    truth_bin_content_path = "../data/{}/truth_bin_content.txt".format(args.distr)
    truth_bin_err_path = "../data/{}/truth_bin_err.txt".format(args.distr)
    meas_bin_content_path = "../data/{}/meas_bin_content.txt".format(args.distr)
    meas_bin_err_path = "../data/{}/meas_bin_err.txt".format(args.distr)
    response_path = "../data/{}/response.txt".format(args.distr)

    # Load histograms and response from file
    np_truth_bin_content = np.loadtxt(truth_bin_content_path)
    np_truth_bin_err = np.loadtxt(truth_bin_err_path)
    np_meas_bin_content = np.loadtxt(meas_bin_content_path)
    np_meas_bin_err = np.loadtxt(meas_bin_err_path)
    np_response = np.loadtxt(response_path)

    # Convert to ROOT variables
    h_truth = array_to_TH1(np_truth_bin_content, np_truth_bin_err, "truth")
    h_meas = array_to_TH1(np_meas_bin_content, np_meas_bin_err, "meas")
    h_response = array_to_TH2(np_response, "response")

    # Initialize the RooUnfold response matrix
    m_response = r.RooUnfoldResponse(h_meas, h_truth, h_response)
    m_response.UseOverflow(False)  # disable the overflow bin which takes the outliers
    dof = h_truth.GetNbinsX() - 1

    # Performing the unfolding with different methods
    unfolded_MI = unfolder("MI", m_response, h_meas, h_truth, dof)  # matrix inversion
    print()
    unfolded_IBU = unfolder(
        "IBU", m_response, h_meas, h_truth, dof
    )  # iterative Bayesian
    print()
    unfolded_SVD = unfolder("SVD", m_response, h_meas, h_truth, dof)  # Tikhonov

    # Produce plots
    plot_unfolding(h_truth, h_meas, unfolded_MI)
    plot_unfolding(h_truth, h_meas, unfolded_IBU)
    plot_unfolding(h_truth, h_meas, unfolded_SVD)


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
