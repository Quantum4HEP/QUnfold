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
sys.path.append("../../src")
from QUnfold.utils.custom_logger import INFO, ERROR, RESULT
from studies.RooUnfold.utils.ROOT_converter import (
    array_to_TH1,
    TH1_to_array,
    array_to_TH2,
)

# Loading RooUnfold
loaded_RooUnfold = r.gSystem.Load("RooUnfold/libRooUnfold.so")
if not loaded_RooUnfold == 0:
    ERROR("RooUnfold not found!")
    sys.exit(0)

# ROOT settings
r.gROOT.SetBatch(True)


def main():

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

    # Saving the response matrix
    m_response_save = m_response.HresponseNoOverflow()
    m_response_canvas = r.TCanvas()
    m_response_save.SetStats(0)  # delete statistics box
    m_response_save.Draw("colz")  # to have heatmap
    m_response_canvas.Draw()
    if not os.path.exists("img"):
        os.makedirs("img")
    m_response_canvas.SaveAs("img/response.png")

    # Performing the unfolding with matrix inversion
    # unfolder_mi = r.RooUnfoldInvert("MI", "Matrix Inversion")
    # unfolder_mi.SetVerbose(0)
    # unfolder_mi.SetResponse(m_response)
    # unfolder_mi.SetMeasured(h_pdata_reco)
    # histo_mi = unfolder_mi.Hunfold()
    # histo_mi.SetName("unfolded_mi")
    # histo_mi_bin_c, histo_mi_bin_e = TH1_to_array(histo_mi)
    # RESULT("Unfolded with matrix inversion:")
    # print("Bin contents: {}".format(histo_mi_bin_c))
    # print("Bin errors: {}".format(histo_mi_bin_e))
    # chi2_mi, pval_mi = sc.chisquare(histo_mi_bin_c, pdata_truth)
    # print("chi2 / dof = {} / {} = {}".format(chi2_mi, dof, chi2_mi/float(dof)))

    # Performing the unfolding with IBU
    # ...

    # Performing the unfolding with SVD Tikhonov
    # unfolder_svd = r.RooUnfoldSvd("SVD", "SVD Tikhonov")
    # unfolder_svd.SetKterm(3)
    # unfolder_svd.SetVerbose(0)
    # unfolder_svd.SetResponse(m_response)
    # unfolder_svd.SetMeasured(h_pdata_reco)
    # histo_svd = unfolder_svd.Hunfold()
    # histo_svd.SetName("unfolder_svd")
    # histo_svd_bin_c, histo_svd_bin_e = TH1_to_array(histo_svd)
    # RESULT("Unfolded with SVD Tikhonov algorithm:")
    # print("Bin contents: {}".format(histo_svd_bin_c))
    # print("Bin errors: {}".format(histo_svd_bin_e))
    # chi2_svd, pval_svd = sc.chisquare(histo_svd_bin_c, pdata_truth)
    # print("chi2 / dof = {} / {} = {}".format(chi2_svd, dof, chi2_svd/float(dof)))


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
