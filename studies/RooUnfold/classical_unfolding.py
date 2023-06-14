#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Tue Jun 13 17:10:00 2023
Author: Gianluca Bianco
"""

# Standard modules
import argparse as ap
import sys

# Data science modules
import ROOT as r
import pandas as pd
import numpy as np

# Utils modules
sys.path.append("../../src")
from QUnfold.utils.custom_logging import INFO, ERROR
from utils.ROOT_converter import array_to_TH1, TH1_to_array, array_to_TH2

# Loading RooUnfold
loaded_RooUnfold = r.gSystem.Load("RooUnfold/libRooUnfold.so")
if not loaded_RooUnfold == 0:
    ERROR("RooUnfold not found!")
    sys.exit(0)


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
    h_truth = array_to_TH1(truth)
    h_pdata_truth = array_to_TH1(h_pdata_truth)
    h_response = array_to_TH2(response)
    h_reco = array_to_TH1(reco)
    h_pdata_reco = array_to_TH1(pdata_reco)

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
