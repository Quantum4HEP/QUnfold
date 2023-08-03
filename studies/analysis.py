#!/usr/bin/env python
# -*- coding: utf-8 -*-

import ROOT
import numpy as np

from generator import *
from roounfold import *
from qunfold import *
from root_converter import *


# Load RooUnfold library from local installation
ROOT.gSystem.Load("../HEP_deps/RooUnfold/libRooUnfold.so")

# Get ROOT random generator and set random seed
gRandom = ROOT.gRandom
gRandom.SetSeed(42)

############################## CONFIG INPUT VARIABLES ############################
##################################################################################
distributions = {
    "normal": {
        "generator": gRandom.Gaus,
        "parameters": (5.0, 1.4),  # (mu, sigma)
    },
    "breit-wigner": {
        "generator": gRandom.BreitWigner,
        "parameters": (5.3, 2.1),  # (mu, gamma)
    },
    "exponential": {
        "generator": gRandom.Exp,
        "parameters": (2.5,),  # (tau,)
    },
    "double-peaked": {
        "generator": gRandom.Gaus,
        "parameters": ((3.4, 1.1), (6.7, 0.8)),  # ((mu1, sigma1), (mu2, sigma2))
    },
}
samples = 10000
bins = 40
min_bin = 0.0
max_bin = 10.0
bias = -0.8
smear = 0.4
eff = 1.0
##################################################################################
##################################################################################


def main():
    for distr in distributions:
        true, meas, response = generate_data(
            distr, samples, bins, min_bin, max_bin, bias, smear, eff
        )
        roounfold_plot_response(response, distr)

        ########################## Classical ##########################
        # Response Matrix Inversion (RMI)
        unfolded_RMI = roounfold_unfolder(response, meas, method="RMI")
        roounfold_plot_results(true, meas, unfolded_RMI, distr)

        # Iterative Bayesian Unfolding (IBU)
        unfolded_IBU = roounfold_unfolder(response, meas, method="IBU")
        roounfold_plot_results(true, meas, unfolded_IBU, distr)

        # SVD Tikhonov unfolding (SVD)
        unfolded_SVD = roounfold_unfolder(response, meas, method="SVD")
        roounfold_plot_results(true, meas, unfolded_SVD, distr)

        # Bin-by-Bin unfolding (B2B)
        unfolded_B2B = roounfold_unfolder(response, meas, method="B2B")
        roounfold_plot_results(true, meas, unfolded_B2B, distr)

        ########################### Quantum ###########################
        response = TMatrix_to_array(response.Mresponse())
        meas = TH1_to_array(meas, overflow=False)
        true = TH1_to_array(true, overflow=False)

        # Simulated Annealing unfolding (SA)
        unfolded_SA = qunfold_unfolder(response, meas, true)
        binning = np.linspace(min_bin, max_bin, bins + 1)
        qunfold_plot_results(true, meas, unfolded_SA, distr, binning)


if __name__ == "__main__":
    main()
