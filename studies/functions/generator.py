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
from utils.helpers import load_RooUnfold

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





