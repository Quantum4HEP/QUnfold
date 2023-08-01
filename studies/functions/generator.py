#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ---------------------- Metadata ----------------------
#
# File name:  generator.py
# Author:     Gianluca Bianco (biancogianluca9@gmail.com)
# Date:       2023-06-14
# Copyright:  (c) 2023 Gianluca Bianco under the MIT license.

# Followed the guide at: https://statisticalmethods.web.cern.ch/StatisticalMethods/unfolding/RooUnfold_01-Methods_PY/#Aproximating-Smearing

# STD modules
import os

# Data science modules
import ROOT as r
import numpy as np

# Utils modules
from functions.helpers import load_RooUnfold

# ROOT settings
load_RooUnfold()
r.gROOT.SetBatch(True)


def smear(xt, eff=1):
    """
    Applies a Gaussian smearing effect to a given input value, xt.

    Args:
        xt (float): The input value to apply the smearing to.
        eff (float): smearing efficiency.

    Returns:
        float or None: The resulting value after applying the smearing. Returns None if the value is filtered based on efficiency.
    """
    x = r.gRandom.Rndm()
    if x > eff:
        return None
    xsmear = r.gRandom.Gaus(2.5, 0.2)  #  bias and smear
    return xt + xsmear


def generate_standard(truth, meas, response, type, distr, samples):
    """
    Generate data for standard distributions.

    Args:
        truth (ROOT.TH1F): truth histogram.
        meas (ROOT.TH1F): measured histogram.
        response (ROOT.TH2F): response matrix.
        type (str): type of data generation (data or response).
        distr (distr): the distribution to be generated.
        samples (int): number of samples to be generated.

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
            elif distr == "exponential":
                xt = r.gRandom.Exp(1.0)
            truth.Fill(xt)
            x = smear(xt)
            if x != None:
                meas.Fill(x)

        return truth, meas

    # Response generation
    elif type == "response":
        r.gRandom.SetSeed(556)
        for i in range(samples):
            xt = 0
            if distr == "breit-wigner":
                xt = r.gRandom.BreitWigner(0.3, 2.5)
            elif distr == "normal":
                xt = r.gRandom.Gaus(0.0, 2.0)
            elif distr == "exponential":
                xt = r.gRandom.Exp(1.0)
            x = smear(xt)
            if x != None:
                response.Fill(x, xt)
            else:
                response.Miss(xt)

        return response


def generate_double_peaked(truth, meas, response, type, samples):
    """
    Generate data for the double peaked distributions.

    Args:
        truth (ROOT.TH1F): truth histogram.
        meas (ROOT.TH1F): measured histogram.
        response (ROOT.TH2F): response matrix.
        type (str): type of data generation (data or response).
        samples (int): number of samples to be generated.

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
            truth.Fill(xt)
            x = r.gRandom.Gaus(
                xt,
            )
            if x != None:
                meas.Fill(x)
        for i in range(samples):
            xt = r.gRandom.Gaus(-2, 1.5)
            truth.Fill(xt)
            x = r.gRandom.Gaus(xt, 1)
            if x != None:
                meas.Fill(x)

        return truth, meas

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

def generate(distr, bins, min_bin, max_bin, samples):
    """
    Generate simulated data and response for a given distribution.

    Args:
        distr (str): The name of the distribution to generate data from.
        bins (int): The number of bins in the histograms.
        min_bin (float): The minimum value of the histogram range.
        max_bin (float): The maximum value of the histogram range.
        samples (int): The number of data samples to generate.

    Returns:
        ROOT.TH1F: The histogram representing the truth distribution.
        ROOT.TH1F: The histogram representing the measured distribution.
        ROOT.RooUnfoldResponse: The response object used for unfolding.
    """
    
    # Case for exponential
    if distr == "exponential":
        min_bin = 0

    # Initialize variables
    truth = r.TH1F("Truth", "", bins, min_bin, max_bin)
    meas = r.TH1F("Measured", "", bins, min_bin, max_bin)
    response = r.RooUnfoldResponse(bins, min_bin, max_bin)
    
    # Fill histograms
    if any(d in distr for d in ["normal", "breit-wigner", "exponential"]):
        truth, meas = generate_standard(truth, meas, response, "data", distr, samples)
        response = generate_standard(truth, meas, response, "response", distr, samples)
    elif any(d in distr for d in ["double-peaked"]):
        truth, meas = generate_double_peaked(truth, meas, response, "data", samples)
        response = generate_double_peaked(truth, meas, response, "response", samples)
    
    return truth, meas, response