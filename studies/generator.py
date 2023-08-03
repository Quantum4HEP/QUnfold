#!/usr/bin/env python
# -*- coding: utf-8 -*-

import ROOT


def smearing(xt, bias, smear, eff):
    """
    Apply a Gaussian smearing to the input sample xt.

    Args:
        xt (float): input sample value.
        bias (float): distortion bias.
        smear (float): distortion variance.
        eff (float): smearing efficiency.

    Returns:
        float: smeared output value.
        None: if the sample is filtered out because of limited efficiency.
    """
    from analysis import gRandom

    if gRandom.Rndm() > eff:
        return None
    return xt + gRandom.Gaus(bias, smear)


def generate_data(distr, samples, bins, min_bin, max_bin, bias, smear, eff):
    """
    Generate true/measured histograms and response matrix for a given distribution.

    Args:
        distr (str): name of the distribution.
        samples (int): number of data samples.
        bins (int): number of bins in the histograms.
        min_bin (float): minimum value of the histogram range.
        max_bin (float): maximum value of the histogram range.
        bias (float): distortion bias.
        smear (float): distortion variance.
        eff (float): smearing efficiency.

    Returns:
        ROOT.TH1F: true distribution histogram.
        ROOT.TH1F: measured distribution histogram.
        ROOT.RooUnfoldResponse: response object.
    """
    from analysis import distributions

    if distr == "double-peaked":  # needs custom function for generation
        return gen_double_peaked(samples, bins, min_bin, max_bin, bias, smear, eff)

    # Initialize ROOT histograms and response
    true = ROOT.TH1F(f"True {distr}", distr, bins, min_bin, max_bin)
    meas = ROOT.TH1F(f"Meas {distr}", distr, bins, min_bin, max_bin)
    response = ROOT.RooUnfoldResponse(bins, min_bin, max_bin)

    # Get ROOT random generator and distribution parameters
    RandGen = distributions[distr]["generator"]
    pars = distributions[distr]["parameters"]

    for _ in range(samples):
        # Fill true/meas histograms
        xt = RandGen(*pars)
        true.Fill(xt)
        x = smearing(xt, bias, smear, eff)
        if x != None:
            meas.Fill(x)
        # Fill response object
        xt = RandGen(*pars)
        x = smearing(xt, bias, smear, eff)
        if x != None:
            response.Fill(x, xt)
        else:
            response.Miss(xt)

    return true, meas, response


def gen_double_peaked(samples, bins, min_bin, max_bin, bias, smear, eff):
    """
    Generate true/measured histograms and response matrix for the double peaked distribution.

    Args:
        samples (int): number of data samples.
        bins (int): number of bins in the histograms.
        min_bin (float): minimum value of the histogram range.
        max_bin (float): maximum value of the histogram range.
        bias (float): distortion bias.
        smear (float): distortion variance.
        eff (float): smearing efficiency.

    Returns:
        ROOT.TH1F: true distribution histogram.
        ROOT.TH1F: measured distribution histogram.
        ROOT.RooUnfoldResponse: response object.
    """
    from analysis import distributions

    true = ROOT.TH1F("True double-peaked", "", bins, min_bin, max_bin)
    meas = ROOT.TH1F("Meas double-peaked", "", bins, min_bin, max_bin)
    response = ROOT.RooUnfoldResponse(bins, min_bin, max_bin)

    RandGen = distributions["double-peaked"]["generator"]
    parameters = distributions["double-peaked"]["parameters"]

    for _ in range(samples // 2):
        for pars in parameters:
            # Fill true/meas histograms
            xt = RandGen(*pars)
            true.Fill(xt)
            x = smearing(xt, bias, smear, eff)
            if x != None:
                meas.Fill(x)
            # Fill response object
            xt = RandGen(*pars)
            x = smearing(xt, bias, smear, eff)
            if x != None:
                response.Fill(x, xt)
            else:
                response.Miss(xt)

    return true, meas, response
