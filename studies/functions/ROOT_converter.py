#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ---------------------- Metadata ----------------------
#
# File name:  ROOT_converter.py
# Author:     Gianluca Bianco (biancogianluca9@gmail.com)
# Date:       2023-06-14
# Copyright:  (c) 2023 Gianluca Bianco under the MIT license.

# Data science modules
import ROOT as r
import numpy as np


def array_to_TH1(bin_contents, bins, min_bin, max_bin, name="histo"):
    """
    Converts NumPy arrays representing bin contents of a ROOT.TH1F histogram.

    Args:
        bin_contents (numpy.array): The NumPy array representing bin contents.
        bins (int): The number of bins of the histogram.
        min_bin (int): The minimum bin value.
        max_bin (int): The maximum bin value.
        name (str): Name of the ROOT.TH1F histogram. Default is "hist".

    Returns:
        ROOT.TH1F: The converted ROOT.TH1F histogram.
    """

    # Initial settings
    histo = r.TH1F(name, ";X;Entries", bins, min_bin, max_bin)

    # Fill histogram with bin contents
    for i in range(bins):
        histo.SetBinContent(i + 1, bin_contents[i])

    return histo


def array_to_TH2(
    array, bins_x, min_bin_x, max_bin_x, bins_y, min_bin_y, max_bin_y, hname="res"
):
    """
    Convert a 2D numpy.array into a ROOT.TH2F.

    Args:
        array (numpy-array): a 2F numpy.array.
        bins_x (int): The number of bins of the histogram in the x-axis.
        min_bin_x (int): The minimum bin value of the x-axis.
        max_bin_x (int): The maximum bin value of the x-axis.
        bins_y (int): The number of bins of the histogram in the y-axis.
        min_bin_y (int): The minimum bin value in the y-axis.
        max_bin_y (int): The maximum bin value in the y-axis.
        hname (str, optional): The name of the ROOT.TH2F histogram. Defaults to "res".

    Returns:
        ROOT.TH2F: the converted ROOT.TH2F.
    """

    # Variables and histo properties
    histo = r.TH2F(
        hname, ";reco;truth", bins_x, min_bin_x, max_bin_x, bins_y, min_bin_y, max_bin_y
    )

    # Filling the histo
    for i in range(bins_x):
        for j in range(bins_y):
            histo.SetBinContent(i + 1, j + 1, array[i][j])

    return histo


def TH1_to_array(histo, overflow=True):
    """
    Convert a ROOT.TH1F into a numpy.array.

    Args:
        histo (ROOT.TH1F): the input ROOT.TH1F to be converted.
        overflow (bool): enable disable first and last bins overflow.

    Returns:
        numpy.array: a numpy.array of the histo bin contents.
    """

    if overflow:
        start, stop = 0, histo.GetNbinsX() + 2
    else:
        start, stop = 1, histo.GetNbinsX() + 1
    return np.array([histo.GetBinContent(i) for i in range(start, stop)])


def TH2_to_array(histo, overflow=True):
    """
    Convert a ROOT.TH2F object into a numpy.array.

    Parameters:
        hist (ROOT.TH2F): The TH2F object to convert.
        overflow (bool): enable disable first and last bins overflow.

    Returns:
        numpy_array (numpy.array): The numpy.array representing the contents of the TH2F.

    """

    if overflow:
        x_start, x_stop = 0, histo.GetNbinsX() + 2
        y_start, y_stop = 0, histo.GetNbinsY() + 2
    else:
        x_start, x_stop = 1, histo.GetNbinsX() + 1
        y_start, y_stop = 1, histo.GetNbinsY() + 1

    return np.array(
        [
            [histo.GetBinContent(i, j) for j in range(y_start, y_stop)]
            for i in range(x_start, x_stop)
        ]
    )


def remove_zero_entries_bins(g0, f0, response):
    """
    Remove zero entries bins from histograms g0 (measured) and f0 (truth) and the corresponding bins in the response matrix.

    Parameters:
        g0 (ROOT.TH1): The input measured histogram g0.
        f0 (ROOT.TH1): The input truth histogram f0 with the same binning as g0.
        response (ROOT.TH2): The response matrix with y-axis representing the bins of g0 and x-axis representing the bins of f0.

    Returns:
        tuple: A tuple containing three objects:
               g0_new (ROOT.TH1): The new measured histogram g0 with zero entries bins removed.
               f0_new (ROOT.TH1): The new truth histogram f0 with zero entries bins removed.
               response_new (ROOT.TH2): The new response matrix with corresponding bins removed.
    """

    # Find the first non-empty bin of g0
    first_non_empty_bin = next(
        (
            bin_idx
            for bin_idx in range(1, g0.GetNbinsX() + 1)
            if g0.GetBinContent(bin_idx) > 0
        ),
        None,
    )

    # Find the last non-empty bin of g0
    last_non_empty_bin = next(
        (
            bin_idx
            for bin_idx in range(g0.GetNbinsX(), first_non_empty_bin - 1, -1)
            if g0.GetBinContent(bin_idx) > 0
        ),
        None,
    )

    # Get the minimum and maximum values of the binning in g0
    x_min = g0.GetXaxis().GetBinLowEdge(first_non_empty_bin)
    x_max = g0.GetXaxis().GetBinUpEdge(last_non_empty_bin)

    # Count the number of non-empty bins in g0
    num_non_empty_bins = sum(
        1
        for bin_idx in range(first_non_empty_bin, last_non_empty_bin + 1)
        if g0.GetBinContent(bin_idx) > 0
    )

    # Create new histograms and matrices
    g0_new = r.TH1F(
        g0.GetName() + "_new", g0.GetTitle(), num_non_empty_bins, x_min, x_max
    )

    # Create a new histogram f0_new with the same binning as g0_new
    f0_new = r.TH1F(
        f0.GetName() + "_new", f0.GetTitle(), num_non_empty_bins, x_min, x_max
    )

    # Create a new response matrix with corresponding bins removed
    response_new = r.TH2F(
        response.GetName() + "_new",
        response.GetTitle(),
        num_non_empty_bins,
        x_min,
        x_max,
        num_non_empty_bins,
        x_min,
        x_max,
    )

    # Loop over all bins in g0
    new_bin_idx = 1
    for bin_idx in range(first_non_empty_bin, last_non_empty_bin + 1):
        bin_entries = g0.GetBinContent(bin_idx)

        # If the bin has more than 0 entries, fill histograms g0_new and f0_new with the content of the corresponding bin
        if bin_entries > 0:
            g0_new.SetBinContent(new_bin_idx, bin_entries)
            f0_new.SetBinContent(new_bin_idx, f0.GetBinContent(bin_idx))

            # Copy the corresponding row and column in the response matrix
            for y_bin_idx in range(1, response.GetNbinsY() + 1):
                response_new.SetBinContent(
                    new_bin_idx, y_bin_idx, response.GetBinContent(bin_idx, y_bin_idx)
                )
            for x_bin_idx in range(1, response.GetNbinsX() + 1):
                response_new.SetBinContent(
                    x_bin_idx, new_bin_idx, response.GetBinContent(x_bin_idx, bin_idx)
                )

            new_bin_idx += 1

    return g0_new, f0_new, response_new
