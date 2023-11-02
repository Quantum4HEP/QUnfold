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
