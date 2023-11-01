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


def TH1_to_array(histo, overflow=False):
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


def TH2_to_array(histo, overflow=False):
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

