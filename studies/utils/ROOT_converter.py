#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Tue Jun 13 17:10:00 2023
Author: Gianluca Bianco
"""

import sys

# Data science modules
import ROOT as r
import numpy as np


def array_to_TH1(array, hname="histo"):
    """
    Convert a numpy.array into a ROOT.TH1F histogram.

    Args:
        array (numpy.array): the numpy.array to be converted.
        hname (str, optional): The name of the new ROOT histogram. Defaults to "h".

    Returns:
        ROOT.TH1F: the new ROOT.TH1F histogram given from the numpy.array conversion.
    """

    # Declaring histo properties
    n_bins = array.shape[0]
    histo = r.TH1F(hname, ";X;Entries", n_bins, 0.5, n_bins + 0.5)

    # Initialize the histogram
    for i in range(n_bins):
        histo.SetBinContent(i + 1, array[i])

    return histo


def TH1_to_array(histo):
    """
    Convert a ROOT.TH1F into a numpy.array.

    Args:
        histo (ROOT.TH1F): the input ROOT.TH1F to be converted.

    Returns:
        numpy.array: a numpy.array of the histo bin contents.
        numpy.array: a numpy.array of the histo bin errors.
    """

    n_bins = histo.GetNbinsX()
    bin_contents = np.array([histo.GetBinContent(i + 1) for i in range(n_bins)])
    bin_errors = np.array([histo.GetBinError(i + 1) for i in range(n_bins)])

    return bin_contents, bin_errors


def array_to_TH2(array, hname="res", factor=0.10):
    """
    Convert a 2D numpy.array into a ROOT.TH2F.

    Args:
        array (numpy-array): a 2F numpy.array.
        hname (str, optional): The name of the ROOT.TH2F histogram. Defaults to "res".
        factor (float, optional): Scaling factor for the ROOT.TH2F bin errors. Defaults to 0.10.

    Returns:
        ROOT.TH2F: the converted ROOT.TH2F.
    """

    # Sanity check
    assert len(array.shape) == 2

    # Variables and histo properties
    n_bins_x = array.shape[0]
    n_bins_y = array.shape[1]
    histo = r.TH2F(
        hname,
        ";reco;truth",
        n_bins_x,
        0.5,
        n_bins_x + 0.5,
        n_bins_y,
        0.5,
        n_bins_y + 0.5,
    )

    # Filling the histo
    for i in range(n_bins_x):
        for j in range(n_bins_y):
            histo.SetBinContent(i + 1, j + 1, array[i][j])
            histo.SetBinError(i + 1, j + 1, factor * array[i][j])

    return histo


def TH2_to_array(histo):
    """
    Convert a ROOT.TH2F object into a numpy.array.

    Parameters:
        hist (ROOT.TH2F): The TH2F object to convert.

    Returns:
        numpy_array (numpy.array): The numpy.array representing the contents of the TH2F.

    """

    # Variables
    num_bins_x = histo.GetNbinsX()
    num_bins_y = histo.GetNbinsY()
    numpy_array = np.zeros((num_bins_x, num_bins_y))

    # Filling the array
    for i in range(1, num_bins_x + 1):
        for j in range(1, num_bins_y + 1):
            bin_content = histo.GetBinContent(i, j)
            numpy_array[i - 1, j - 1] = bin_content

    return numpy_array
