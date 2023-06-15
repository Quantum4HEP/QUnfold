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


def array_to_TH1(bin_contents, bin_errors, name="histo"):
    """
    Converts NumPy arrays representing bin contents and bin errors to a ROOT.TH1F histogram.

    Args:
        bin_contents (numpy.array): The NumPy array representing bin contents.
        bin_errors (numpy.array): The NumPy array representing bin errors.
        name (str): Name of the ROOT.TH1F histogram. Default is "hist".

    Returns:
        ROOT.TH1F: The converted ROOT.TH1F histogram with bin errors.
    """

    # Initial settings
    n_bins = len(bin_contents)
    hist = r.TH1F(name, ";X;Entries", 40, -10, 10)

    # Fill histogram with bin contents and errors
    for i in range(n_bins):
        hist.SetBinContent(i + 1, bin_contents[i])
        hist.SetBinError(i + 1, bin_errors[i])

    return hist


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


def array_to_TH2(array, hname="res"):
    """
    Convert a 2D numpy.array into a ROOT.TH2F.

    Args:
        array (numpy-array): a 2F numpy.array.
        hname (str, optional): The name of the ROOT.TH2F histogram. Defaults to "res".

    Returns:
        ROOT.TH2F: the converted ROOT.TH2F.
    """

    # Sanity check
    assert len(array.shape) == 2

    # Variables and histo properties
    n_bins_x = array.shape[0]
    n_bins_y = array.shape[1]
    histo = r.TH2F(hname, ";reco;truth", 40, -10.0, 10.0, 40, -10.0, 10.0)

    # Filling the histo
    for i in range(n_bins_x):
        for j in range(n_bins_y):
            histo.SetBinContent(i + 1, j + 1, array[i][j])
            histo.SetBinError(i + 1, j + 1, array[i][j])

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
