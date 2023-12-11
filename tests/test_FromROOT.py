#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ---------------------- Metadata ----------------------
#
# File name:  test_FromROOT.py
# Author:     Gianluca Bianco (biancogianluca9@gmail.com)
# Date:       2023-11-10
# Copyright:  (c) 2023 Gianluca Bianco under the MIT license.

import ROOT as r
import numpy as np
import numpy.testing as nptest
from QUnfold.utility import (
    TH1_to_array,
    TH2_to_array,
    TMatrix_to_array,
    TVector_to_array,
)


def test_TH1_to_array():
    """
    Testing the TH1_to_array function properties. Note that hypothesis is not used since it doesn't support ROOT types.
    """

    # Variables
    histo = r.TH1F("Test", ";X;Entries", 5, 0, 5)
    for i in range(histo.GetNbinsX()):
        histo.SetBinContent(i + 1, i)

    # Test no overflow case
    bin_contents = TH1_to_array(histo, overflow=False)
    assert bin_contents.shape[0] == 5  # array size
    nptest.assert_array_equal(
        bin_contents, np.array([0, 1, 2, 3, 4])
    )  # bin contents equality

    # Test overflow case
    bin_contents = TH1_to_array(histo, overflow=True)
    assert bin_contents.shape[0] == 7  # array size
    nptest.assert_array_equal(
        bin_contents, np.array([0, 0, 1, 2, 3, 4, 0])
    )  # bin contents equality


def test_TH2_to_array():
    """
    Testing the TH2F_to_numpy function properties. Note that hypothesis is not used since it doesn't support ROOT types.
    """

    # Create TH2F histogram
    histo = r.TH2F("Test", ";X;Y", 5, 0.5, 5.5, 3, 0.5, 3.5)
    for i in range(histo.GetNbinsX()):
        for j in range(histo.GetNbinsY()):
            histo.SetBinContent(i + 1, j + 1, i * j)

    # Convert TH2F to NumPy array
    numpy_array = TH2_to_array(histo, overflow=False)

    # Tests
    assert numpy_array.shape == (5, 3)  # array shape
    nptest.assert_array_equal(
        numpy_array,
        np.array([[0, 0, 0], [0, 1, 2], [0, 2, 4], [0, 3, 6], [0, 4, 8]]),
    )  # array content equality


def test_TVector_to_array():
    """
    Testing the TVector_to_array function properties.
    """

    # Create TVectorD
    vector = r.TVectorD(3)
    for i in range(vector.GetNoElements()):
        vector[i] = i

    # Convert TVectorD to NumPy array
    numpy_array = TVector_to_array(vector)

    # Tests
    assert numpy_array.shape[0] == 3  # array size
    nptest.assert_array_equal(
        numpy_array, np.array([0, 1, 2])
    )  # array content equality


def test_TMatrix_to_array():
    """
    Testing the TMatrix_to_array function properties.
    """

    # Create TMatrixD
    matrix = r.TMatrixD(2, 2)
    for i in range(matrix.GetNrows()):
        for j in range(matrix.GetNcols()):
            matrix[i][j] = i * j

    # Convert TMatrixD to NumPy array
    numpy_array = TMatrix_to_array(matrix)

    # Tests
    assert numpy_array.shape == (2, 2)  # array shape
    nptest.assert_array_equal(
        numpy_array, np.array([[0, 0], [0, 1]])
    )  # array content equality
