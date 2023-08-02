#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ---------------------- Metadata ----------------------
#
# File name:  tests_ROOT_converter.py
# Author:     Gianluca Bianco (biancogianluca9@gmail.com)
# Date:       2023-06-13
# Copyright:  (c) 2023 Gianluca Bianco under the MIT license.

# STD modules
import sys

# Data science modules
import ROOT as r
import numpy as np

# Testing modules
import pytest as pt
from hypothesis import given
import hypothesis.strategies as st
from hypothesis.extra.numpy import arrays
import numpy.testing as nptest

# Utils modules
from functions.ROOT_converter import (
    array_to_TH1,
    TH1_to_array,
    array_to_TH2,
    TH2_to_array,
    remove_zero_entries_bins,
)


@given(
    bin_contents=arrays(
        dtype=np.float64,
        shape=40,
        elements=st.floats(0, 100),
    )
)
def test_array_to_TH1(bin_contents):
    """
    Test the array_to_TH1 function.

    Args:
        bin_contents (np.array): The NumPy array representing bin contents.
    """

    # Variables
    histo = array_to_TH1(bin_contents, 40, 10, -10)

    # Tests
    assert type(histo) == r.TH1F  # type check
    assert histo.GetNbinsX() == 40  # n_bins check
    assert histo.GetName() == "histo"  # check histo name
    assert histo.GetTitle() == ""  # check histo title
    for i in range(40):  # values check
        assert histo.GetBinContent(i + 1) == pt.approx(bin_contents[i])


@given(
    array=arrays(
        dtype=np.float64,
        shape=(40, 40),
        elements=st.floats(0, 100),
    )
)
def test_array_to_TH2(array):
    """
    Testing the array_to_TH2 function properties.

    Args:
        array (np.array): the numpy.array used for testing the function.
    """

    # Variables
    histo = array_to_TH2(array, 40, -10, 10, 40, -10, -10)

    # Tests
    assert histo.GetNbinsX() == 40  # n_bins check
    assert histo.GetNbinsY() == 40  # n_bins check
    assert histo.GetName() == "res"  # check histo name
    assert histo.GetTitle() == ""  # check histo title
    for i in range(40):  # values check
        for j in range(40):
            assert histo.GetBinContent(i + 1, j + 1) == pt.approx(array[i][j])


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


def test_remove_zero_entries_bins():
    """
    Testing the remove_zero_entries_bins function.
    """

    # Create mock histograms with 10 bins and fill some random entries
    g0 = r.TH1F("g0", "Mock Histogram g0", 5, 0, 5)
    f0 = r.TH1F("f0", "Mock Histogram f0", 5, 0, 5)
    response = r.TH2F("response", "Mock Response Matrix", 5, 0, 5, 5, 0, 5)

    # Fill some random entries in g0, f0, and the response matrix
    g0.SetBinContent(1, 0)
    g0.SetBinContent(2, 4)
    g0.SetBinContent(3, 4)
    g0.SetBinContent(4, 1)
    g0.SetBinContent(5, 0)

    f0.SetBinContent(1, 2)
    f0.SetBinContent(2, 4)
    f0.SetBinContent(3, 4)
    f0.SetBinContent(4, 1)
    f0.SetBinContent(5, 5)

    response.SetBinContent(1, 1, 1)
    response.SetBinContent(2, 1, 2)
    response.SetBinContent(3, 1, 3)
    response.SetBinContent(4, 1, 4)
    response.SetBinContent(5, 1, 5)

    response.SetBinContent(1, 2, 2)
    response.SetBinContent(2, 2, 4)
    response.SetBinContent(3, 2, 5)
    response.SetBinContent(4, 2, 8)
    response.SetBinContent(5, 2, 9)

    response.SetBinContent(1, 3, 3)
    response.SetBinContent(2, 3, 5)
    response.SetBinContent(3, 3, 6)
    response.SetBinContent(4, 3, 9)
    response.SetBinContent(5, 3, 10)

    response.SetBinContent(1, 4, 4)
    response.SetBinContent(2, 4, 8)
    response.SetBinContent(3, 4, 9)
    response.SetBinContent(4, 4, 16)
    response.SetBinContent(5, 4, 17)

    response.SetBinContent(1, 5, 5)
    response.SetBinContent(2, 5, 10)
    response.SetBinContent(3, 5, 11)
    response.SetBinContent(4, 5, 17)
    response.SetBinContent(5, 5, 19)

    # Verify that the bin contents with 0 entries are removed in both histograms and the response matrix
    g0_n, f0_n, response_n = remove_zero_entries_bins(g0, f0, response)
    for bin in range(1, g0_n.GetNbinsX() + 1):
        assert g0_n.GetBinContent(bin) != 0
        assert g0_n.GetNbinsX() == 3
        assert f0_n.GetNbinsX() == 3
        assert response_n.GetNbinsX() == 3
        assert response_n.GetNbinsY() == 3

    # Verify that the bin contents are not removed if there are no 0-entries bins
    h0 = r.TH1F("h0", "Mock Histogram h0", 5, 0, 5)
    h0.SetBinContent(1, 5)
    h0.SetBinContent(2, 4)
    h0.SetBinContent(3, 4)
    h0.SetBinContent(4, 1)
    h0.SetBinContent(5, 5)

    h0_n, f0_n, response_n = remove_zero_entries_bins(h0, f0, response)
    for bin in range(1, h0_n.GetNbinsX() + 1):
        assert h0_n.GetBinContent(bin) != 0
        assert h0_n.GetNbinsX() == 5
        assert f0_n.GetNbinsX() == 5
        assert response_n.GetNbinsX() == 5
        assert response_n.GetNbinsY() == 5
