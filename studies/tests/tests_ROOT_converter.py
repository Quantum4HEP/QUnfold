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
sys.path.append("..")
from studies.utils.ROOT_converter import (
    array_to_TH1,
    TH1_to_array,
    array_to_TH2,
    TH2_to_array,
)


@given(
    bin_contents=arrays(
        dtype=np.float64,
        shape=40,
        elements=st.floats(0, 100),
    ),
    bin_errors=arrays(
        dtype=np.float64,
        shape=40,
        elements=st.floats(0, 100),
    ),
)
def test_array_to_TH1(bin_contents, bin_errors):
    """
    Test the array_to_TH1 function.

    Args:
        bin_contents (np.array): The NumPy array representing bin contents.
        bin_errors (np.array): The NumPy array representing bin errors.
    """

    # Variables
    histo = array_to_TH1(bin_contents, bin_errors)

    # Tests
    assert type(histo) == r.TH1F  # type check
    assert histo.GetNbinsX() == 40  # n_bins check
    assert histo.GetName() == "histo"  # check histo name
    assert histo.GetTitle() == ""  # check histo title
    for i in range(40):  # values check
        assert histo.GetBinContent(i + 1) == pt.approx(bin_contents[i])
        assert histo.GetBinError(i + 1) == pt.approx(bin_errors[i])


def test_TH1_to_array():
    """
    Testing the TH1_to_array function properties. Note that hypothesis is not used since it doesn't support ROOT types.
    """

    # Variables
    histo = r.TH1F("Test", ";X;Entries", 5, 0.5, 5.5)
    for i in range(histo.GetNbinsX()):
        histo.SetBinContent(i + 1, i)
    bin_contents, bin_errors = TH1_to_array(histo)

    # Tests
    assert bin_contents.shape[0] == 5  # array size
    assert bin_errors.shape[0] == 5  # array size
    nptest.assert_array_equal(
        bin_contents, np.array([0, 1, 2, 3, 4])
    )  # bin contents equality
    nptest.assert_allclose(
        bin_errors, np.array([0, 1, 1.414214, 1.732051, 2]), rtol=1e-5
    )  # bin errors equality


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
    histo = array_to_TH2(array)

    # Tests
    with pt.raises(AssertionError):
        array_to_TH2(np.array([]))  # check the array dimension
    assert histo.GetNbinsX() == 40  # n_bins check
    assert histo.GetNbinsY() == 40  # n_bins check
    assert histo.GetName() == "res"  # check histo name
    assert histo.GetTitle() == ""  # check histo title
    for i in range(40):  # values check
        for j in range(40):
            assert histo.GetBinContent(i + 1, j + 1) == pt.approx(array[i][j])
            assert histo.GetBinError(i + 1, j + 1) == pt.approx(array[i][j])


def test_TH2_to_array():
    """
    Testing the TH2F_to_numpy function properties. Note that hypothesis is not used since it doesn't support ROOT types.
    """

    # Create TH2F histogram
    histo = r.TH2F("Test", ";X;Y", 5, 0.5, 5.5, 3, 0.5, 3.5)
    for i in range(histo.GetNbinsX()):
        for j in range(histo.GetNbinsY()):
            bin_content = i * j
            histo.SetBinContent(i + 1, j + 1, bin_content)

    # Convert TH2F to NumPy array
    numpy_array = TH2_to_array(histo)

    # Tests
    assert numpy_array.shape == (5, 3)  # array shape
    nptest.assert_array_equal(
        numpy_array,
        np.array([[0, 0, 0], [0, 1, 2], [0, 2, 4], [0, 3, 6], [0, 4, 8]]),
    )  # array content equality
