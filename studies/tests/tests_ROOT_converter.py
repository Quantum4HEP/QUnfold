#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Tue Jun 13 17:10:00 2023
Author: Gianluca Bianco
"""

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
)


@given(
    array=arrays(
        dtype=np.float64,
        shape=st.integers(min_value=1, max_value=100),
        elements=st.floats(0, 100),
    )
)
def test_array_to_TH1(array):
    """
    Testing the array_to_TH1 function properties.

    Args:
        array (np.array): the numpy.array used for testing the function.
    """

    # Variables
    histo = array_to_TH1(array)

    # Tests
    assert type(histo) == r.TH1F  # type check
    assert histo.GetNbinsX() == array.shape[0]  # n_bins check
    assert histo.GetName() == "histo"  # check histo name
    assert histo.GetTitle() == ""  # check histo title
    for i in range(array.shape[0]):  # values check
        assert histo.GetBinContent(i + 1) == pt.approx(array[i])


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
        shape=st.tuples(
            st.integers(min_value=1, max_value=100),
            st.integers(min_value=1, max_value=100),
        ),
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
    assert histo.GetNbinsX() == array.shape[0]  # n_bins check
    assert histo.GetNbinsY() == array.shape[1]  # n_bins check
    assert histo.GetName() == "res"  # check histo name
    assert histo.GetTitle() == ""  # check histo title
    for i in range(array.shape[0]):  # values check
        for j in range(array.shape[1]):
            assert histo.GetBinContent(i + 1, j + 1) == pt.approx(array[i][j])
            assert histo.GetBinError(i + 1, j + 1) == pt.approx(0.1 * array[i][j])
