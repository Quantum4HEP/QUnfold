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

# Utils modules
sys.path.append("..")
from studies.functions.extra_ROOT_converter import array_to_TH1, array_to_TH2


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
