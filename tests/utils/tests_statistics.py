#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ---------------------- Metadata ----------------------
#
# File name:  tests_statistics.py.py
# Author:     Gianluca Bianco (biancogianluca9@gmail.com)
# Date:       2023-06-18
# Copyright:  (c) 2023 Gianluca Bianco under the MIT license.

# Data science modules
import numpy as np

# Test modules
from hypothesis import given
import hypothesis.strategies as st
from hypothesis.extra.numpy import arrays

# My modules
from QUnfold.utils import is_histogram


@given(
    bin_contents=arrays(
        dtype=np.float64,
        shape=st.integers(min_value=1, max_value=100),
        elements=st.floats(0, 100),
    ),
    bin_edges=arrays(
        dtype=np.intc,
        shape=st.integers(min_value=2, max_value=101),
        elements=st.integers(0, 100),
    ),
)
def test_is_histogram(bin_contents, bin_edges):
    """
    Test is_histogram function.

    Args:
        bin_contents (numpy.array): The numpy.array representing bin contents.
        bin_edges (numpy.array): The numpy.array representing bin edges.
    """

    # Create a numpy histogram
    bin_edges = np.sort(bin_edges)
    histogram = np.histogram(bin_contents, bin_edges)

    # Check if the created histogram is recognized as a numpy.histogram
    assert is_histogram(histogram) == True
