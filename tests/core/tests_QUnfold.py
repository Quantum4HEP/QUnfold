#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ---------------------- Metadata ----------------------
#
# File name:  tests_QUnfold.py
# Author:     Gianluca Bianco (biancogianluca9@gmail.com)
# Date:       2023-06-18
# Copyright:  (c) 2023 Gianluca Bianco under the MIT license.

# Data science modules
import numpy as np

# Test modules
from hypothesis import given
import hypothesis.strategies as st
from hypothesis.extra.numpy import arrays
import pytest

# My modules
from QUnfold import QUnfold
from QUnfold.utils import is_matrix


@given(
    response=arrays(
        dtype=np.float64,
        shape=st.integers(min_value=1, max_value=100),
        elements=st.floats(0, 100),
    ),
    bin_contents=arrays(
        dtype=np.float64,
        shape=st.integers(min_value=1, max_value=1),
        elements=st.floats(0, 100),
    ),
)
def test_QUnfold_constructor(response, bin_contents):

    # Declaring the histogram
    hist_values = np.histogram(bin_contents, bins=40, range=(-10, 10))

    # Check the 1-dim constructor
    unfolded_1 = QUnfold()
    assert unfolded_1.measured == None
    assert unfolded_1.response == None

    # Check the 2-dim constructor
    if is_matrix(response):
        unfolded_2 = QUnfold(response, hist_values)
        assert unfolded_2.measured == hist_values, True
        assert unfolded_2.response == response, True
    else:
        with pytest.raises(AssertionError):
            QUnfold(response, hist_values)

    # Check >2-dim constructor
    with pytest.raises(AssertionError):
        QUnfold(response)
        QUnfold(response, response)
        QUnfold(response, hist_values, hist_values)
