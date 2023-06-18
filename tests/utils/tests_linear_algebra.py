#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ---------------------- Metadata ----------------------
#
# File name:  tests_linear_algebra.py.py
# Author:     Gianluca Bianco (biancogianluca9@gmail.com)
# Date:       2023-06-17
# Copyright:  (c) 2023 Gianluca Bianco under the MIT license.

# Data science modules
import numpy as np

# Test modules
from hypothesis import given
import hypothesis.strategies as st
from hypothesis.extra.numpy import arrays

# My modules
from QUnfold.utils import is_matrix


@given(
    input_array=arrays(
        dtype=np.float64,
        shape=st.integers(min_value=1, max_value=100),
        elements=st.floats(0, 100),
    )
)
def test_is_matrix(input_array):
    """
    Test function to verify if an input array is a matrix.

    Args:
        input_array (numpy.ndarray): The input array to be tested.
    """
    if np.array(input_array).ndim == 2:
        assert is_matrix(np.array(input_array)) == True
    else:
        assert is_matrix(np.array(input_array)) == False
