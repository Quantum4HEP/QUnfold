#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ---------------------- Metadata ----------------------
#
# File name:  tests_binary_encoder.py
# Author:     Gianluca Bianco (biancogianluca9@gmail.com)
# Date:       2023-06-21
# Copyright:  (c) 2023 Gianluca Bianco under the MIT license.

# Data science modules
import numpy as np

# Test modules
from hypothesis import given
import hypothesis.strategies as st
from hypothesis.extra.numpy import arrays
import pytest

# My modules
from QUnfold.utils import BinaryEncoder


@given(
    alpha=arrays(
        dtype=np.float64,
        shape=st.integers(min_value=1, max_value=1),
        elements=st.floats(0, 100),
    ),
    beta=arrays(
        dtype=np.float64,
        shape=st.integers(min_value=1, max_value=1),
        elements=st.floats(0, 100),
    ),
    encoding=arrays(
        dtype=np.float64,
        shape=st.integers(min_value=1, max_value=1),
        elements=st.floats(4, 12),
    ),
)
def test_QUnfold_constructor(alpha, beta, encoding):
    """
    Test the QUnfold constructor when encoding bits is set to 8.

    Parameters:
        alpha (np.ndarray): The alpha parameter array.
        beta (np.ndarray): The beta parameter array.
        encoding (np.ndarray): The encoding array.

    Raises:
        AssertionError: If any of the assertions fail.
    """

    # Check the constructor
    encoder_par = BinaryEncoder(alpha, beta, encoding)
    assert encoder_par.alpha == alpha
    assert encoder_par.beta == beta
    assert encoder_par.encoding_bits == encoding
