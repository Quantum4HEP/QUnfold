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
)
def test_QUnfold_constructor(alpha, beta):
    """
    Test the QUnfold constructor when encoding bits is set to 8.

    Parameters:
        alpha (np.ndarray): The alpha parameter array.
        beta (np.ndarray): The beta parameter array.

    Raises:
        AssertionError: If any of the assertions fail.
    """

    # Check the 1-dim constructor
    encoder = BinaryEncoder()
    assert encoder.alpha == None
    assert encoder.beta == None
    assert encoder.encoding_bits == None

    # Check the 2-dim constructor
    encoder_par = BinaryEncoder(alpha, beta, 8)
    assert encoder_par.alpha == alpha
    assert encoder_par.beta == beta
    assert encoder_par.encoding_bits == 8

    # Check >2-dim constructor
    with pytest.raises(AssertionError):
        BinaryEncoder(alpha, beta, alpha)
        BinaryEncoder(alpha, beta)
