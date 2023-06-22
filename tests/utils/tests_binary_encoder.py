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
        dtype=np.int64,
        shape=st.integers(min_value=1, max_value=100),
        elements=st.integers(0, 100),
    ),
    beta=arrays(
        dtype=np.int64,
        shape=st.integers(min_value=1, max_value=100),
        elements=st.integers(0, 100),
    ),
    encoding=arrays(
        dtype=np.int64,
        shape=st.integers(min_value=1, max_value=100),
        elements=st.integers(4, 8),
    ),
)
def test_BinaryEncoder(alpha, beta, encoding):
    """
    Test the BinaryEncoder class.

    Parameters:
        alpha (np.ndarray): The alpha parameter array.
        beta (np.ndarray): The beta parameter array.
        encoding (np.ndarray): The encoding array.

    Raises:
        AssertionError: If any of the assertions fail.
    """

    # Check the constructor
    encoder_par = BinaryEncoder(alpha, beta, encoding)

    # Test the constructor
    def test_constructor():
        np.array_equal(encoder_par.alpha, alpha)
        np.array_equal(encoder_par.beta, beta)
        np.array_equal(encoder_par.encoding_bits, encoding)

    test_constructor()

    # @given(
    #     array=arrays(
    #         dtype=np.float64,
    #         shape=st.integers(min_value=1, max_value=1),
    #         elements=st.floats(0, 100),
    #     )
    # )
    # def test_encode(array):

    #     assert encoder_par.encode(array) == 0

    # test_encode()
