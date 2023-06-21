#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ---------------------- Metadata ----------------------
#
# File name:  binary_encoder.py
# Author:     Gianluca Bianco (biancogianluca9@gmail.com)
# Date:       2023-06-21
# Copyright:  (c) 2023 Gianluca Bianco under the MIT license.

# Data science modules
import numpy as np


class BinaryEncoder:
    """
    This class represents a binary encoder for arrays. Encoding is based on a non-standard n-bit integer encoding: since current quantum computing hardware requires the bit encodings to be small each array value x_i is encoded with n bits using an offset alpha_i, and a scaling parameter, beta_i.
    """

    def __init__(self, *args):
        """
        Initializes a BinaryEncoder object.

        Args
            *args: Variable-length arguments. The supported constructor signatures are:
                - BinaryEncoder(): Initializes a BinaryEncoder object with alpha, beta, and encoding_bits set to None.
                - BinaryEncoder(alpha: np.ndarray, beta: np.ndarray, encoding_bits: int): Initializes a BinaryEncoder object with the provided alpha, beta, and encoding_bits.

        Raises:
            AssertionError: If the inputs are incorrect.
        """

        # BinaryEncoder()
        if len(args) == 0:
            self.alpha = None
            self.beta = None
            self.encoding_bits = None

        # BinaryEncoder(alpha, beta, rho)
        elif len(args) == 3:

            # Check that inputs are correct
            assert isinstance(
                args[0], np.ndarray
            ), "The first element of the 3-dim constructor must be a NumPy 1-dim array (alpha)"
            assert isinstance(
                args[1], np.ndarray
            ), "The second element of the 3-dim constructor must be a NumPy 1-dim array (beta)"
            assert isinstance(
                args[2], int
            ), "The third element of the 3-dim constructor must be an integer (encoding bits)"

            # Initialize variables
            self.alpha = args[0]
            self.beta = args[1]
            self.encoding_bits = args[2]

        # Other conditions are not possible
        else:
            assert False, "This constructor signature is not supported!"
