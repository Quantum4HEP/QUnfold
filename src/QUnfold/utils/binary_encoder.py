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

    def __init__(self, alpha, beta, encoding_bits):
        """
        Initializes a BinaryEncoder object.

        Args:
            alpha (np.ndarray): The alpha parameter array.
            beta (np.ndarray): The beta parameter array.
            encoding_bits (np.ndarray): The number of encoding bits.
        """

        self.alpha = alpha
        self.beta = beta
        self.encoding_bits = encoding_bits

    def encode(self, array: np.ndarray):

        # Variables
        pass
        """array_bin = np.zeros(self)
        n_bits_total = int(sum(self.rho))
        array_length = array.shape[0]

        # Encode the vector
        for i in range(array_length - 1, -1, -1):
            n = int(self.encoding_bits[i])
            array_bin = array[i] - self.alpha[i]

            for j in range(0, n - 1):
                a = int(np.sum(self.rho[:i]) + j)"""
