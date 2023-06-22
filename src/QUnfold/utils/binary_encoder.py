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
from decimal import Decimal


class BinaryEncoder:
    """
    This class represents a binary encoder for arrays. Encoding is based on a non-standard n-bit integer encoding: since current quantum computing hardware requires the bit encodings to be small each array value x_i is encoded with n bits using an offset alpha_i, and a scaling parameter, beta_i.
    """

    def __init__(self, alpha, beta, encoding_bits):
        """
        Initializes a BinaryEncoder object.

        Args:
            alpha (np.ndarray): The alpha parameter array.
            beta (np.ndarray): The beta parameter array (it is a matrix).
            encoding_bits (np.ndarray): The number of encoding bits.
        """

        self.alpha = alpha
        self._beta = beta
        self.encoding_bits = encoding_bits
        
    @property
    def beta(self):
        return self._beta

    @beta.setter
    def __compute_beta_ij(self, value: np.ndarray):

        self._beta = np.array([])
        lenght = len(value)
        for i in range(lenght):
            self._beta = np.concatenate(self._beta, np.array(value[i]))

    def encode(self, array: np.ndarray):
        """
        Encodes the given array using the specified encoding parameters.

        Args:
            array (np.ndarray): The array to be encoded.

        Returns:
            np.ndarray: The encoded binary representation of the array.

        """

        # Variables
        n_bits_total = int(sum(self.encoding_bits))
        array_bin = np.zeros(n_bits_total, dtype="uint")
        array_length = array.shape[0]

        # Convert each element of the array
        for i in range(array_length - 1, -1, -1):
            n = int(self.encoding_bits[i])
            array_dec = array[i] - self.alpha[i]

            # Develope the sum from 0 to n-1
            for j in range(0, n - 1):
                a = int(np.sum(self.encoding_bits[:i]) + j)
                more_than = Decimal(array_dec) // Decimal(self.beta[i][a])
                equal_to = np.isclose(array_dec, self.beta[i][a])
                array_bin[a] = min([1, more_than or equal_to])
                array_dec = array_dec - array_bin[a] * self.beta[i][a]

        return array_bin

    # Setter ?
