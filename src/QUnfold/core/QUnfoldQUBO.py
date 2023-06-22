#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ---------------------- Metadata ----------------------
#
# File name:  QUnfoldQUBO.py
# Author:     Gianluca Bianco (biancogianluca9@gmail.com)
# Date:       2023-06-16
# Copyright:  (c) 2023 Gianluca Bianco under the MIT license.


class QUnfoldQUBO:
    """
    Represents the QUnfold class which uses QUBO approach to solve the unfolding problem.
    """

    def __init__(self, response, meas_bin_contents, meas_bin_edges):
        """
        Construct a QUnfoldQUBO object with measured histogram and response matrix.

        Args:
            response (np.ndarray): 2-dim NumPy array which represents the response matrix.
            meas_bin_contents (np.ndarray): 1-dim NumPy array which represents the measured histogram bin contents.
            meas_bin_edges (np.ndarray): 1-dim NumPy array which represents the measured histogram bin edges.
        """

        # Initialize variables
        self.response = response
        self.measured_bin_contents = meas_bin_contents
        self.measured_bin_edges = meas_bin_edges

        # Transform the inputs into binary
        # ...

    def unfold(self):
        """
        Method used to perform the unfolding using QUBO approach.
        """

        # Todo...
        pass
