#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ---------------------- Metadata ----------------------
#
# File name:  tools.py
# Author:     Gianluca Bianco (biancogianluca9@gmail.com)
# Date:       2023-11-06
# Copyright:  (c) 2023 Gianluca Bianco under the MIT license.


def normalize_response(response, truth_mc):
    """
    Function used to normalize the response matrix using the Monte Carlo generated truth distribution.

    Args:
        response (numpy.ndarray): the response matrix to be normalized.
        truth_mc (numpy.ndarray): the Monte Carlo truth histogram used to normalize the response matrix.

    Returns:
        numpy.ndarray: the normalized response matrix.
    """
    return response / (truth_mc + 1e-6)
