#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ---------------------- Metadata ----------------------
#
# File name:  linear_algebra.py
# Author:     Gianluca Bianco (biancogianluca9@gmail.com)
# Date:       2023-06-17
# Copyright:  (c) 2023 Gianluca Bianco under the MIT license.

# Data science modules
import numpy as np


def is_matrix(entity):
    """
    Check if the input object is a numpy matrix.

    Args:
        entity (generic): an input variable which type should be checked.

    Returns:
        bool: True if the input is a NumPy matrix, False otherwise.
    """

    if isinstance(entity, np.ndarray) and entity.ndim == 2:
        return True
    else:
        return False
