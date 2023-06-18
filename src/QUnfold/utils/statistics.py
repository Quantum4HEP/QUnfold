#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ---------------------- Metadata ----------------------
#
# File name:  statistics.py
# Author:     Gianluca Bianco (biancogianluca9@gmail.com)
# Date:       2023-06-18
# Copyright:  (c) 2023 Gianluca Bianco under the MIT license.

# Data science modules
import numpy as np


def is_histogram(obj):
    """
    Check if an object is a numpy.histogram.

    Args:
        obj: The object to check.

    Returns:
        bool: True if the object is a numpy.histogram, False otherwise.
    """
    if isinstance(obj, tuple) and len(obj) == 2:
        return True
    else:
        return False
