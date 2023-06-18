#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ---------------------- Metadata ----------------------
#
# File name:  QUnfold.py
# Author:     Gianluca Bianco (biancogianluca9@gmail.com)
# Date:       2023-06-16
# Copyright:  (c) 2023 Gianluca Bianco under the MIT license.

# Data science modules
import numpy as np

# Utils modules
from ..utils.linear_algebra import is_matrix
from ..utils.statistics import is_histogram


class QUnfold:
    """
    Base class of QUnfold algorithms. It stores all the common properties of each QUnfold derived class. Do not use this for unfolding, instead use the other derived classes.
    """

    def __init__(self, *args):
        """
        Todo
        """

        # QUnfold()
        if len(args) == 0:
            self.response = None
            self.measured = None

        # QUnfold(response, measured)
        elif len(args) == 2:

            # Initialize variables
            self.response = args[0]
            self.measured = args[1]

            # Check that inputs are corrects
            assert is_matrix(
                self.response
            ), "The first element of the 2-dim constructor must be a NumPy matrix!"
            assert is_histogram(
                self.measured
            ), "THe second element of the 2-dim constructor must be a NumPy histogram"

        # QUnfold(response,measured_bin_content,measured_bin_errors)
        elif len(args) == 3:

            # Check that inputs are correct
            # ...

            # Initialize variables
            self.measured = args[0]

            # Create histogram
            # ...
