#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ---------------------- Metadata ----------------------
#
# File name:  QUnfoldQUBO.py
# Author:     Gianluca Bianco (biancogianluca9@gmail.com)
# Date:       2023-06-16
# Copyright:  (c) 2023 Gianluca Bianco under the MIT license.

# QUnfold modules
from .QUnfold import QUnfold


class QUnfoldQUBO(QUnfold):
    """
    Represents the QUnfold class which uses QUBO algorithms.

    Args:
        QUnfold: the mother class holding basic QUnfold properties.
    """

    def __init__(self, *args):
        """
        Todo
        """

        # Call the QUnfold constructor
        super().__init__(*args)
