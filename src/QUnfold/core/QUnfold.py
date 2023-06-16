#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ---------------------- Metadata ----------------------
#
# File name:  QUnfold.py
# Author:     Gianluca Bianco (biancogianluca9@gmail.com)
# Date:       2023-06-16
# Copyright:  (c) 2023 Gianluca Bianco under the MIT license.


class QUnfold:
    """
    Base class of QUnfold algorithms. It stores all the common properties of each QUnfold derived class. Do not use this for unfolding, instead use the other derived classes.
    """

    def __init__(self, *args):
        """
        Todo
        """

        if len(args) == 0:
            pass
        elif len(args) == 2:
            pass
        elif len(args) == 3:
            pass
