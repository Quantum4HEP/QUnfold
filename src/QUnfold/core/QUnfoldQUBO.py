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
    Represents the QUnfold class which uses QUBO approach to solve the unfolding problem.

    Args:
        QUnfold: the mother class holding basic QUnfold properties.
    """

    def __init__(self, *args):
        """
        Initialize the QUBOUnfold object.

        Args:
            *args: Variable-length arguments. These arguments are passed to the parent class QUnfold's constructor.
        """

        # Call the QUnfold constructor
        super().__init__(*args)

        # Transform the inputs into binary
        # ...

    # @encoding_bits.setter
    # def encoding_bits(self, value):

    #     if isinstance(value, int):
    #         pass
    #     else:
    #         self.encoding_bits = value

    def unfold(self):
        """
        Method used to perform the unfolding using QUBO approach.
        """

        # Todo...
        pass
