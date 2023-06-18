#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ---------------------- Metadata ----------------------
#
# File name:  QUnfold.py
# Author:     Gianluca Bianco (biancogianluca9@gmail.com)
# Date:       2023-06-16
# Copyright:  (c) 2023 Gianluca Bianco under the MIT license.

# STD modules
import sys, os

# Data science modules
import numpy as np
import matplotlib.pyplot as plt

# Utils modules
from ..utils.linear_algebra import is_matrix
from ..utils.statistics import is_histogram
from ..utils.custom_logger import INFO


class QUnfold:
    """
    Base class of QUnfold algorithms. It stores all the common properties of each QUnfold derived class. Do not use this for unfolding, instead use the other derived classes.
    """

    def __init__(self, *args):
        """
        Initialize a QUnfold object.

        Args:
            *args: Variable-length arguments. The supported constructor signatures are:
                - QUnfold(): Initializes an empty QUnfold object.
                - QUnfold(response, measured): Initializes a QUnfold object with the given response matrix and measured histogram.

        Raises:
            AssertionError: If the constructor signature is not supported or the inputs are incorrect.
        """

        # QUnfold()
        if len(args) == 0:
            self.response = None
            self.measured = None

        # QUnfold(response, measured)
        elif len(args) == 2:

            # Check that inputs are corrects
            assert is_matrix(
                args[0]
            ), "The first element of the 2-dim constructor must be a NumPy matrix (response)!"
            assert is_histogram(
                args[1]
            ), "The second element of the 2-dim constructor must be a NumPy histogram (measured histo)"

            # Initialize variables
            self.response = args[0]
            self.measured = args[1]

        # Other conditions are not possible
        else:
            assert False, "This constructor signature is not supported!"

        # Save bin contents and bin edges
        self.bin_contents = self.measured[0]
        self.bin_edges = self.measured[1]

        # Transform the inputs into binary
        # ...

    def printResponse(self):
        """
        Print the response matrix.
        """

        INFO("Response matrix is:\n{}".format(self.response))

    def printMeasured(self):
        """
        Print the measured histogram.
        """

        INFO("Measured histogram bin contents are:\n{}".format(self.bin_contents))
        INFO("Measured histogram bin edges are:\n{}".format(self.bin_edges))

    def __plotResponseSetup(self):
        """
        Set up the response matrix plot for drawing or saving: the response matrix is set up as a heatmap, with the column representing the measured variable and the row representing the truth variable.
        """

        # Check if response matrix has been initialized
        assert self.response.any() != None, True

        # Set up plot settings
        plt.imshow(
            np.transpose(self.response),
            cmap="viridis",
            extent=[
                self.bin_edges[0],
                self.bin_edges[-1],
                self.bin_edges[0],
                self.bin_edges[-1],
            ],
            origin="lower",
        )
        plt.colorbar(label="Response Value")
        plt.xlabel("Column (measured))")
        plt.ylabel("Row (truth)")
        plt.title("Response Matrix")

    def plotResponse(self):
        """
        Plot the response matrix with matplotlib style.
        """

        self.__plotResponseSetup()
        plt.show()

    def saveResponsePlot(self, path):
        """
        Save the plotted response matrix with matplotlib style to a file.

        Args:
            path (str): The file path to save the plot.
        """

        self.__plotResponseSetup()
        plt.savefig(path)
