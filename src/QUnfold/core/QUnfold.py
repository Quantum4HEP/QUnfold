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
from ..utils.custom_logger import INFO


class QUnfold:
    """
    Base class of QUnfold algorithms. It stores all the common properties of each QUnfold derived class. Do not use this for unfolding, instead use the other derived classes.
    """

    # ==============================================
    #     Constructor
    # ==============================================
    def __init__(self, *args):
        """
        Initialize a QUnfold object.

        Args:
            *args: Variable-length arguments. The supported constructor signatures are:
                - QUnfold(): Initializes an empty QUnfold object.
                - QUnfold(response, measured_bin_contents): Initializes a QUnfold object with the given response matrix and measured histogram bin contents.

        Raises:
            AssertionError: If the constructor signature is not supported or the inputs are incorrect.
        """

        # QUnfold()
        if len(args) == 0:
            self.response = None
            self.measured_bin_contents = None
            self.measured_bin_edges = None

        # QUnfold(response, measured_bin_contents)
        elif len(args) == 3:

            # Check that inputs are correct
            assert is_matrix(
                args[0]
            ), "The first element of the 2-dim constructor must be a NumPy matrix (response)!"
            assert isinstance(
                args[1], np.ndarray
            ), "The second element of the 2-dim constructor must be a NumPy array (bin contents)!"
            assert isinstance(
                args[2], np.ndarray
            ), "The second element of the 2-dim constructor must be a NumPy array (bin edges)!"

            # Initialize variables
            self.response = args[0]
            self.measured_bin_contents = args[1]
            self.measured_bin_edges = args[2]

        # Other conditions are not possible
        else:
            assert False, "This constructor signature is not supported!"

    # ==============================================
    #     Print methods
    # ==============================================
    def printResponse(self):
        """
        Print the response matrix.
        """

        INFO("Response matrix is:\n{}".format(self.response))

    def printMeasured(self):
        """
        Print the measured histogram bin contents and bin edges.
        """

        INFO(
            "Measured histogram bin contents are:\n{}".format(
                self.measured_bin_contents
            )
        )
        INFO("Measured histogram bin edges are:\n{}".format(self.measured_bin_edges))

    # ==============================================
    #     Plot methods
    # ==============================================
    def __plotResponseSetup(self):
        """
        Set up the response matrix plot for drawing or saving: the response matrix is set up as a heatmap, with the column representing the measured variable and the row representing the truth variable.
        """

        # Check if response matrix has been initialized
        assert (
            self.response.any() != None
        ), "Response matrix should be initialized before plotting it!"

        # Set up plot
        plt.imshow(
            np.transpose(self.response),
            cmap="viridis",
            extent=[
                self.measured_bin_edges[0],
                self.measured_bin_edges[-1],
                self.measured_bin_edges[0],
                self.measured_bin_edges[-1],
            ],
            origin="lower",
        )

        # Plot settings
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
        plt.close()

    def saveResponsePlot(self, path: str):
        """
        Save the plotted response matrix with matplotlib style to a file.

        Args:
            path (str): The file path to save the plot.
        """

        self.__plotResponseSetup()
        plt.savefig(path)
        plt.close()

    def __plotMeasuredSetup(self):
        """
        Create a bar chart histogram based on the bin contents and bin edges of the measured distribution.
        """

        # Check that inputs are correct
        assert isinstance(
            self.measured_bin_contents, np.ndarray
        ), "The second element of the 2-dim constructor must be a NumPy array (bin contents)!"
        assert isinstance(
            self.measured_bin_edges, np.ndarray
        ), "The second element of the 2-dim constructor must be a NumPy array (bin edges)!"

        # Set style
        plt.style.use("seaborn-whitegrid")

        # Set up plot
        plt.bar(
            self.measured_bin_edges[:-1],
            self.measured_bin_contents,
            width=np.diff(self.measured_bin_edges),
            align="edge",
            facecolor="#2ab0ff",
            edgecolor="#e0e0e0",
            linewidth=0.5,
        )

        # Plot settings
        plt.xlabel("Bins")
        plt.ylabel("Events")
        plt.title("Measured histogram")

    def plotMeasured(self):
        """
        Plot the measured distribution histogram.
        """

        self.__plotMeasuredSetup()
        plt.show()
        plt.close()

    def saveMeasuredPlot(self, path: str):
        """
        Save the plot of the measured distribution histogram into path.

        Args:
            path (str): The file path to save the plot.
        """

        self.__plotMeasuredSetup()
        plt.savefig(path)
        plt.close()
