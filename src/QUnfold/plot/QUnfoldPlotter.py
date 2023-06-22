#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ---------------------- Metadata ----------------------
#
# File name:  QUnfoldPlotter.py
# Author:     Gianluca Bianco (biancogianluca9@gmail.com)
# Date:       2023-06-22
# Copyright:  (c) 2023 Gianluca Bianco under the MIT license.

# Data science modules
import matplotlib.pyplot as plt
import numpy as np


class QUnfoldPlotter:
    """
    Class used to plot QUnfold data and results.
    """

    def __init__(self, unfolder):
        """
        Construct a QUnfoldPlotter class from the unfolder type of QUnfold.

        Args:
            unfolder (QUnfoldQUBO): the used unfolder.
        """

        self.response = unfolder.response
        self.meas_bin_contents = unfolder.measured_bin_contents
        self.meas_bin_edges = unfolder.measured_bin_edges

    def __plotResponseSetup(self):
        """
        Set up the response matrix plot for drawing or saving: the response matrix is set up as a heatmap, with the column representing the measured variable and the row representing the truth variable.
        """

        # Set up plot
        plt.imshow(
            np.transpose(self.response),
            cmap="viridis",
            extent=[
                self.meas_bin_edges[0],
                self.meas_bin_edges[-1],
                self.meas_bin_edges[0],
                self.meas_bin_edges[-1],
            ],
            origin="lower",
        )

        # Plot settings
        plt.colorbar(label="Response Value")
        plt.xlabel("Column (measured)")
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

        # Set style
        plt.style.use("seaborn-whitegrid")

        # Set up plot
        plt.bar(
            self.meas_bin_edges[:-1],
            self.meas_bin_contents,
            width=np.diff(self.meas_bin_edges),
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
