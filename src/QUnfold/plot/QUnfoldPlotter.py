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
import seaborn as sns


class QUnfoldPlotter:
    """
    Class used to plot QUnfold data and results.
    """

    def __init__(self, unfolder, truth, binning):
        """
        Construct a QUnfoldPlotter class from the unfolder type of QUnfold.

        Args:
            unfolder (QUnfoldQUBO): the used unfolder.
            truth (np.array): the true distribution.
            binning (np.array): the binning of the histograms.
        """

        self.response = unfolder.response[1:, :-1]
        self.measured = unfolder.measured[1:-1]
        self.unfolded = unfolder.unfolded[1:-1]
        self.truth = truth[1:-1]
        self.binning = binning

    def __plotResponseSetup(self):
        """
        Set up the response matrix plot for drawing or saving: the response matrix is set up as a heatmap, with the column representing the measured variable and the row representing the truth variable.
        """

        # Set up plot
        plt.imshow(
            np.transpose(self.response),
            cmap="viridis",
            extent=[
                self.binning[0],
                self.binning[-1],
                self.binning[0],
                self.binning[-1],
            ],
            origin="lower",
        )

        # Plot settings
        plt.colorbar(label="Response Value")
        plt.xlabel("Truth")
        plt.ylabel("Measured")
        plt.title("Response Matrix")

    def plotResponse(self):
        """
        Plot the response matrix with matplotlib style.
        """

        self.__plotResponseSetup()
        plt.show()
        plt.close()

    def saveResponse(self, path: str):
        """
        Save the plotted response matrix with matplotlib style to a file.

        Args:
            path (str): The file path to save the plot.
        """

        self.__plotResponseSetup()
        plt.savefig(path)
        plt.close()

    def __plotSetup(self, method):
        """
        Create an histogram comparison among measured, truth and unfolded distributions, with the chi2 among unfolded and truth distribution.

        Args:
            method (str): Unfolding method type.
        """

        # Initial settings
        sns.set()

        # Measured histogram
        plt.step(
            x=np.concatenate(
                (
                    [self.binning[0] - (self.binning[1] - self.binning[0])],
                    self.binning[:-1],
                )
            ),
            y=np.concatenate(([self.measured[0]], self.measured)),
            label="Measured",
            color="blue",
        )

        # Truth histogram
        plt.step(
            x=np.concatenate(
                (
                    [self.binning[0] - (self.binning[1] - self.binning[0])],
                    self.binning[:-1],
                )
            ),
            y=np.concatenate(([self.truth[0]], self.truth)),
            label="Truth",
            color="red",
        )

        # Unfolded histogram
        plt.step(
            x=np.concatenate(
                (
                    [self.binning[0] - (self.binning[1] - self.binning[0])],
                    self.binning[:-1],
                )
            ),
            y=np.concatenate(([self.unfolded[0]], self.unfolded)),
            label="Unfolded",
            color="green",
        )

        # Plot settings
        plt.title(method)
        plt.xlabel("Bins")
        plt.ylabel("Events")
        plt.tight_layout()
        plt.legend()

    def plot(self, method=""):
        """
        Plot the measured distribution histogram.

        Args:
            method (str): Unfolding method type. Default "".
        """

        self.__plotSetup(method)
        plt.show()
        plt.close()

    def savePlot(self, path: str, method=""):
        """
        Save the plot of the measured distribution histogram into path.

        Args:
            path (str): The file path to save the plot.
            method (str): Unfolding method type. Default "".
        """

        self.__plotSetup(method)
        plt.savefig(path)
        plt.close()
