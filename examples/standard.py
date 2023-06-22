#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ---------------------- Metadata ----------------------
#
# File name:  standard.py
# Author:     Gianluca Bianco (biancogianluca9@gmail.com)
# Date:       2023-06-16
# Copyright:  (c) 2023 Gianluca Bianco under the MIT license.

# Data science modules
import numpy as np

# QUnfold modules
from QUnfold.core import QUnfoldQUBO
from QUnfold.plot import QUnfoldPlotter


def main():

    # Load normal distribution data
    truth_bin_content = np.loadtxt("data/normal/truth_bin_content.txt")
    meas_bin_content = np.loadtxt("data/normal/meas_bin_content.txt")
    response = np.loadtxt("data/normal/response.txt")

    # Declare the unfolder (bin errors are not necessary since they are computed from bins content)
    unfolder = QUnfoldQUBO(response, meas_bin_content, np.linspace(-10, 10, 41))

    # Plot response and measured histo
    plotter = QUnfoldPlotter(unfolder)
    plotter.saveResponsePlot("img/examples/standard/response.png")
    plotter.saveMeasuredPlot("img/examples/standard/measured.png")

    # Perform the unfolding
    unfolder.unfold()

    # Plot unfolding with a method of the class
    # ...


if __name__ == "__main__":
    main()
