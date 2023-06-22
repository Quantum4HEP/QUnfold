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


def main():
    # Load normal distribution data
    truth_bin_content = np.loadtxt("data/normal/truth_bin_content.txt")
    truth_bin_err = np.loadtxt("data/normal/truth_bin_err.txt")
    meas_bin_content = np.loadtxt("data/normal/meas_bin_content.txt")
    meas_bin_err = np.loadtxt("data/normal/meas_bin_err.txt")
    response = np.loadtxt("data/normal/response.txt")

    # Declare the unfolder (bin errors are not necessary since they are computed from bins content)
    unfolder = QUnfoldQUBO(response, meas_bin_content, np.linspace(-10, 10, 41))

    # Print data
    unfolder.printResponse()
    unfolder.printMeasured()

    # Plot response and measured histo
    unfolder.saveResponsePlot("img/examples/standard/response.png")
    unfolder.saveMeasuredPlot("img/examples/standard/measured.png")

    # Perform the unfolding
    unfolder.unfold()

    # Plot unfolding with a method of the class
    # ...


if __name__ == "__main__":
    main()
