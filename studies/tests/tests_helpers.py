#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ---------------------- Metadata ----------------------
#
# File name:  helpers.py
# Author:     Gianluca Banco (biancogianluca9@gmail.com)
# Date:       2023-06-16
# Copyright:  (c) 2023 Gianluca Bianco under the MIT license.

# STD modules
import sys

# Data science modules
import numpy as np

# Testing modules
import numpy.testing as nptest

# Utils modules
sys.path.append("..")
from studies.utils.helpers import load_data


def tests_load_data():
    """
    Tests the load_data function for data loading, by comparing it with the data of the /data directory.
    """

    # Check all the distributions
    for distr in ["normal", "breit-wigner", "double-peaked"]:

        # File names
        truth_bin_content_path = "../data/{}/truth_bin_content.txt".format(distr)
        meas_bin_content_path = "../data/{}/meas_bin_content.txt".format(distr)
        response_path = "../data/{}/response.txt".format(distr)
        binning_path = "../data/{}/binning.txt".format(distr)

        # Load data related to the distribution
        np_truth_bin_content = np.loadtxt(truth_bin_content_path)
        np_meas_bin_content = np.loadtxt(meas_bin_content_path)
        np_response = np.loadtxt(response_path)
        np_binning = np.loadtxt(binning_path)

        # Load data using the helper function
        (
            np_truth_bin_content_,
            np_meas_bin_content_,
            np_response_,
            np_binning_,
        ) = load_data(distr)

        # Check for equality
        nptest.assert_allclose(np_truth_bin_content, np_truth_bin_content_, rtol=1e-5)
        nptest.assert_allclose(np_meas_bin_content, np_meas_bin_content_, rtol=1e-5)
        nptest.assert_allclose(np_response, np_response_, rtol=1e-5)
        nptest.assert_allclose(np_binning, np_binning_, rtol=1e-5)
