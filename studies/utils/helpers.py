#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ---------------------- Metadata ----------------------
#
# File name:  helpers.py
# Author:     Gianluca Bianco (biancogianluca9@gmail.com)
# Date:       2023-06-14
# Copyright:  (c) 2023 Gianluca Bianco under the MIT license.

# STD modules
import sys, os

# Data science modules
import ROOT as r
import numpy as np

# Utils modules
sys.path.append("../src")
from QUnfold.utils.custom_logger import ERROR


def load_RooUnfold():
    """
    Load the RooUnfold library installed with the scripts/fetchRooUnfold.sh script.
    """

    loaded_RooUnfold = r.gSystem.Load("../HEP_deps/RooUnfold/libRooUnfold.so")
    if not loaded_RooUnfold == 0:
        ERROR("RooUnfold not found!")
        sys.exit(0)


def load_data(distr):
    """
    Load data related to a distribution saved in the /data directory.

    Args:
        distr (str): The name of the distribution.

    Returns:
        tuple: A tuple containing numpy arrays representing the following data:
            - np_truth_bin_content: Array loaded from 'truth_bin_content.txt' file.
            - np_meas_bin_content: Array loaded from 'meas_bin_content.txt' file.
            - np_response: Array loaded from 'response.txt' file.
    """

    # File names
    truth_bin_content_path = "../data/{}/truth_bin_content.txt".format(distr)
    meas_bin_content_path = "../data/{}/meas_bin_content.txt".format(distr)
    response_path = "../data/{}/response.txt".format(distr)

    # Load data related to the distribution
    np_truth_bin_content = np.loadtxt(truth_bin_content_path)
    np_meas_bin_content = np.loadtxt(meas_bin_content_path)
    np_response = np.loadtxt(response_path)

    return (
        np_truth_bin_content,
        np_meas_bin_content,
        np_response,
    )
