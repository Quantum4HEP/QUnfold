#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ---------------------- Metadata ----------------------
#
# File name:  benchmarks.py
# Author:     Gianluca Bianco (biancogianluca9@gmail.com)
# Date:       2023-06-16
# Copyright:  (c) 2023 Gianluca Bianco under the MIT license.

# STD modules
import sys

# Data science modules
import ROOT as r

# Testing modules
import pytest

# Utils modules
sys.path.append("..")
from studies.functions.helpers import load_RooUnfold, load_data
from studies.functions.ROOT_converter import array_to_TH1, array_to_TH2

# ROOT settings
load_RooUnfold()


def load_input(request):
    """
    Load and prepare input data for the benchmarking.

    Args:
        request: The pytest request object.

    Returns:
        Tuple: A tuple containing the RooUnfold response matrix (m_response) and the measurement histogram (h_meas).
    """

    # Read input data
    distr = request.config.getoption("--distr")
    (
        np_truth_bin_content,
        np_meas_bin_content,
        np_response,
        np_binning,
    ) = load_data(distr)
    bins = int(np_binning[0])
    min_bin = int(np_binning[1])
    max_bin = int(np_binning[2])

    # Convert to ROOT variables
    h_truth = array_to_TH1(np_truth_bin_content, bins, min_bin, max_bin, "truth")
    h_meas = array_to_TH1(np_meas_bin_content, bins, min_bin, max_bin, "meas")
    h_response = array_to_TH2(
        np_response, bins, min_bin, max_bin, bins, min_bin, max_bin, "response"
    )

    # Initialize the RooUnfold response matrix from the input data
    m_response = r.RooUnfoldResponse(h_meas, h_truth, h_response)
    m_response.UseOverflow(False)

    return m_response, h_meas


def RooUnfoldInvert(m_response, h_meas):
    """
    Inverts the RooUnfold response matrix using the provided measurement histogram.

    Args:
        m_response (ROOT.RooUnfoldResponse): The RooUnfold response matrix.
        h_meas (ROOT.TH1): The measurement histogram.
    """

    r.RooUnfoldInvert(m_response, h_meas)


def test_RooUnfoldInvert(request, benchmark):
    """
    Perform benchmarking of the RooUnfoldInvert function using the provided input data.

    Args:
        request: The pytest request object.
        benchmark: The benchmark fixture.
    """

    # Read input data
    m_response, h_meas = load_input(request)

    # Perform benchmarking
    result = benchmark(RooUnfoldInvert, m_response, h_meas)


def RooUnfoldSvd(m_response, h_meas):
    """
    Applies the SVD (Singular Value Decomposition) unfolding method to the provided measurement histogram.

    Args:
        m_response (ROOT.RooUnfoldResponse): The RooUnfold response matrix.
        h_meas (ROOT.TH1): The measurement histogram.
    """

    r.RooUnfoldSvd(m_response, h_meas, 3)


def test_RooUnfoldSvd(request, benchmark):
    """
    Perform benchmarking of the RooUnfoldSvd function using the provided input data.

    Args:
        request: The pytest request object.
        benchmark: The benchmark fixture.
    """

    # Read input data
    m_response, h_meas = load_input(request)

    # Perform benchmarking
    result = benchmark(RooUnfoldSvd, m_response, h_meas)


def RooUnfoldBayes(m_response, h_meas):
    """
    Applies the Bayesian unfolding method to the provided measurement histogram.

    Args:
        m_response (ROOT.RooUnfoldResponse): The RooUnfold response matrix.
        h_meas (ROOT.TH1): The measurement histogram.
    """

    r.RooUnfoldBayes(m_response, h_meas, 4, 0)


def test_RooUnfoldBayes(request, benchmark):
    """
    Perform benchmarking of the RooUnfoldBayes function using the provided input data.

    Args:
        request: The pytest request object.
        benchmark: The benchmark fixture.
    """

    # Read input data
    m_response, h_meas = load_input(request)

    # Perform benchmarking
    result = benchmark(RooUnfoldBayes, m_response, h_meas)


def RooUnfoldBinByBin(m_response, h_meas):
    """
    Applies the bin-by-bin unfolding method to the provided measurement histogram.

    Args:
        m_response (ROOT.RooUnfoldResponse): The RooUnfold response matrix.
        h_meas (ROOT.TH1): The measurement histogram.
    """

    r.RooUnfoldBinByBin(m_response, h_meas)


def test_RooUnfoldBinByBin(request, benchmark):
    """
    Perform benchmarking of the RooUnfoldBinByBin function using the provided input data.

    Args:
        request: The pytest request object.
        benchmark: The benchmark fixture.
    """

    # Read input data
    m_response, h_meas = load_input(request)

    # Perform benchmarking
    result = benchmark(RooUnfoldBinByBin, m_response, h_meas)


# Pytest main
pytest.main()
