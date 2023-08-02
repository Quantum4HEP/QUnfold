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

# My modules
sys.path.append("..")
sys.path.append("../src")
from studies.functions.ROOT_converter import TH1_to_array, TH2_to_array
from studies.functions.generator import generate
from studies.functions.custom_logger import ERROR
from src.QUnfold import QUnfoldQUBO

# RooUnfold settings
loaded_RooUnfold = r.gSystem.Load("../HEP_deps/RooUnfold/libRooUnfold.so")
if not loaded_RooUnfold == 0:
    ERROR("RooUnfold not found!")
    sys.exit(0)


def load_input(request, software):
    """
    Load and prepare input data for the benchmarking.

    Args:
        request: The pytest request object.
        software: Specific the unfolding library used.

    Returns:
        Tuple: A tuple containing the RooUnfold response matrix (m_response) and the measurement histogram (h_meas).
    """

    # Variables
    distr = request.config.getoption("--distr")
    bins = 40
    min_bin = -10
    max_bin = 10
    if distr == "exponential":
        min_bin = 0
    truth, meas, response = generate(distr, bins, min_bin, max_bin, 10000)

    # Extra settings for QUnfold
    if software == "QUnfold":
        truth = TH1_to_array(truth, overflow=True)
        meas = TH1_to_array(meas, overflow=True)
        response = TH2_to_array(response.Hresponse(), overflow=True)

    return response, meas


def RooUnfoldSvd(m_response, h_meas):
    """
    Applies the SVD (Singular Value Decomposition) unfolding method to the provided measurement histogram.

    Args:
        m_response (ROOT.RooUnfoldResponse): The RooUnfold response matrix.
        h_meas (ROOT.TH1): The measurement histogram.
    """

    m_response.UseOverflow(False)
    r.RooUnfoldSvd(m_response, h_meas, 3)


def test_RooUnfoldSvd(request, benchmark):
    """
    Perform benchmarking of the RooUnfoldSvd function using the provided input data.

    Args:
        request: The pytest request object.
        benchmark: The benchmark fixture.
    """

    m_response, h_meas = load_input(request, "RooUnfold")
    result = benchmark(RooUnfoldSvd, m_response, h_meas)


def RooUnfoldBayes(m_response, h_meas):
    """
    Applies the Bayesian unfolding method to the provided measurement histogram.

    Args:
        m_response (ROOT.RooUnfoldResponse): The RooUnfold response matrix.
        h_meas (ROOT.TH1): The measurement histogram.
    """

    m_response.UseOverflow(False)
    r.RooUnfoldBayes(m_response, h_meas, 4, 0)


def test_RooUnfoldBayes(request, benchmark):
    """
    Perform benchmarking of the RooUnfoldBayes function using the provided input data.

    Args:
        request: The pytest request object.
        benchmark: The benchmark fixture.
    """

    m_response, h_meas = load_input(request, "RooUnfold")
    result = benchmark(RooUnfoldBayes, m_response, h_meas)


def RooUnfoldBinByBin(m_response, h_meas):
    """
    Applies the bin-by-bin unfolding method to the provided measurement histogram.

    Args:
        m_response (ROOT.RooUnfoldResponse): The RooUnfold response matrix.
        h_meas (ROOT.TH1): The measurement histogram.
    """

    m_response.UseOverflow(False)
    r.RooUnfoldBinByBin(m_response, h_meas)


def test_RooUnfoldBinByBin(request, benchmark):
    """
    Perform benchmarking of the RooUnfoldBinByBin function using the provided input data.

    Args:
        request: The pytest request object.
        benchmark: The benchmark fixture.
    """

    m_response, h_meas = load_input(request, "RooUnfold")
    result = benchmark(RooUnfoldBinByBin, m_response, h_meas)


def QUnfoldSimulated(m_response, h_meas):
    """
    Applies the simulated annealing unfolding method to the provided measurement histogram.

    Args:
        m_response (ROOT.RooUnfoldResponse): The RooUnfold response matrix.
        h_meas (ROOT.TH1): The measurement histogram.
    """

    unfolder = QUnfoldQUBO(m_response, h_meas)
    result = unfolder.solve_simulated_annealing(lam=0.1, num_reads=100)


def test_QUnfoldSimulated(request, benchmark):
    """
    Perform benchmarking of the QUnfoldSimulated function using the provided input data.

    Args:
        request: The pytest request object.
        benchmark: The benchmark fixture.
    """

    m_response, h_meas = load_input(request, "QUnfold")
    result = benchmark(QUnfoldSimulated, m_response, h_meas)


# Pytest main
pytest.main()
