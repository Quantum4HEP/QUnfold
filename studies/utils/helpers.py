#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Thr Jun 15 11:08:00 2023
Author: Gianluca Bianco
"""

# STD modules
import sys

# Data science modules
import ROOT as r

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
