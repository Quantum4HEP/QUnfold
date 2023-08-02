#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ---------------------- Metadata ----------------------
#
# File name:  helpers.py
# Author:     Gianluca Bianco (biancogianluca9@gmail.com)
# Date:       2023-06-14
# Copyright:  (c) 2023 Gianluca Bianco under the MIT license.

# STD modules
import sys

# Data science modules
import ROOT as r

# Utils modules
from functions.custom_logger import ERROR


def load_RooUnfold():
    """
    Load the RooUnfold library installed with the scripts/fetchRooUnfold.sh script.
    """

    loaded_RooUnfold = r.gSystem.Load("../HEP_deps/RooUnfold/libRooUnfold.so")
    if not loaded_RooUnfold == 0:
        ERROR("RooUnfold not found!")
        sys.exit(0)
