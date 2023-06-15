#!/bin/bash

# ---------------------- Metadata ----------------------
#
# File name:  classical_unfolding.sh
# Author:     Gianluca Bianco (biancogianluca9@gmail.com)
# Date:       2023-06-13
# Copyright:  (c) 2023 Gianluca Bianco under the MIT license.

# Variables
input="../data/distributions/falling.csv"
response="../data/responses/nominal.txt"
lreg="0.5"

# Run the unfolding
./RooUnfold/unfolding.py --input=${input} --lreg=${lreg} --response=${response}