#!/bin/bash

# ---------------------- Metadata ----------------------
#
# File name:  generator.sh
# Author:     Gianluca Bianco (biancogianluca9@gmail.com)
# Date:       2023-06-15
# Copyright:  (c) 2023 Gianluca Bianco under the MIT license.

# Variables
distr="breit-wigner"
samples=100000

# Aggiungere parametro numero di samples

# Run script
./generator/generator.py --distr=${distr} --samples=${samples}