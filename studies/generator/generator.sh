#!/bin/bash

# ---------------------- Metadata ----------------------
#
# File name:  generator.sh
# Author:     Gianluca Bianco (biancogianluca9@gmail.com)
# Date:       2023-06-15
# Copyright:  (c) 2023 Gianluca Bianco under the MIT license.

# Variables
distr="double-peaked"
samples=100000

# Run script
./generator/generator.py --distr=${distr} --samples=${samples}
