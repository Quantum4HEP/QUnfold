#!/bin/bash

# ---------------------- Metadata ----------------------
#
# File name:  generator.sh
# Author:     Gianluca Bianco (biancogianluca9@gmail.com)
# Date:       2023-06-15
# Copyright:  (c) 2023 Gianluca Bianco under the MIT license.

# Variables
distr="breit-wigner normal double-peaked"
samples=10000
max_bin=10
min_bin=-10
bins=40
only_one_distr="" # intialize to generate only one distribution
remove_empty_bins="no"

# Run script
if [ -n "${only_one_distr}" ] ; then
    echo "Generating data for the ${only_one_distr} distribution:"
    echo ""
    ./generator/generator.py \
        --distr="${only_one_distr}" \
        --samples=${samples} \
        --max_bin=${max_bin} \
        --min_bin=${min_bin} \
        --bins=${bins} \
        --remove_empty_bins=${remove_empty_bins}
else
    echo "Generating all the distributions data:"
    echo ""
    for distr_ in ${distr}
    do
        echo "Generating data for the ${distr_} distribution:"
        echo ""
        ./generator/generator.py \
            --distr="${distr_}" \
            --samples=${samples} \
            --max_bin=${max_bin} \
            --min_bin=${min_bin} \
            --bins=${bins} \
            --remove_empty_bins=${remove_empty_bins}
        echo ""
    done
fi
