#!/bin/bash

# ---------------------- Metadata ----------------------
#
# File name:  comparisons.sh
# Author:     Gianluca Bianco (biancogianluca9@gmail.com)
# Date:       2023-06-26
# Copyright:  (c) 2023 Gianluca Bianco under the MIT license.

# Variables
only_one_distr="" # intialize to compare with only one distribution

# Read all the generated distributions from the data dir
config="../config/distributions.json"
json=$(cat "$config")
distr=$(echo "$json" | jq -r '.distributions[]' | tr '\n' ' ')

# Run the unfolding
if [ -n "${only_one_distr}" ] ; then
    echo "Comparing only the ${only_one_distr} distribution:"
    echo ""
    ./comparisons/comparisons.py --distr="${only_one_distr}"
else
    echo "Comparing all the distributions:"
    echo ""
    for distr_ in ${distr}
    do
        echo "Comparing the ${distr_} distribution:"
        echo ""
        ./comparisons/comparisons.py --distr="${distr_}"
        echo ""
    done
fi