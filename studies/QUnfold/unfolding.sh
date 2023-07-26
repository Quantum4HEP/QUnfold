#!/bin/bash

# ---------------------- Metadata ----------------------
#
# File name:  unfolding.sh
# Author:     Gianluca Bianco (biancogianluca9@gmail.com)
# Date:       2023-07-26
# Copyright:  (c) 2023 Gianluca Bianco under the MIT license.

# Variables
only_one_distr="double-peaked" # intialize to unfold only one distribution

# Read all the generated distributions from the data dir
config="../config/distributions.json"
json=$(cat "$config")
distr=$(echo "$json" | jq -r '.distributions[]' | tr '\n' ' ')

# Run the unfolding
if [ -n "${only_one_distr}" ] ; then
    echo "Unfolding only the ${only_one_distr} distribution:"
    echo ""
    ./QUnfold/unfolding.py --distr="${only_one_distr}"
else
    echo "Unfolding all the distributions:"
    echo ""
    for distr_ in ${distr}
    do
        echo "Unfolding the ${distr_} distribution:"
        echo ""
        ./QUnfold/unfolding.py --distr="${distr_}"
        echo ""
    done
fi

