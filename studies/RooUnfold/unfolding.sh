#!/bin/bash

# ---------------------- Metadata ----------------------
#
# File name:  classical_unfolding.sh
# Author:     Gianluca Bianco (biancogianluca9@gmail.com)
# Date:       2023-06-13
# Copyright:  (c) 2023 Gianluca Bianco under the MIT license.

# Variables
only_one_distr="" # intialize to unfold only one distribution
print_chi2="yes"

# Read all the generated distributions from the data dir
config="../config/distributions.json"
json=$(cat "$config")
distr=$(echo "$json" | jq -r '.distributions[]' | tr '\n' ' ')

# Run the unfolding
if [ -n "${only_one_distr}" ] ; then
    echo "Unfolding only the ${only_one_distr} distribution:"
    echo ""
    ./RooUnfold/unfolding.py --distr="${only_one_distr}" --chi2=${print_chi2}
else
    echo "Unfolding all the distributions:"
    echo ""
    for distr_ in ${distr}
    do
        echo "Unfolding the ${distr_} distribution:"
        echo ""
        ./RooUnfold/unfolding.py --distr="${distr_}" --chi2=${print_chi2}
        echo ""
    done
fi

