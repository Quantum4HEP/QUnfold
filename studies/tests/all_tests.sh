#!/bin/bash

# ---------------------- Metadata ----------------------
#
# File name:  all_tests.sh
# Author:     Gianluca Bianco (biancogianluca9@gmail.com)
# Date:       2023-06-16
# Copyright:  (c) 2023 Gianluca Bianco under the MIT license.


# Run unit tests
find tests -name '*.py' ! -name 'benchmarks.py' -exec pytest {} +
echo ""

# Variables
output="../img/benchmarks"

# Read all the generated distributions from the data dir
config="../config/distributions.json"
json=$(cat "$config")
distributions=$(echo "$json" | jq -r '.distributions[]' | tr '\n' ' ')

# Run benchmarks for each distribution
mkdir -p output/benchmarks
for distr in ${distributions} ; do
    echo "Running benchmarks for ${distr} distribution:"
    echo ""
    pytest \
        --benchmark-json="output/benchmarks/${distr}.json" \
        --benchmark-histogram="../img/benchmarks/${distr}" \
        --distr="${distr}" \
        tests/benchmarks.py
    cairosvg -o "${output}/${distr}".png "${output}/${distr}".svg
    rm -rf .benchmarks "${output}/${distr}".svg
    echo
done