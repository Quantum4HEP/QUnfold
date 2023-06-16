#!/bin/bash

# ---------------------- Metadata ----------------------
#
# File name:  create_distr_config.sh
# Author:     Gianluca Bianco (biancogianluca9@gmail.com)
# Date:       2023-06-16
# Copyright:  (c) 2023 Gianluca Bianco under the MIT license.

# Variables
directory_names=()
data_dir="data"

# Iterate over data directory and add file names to the array
echo "Found the following distributions:"
for dir_path in "$data_dir"/*; do
    if [ -d "$dir_path" ]; then
        echo "- ${dir_path}"
        dir_name=$(basename "$dir_path")
        directory_names+=("\"$dir_name\"")
    fi
done
echo ""

# Create a JSON config file with the name of the distributions
json_file=config/distributions.json
printf "{%s}" "$(IFS=,; echo "\"distributions\": [${directory_names[*]}]")" > "$json_file"

echo "Distribution config file created: $json_file"