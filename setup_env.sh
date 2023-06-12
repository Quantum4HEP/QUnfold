#!/bin/bash

#====================================================
#     Metadata
#====================================================
# File name:  setup_env.sh
# Author:     Gianluca Bianco (biancogianluca9@gmail.com)
# Date:       2023-06-12
# Copyright:  (c) 2022 Gianluca Bianco under the MIT license.

# Variables
version="3.10.6"
venv="qunfold"
reqs="requirements.txt"

# Creating the virtualenv
if [[ ! -d ${venv} ]] ; then
    if [ "${venv}" != "" ] ; then
        if ! virtualenv "${venv}" -p python"${version}" ; then
            echo "Error: missing Python ${version} installation."
            exit 1
        fi
    fi
    if  [ "${reqs}" != "" ] ; then
        echo "Installing prerequisites..."
        pip install -r "${reqs}"
    fi
fi

# Activating virtual environment
echo "Activating the virtual environment..."
source ${venv}/bin/activate
echo "Done."