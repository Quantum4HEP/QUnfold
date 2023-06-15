#!/bin/bash

# ---------------------- Metadata ----------------------
#
# File name:  fetchRooUnfold.sh
# Author:     Gianluca Bianco (biancogianluca9@gmail.com)
# Date:       2023-06-13
# Copyright:  (c) 2023 Gianluca Bianco under the MIT license.

# Variables
repo="https://gitlab.cern.ch/RooUnfold/RooUnfold.git"
tag="3.0.0"

# Install prerequisites
sudo apt install git build-essential
echo ""

# Enter the HEP deps dir
mkdir -p HEP_deps
cd HEP_deps || exit

# Fetch RooUnfold from the official GitLab repository
echo "Installing RooUnfold v${tag} in ${PWD}":
echo ""
echo "- Fetching the library..."
if [ -d RooUnfold ] ; then
    echo "RooUnfold directory already exists. Pulling latest changes from master..."
    cd RooUnfold || exit
    git fetch --all --tags
    git checkout $tag
else
    git clone --branch $tag $repo
fi

# Compile RooUnfold
echo ""
if [ -d build ] ; then
    echo "- Clean previous build..."
    make clean
    echo ""
    echo "- Compiling the library..."
    make
else
    echo "- Compiling the library..."
    cd RooUnfold || exit
    make
fi
