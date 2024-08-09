#!/bin/bash

# Install ROOT by conda-forge channel
# https://root.cern/install/#conda
conda config --set channel_priority strict
conda install -c conda-forge root -y

# Install RooUnfold from source using pip
pip install git+https://gitlab.cern.ch/RooUnfold/RooUnfold

# Clear conda environment
conda clean --all -y
pip cache purge
