#!/bin/bash

# Update the system first
echo "1) Updating the system:"
sudo apt update && sudo apt upgrade
echo ""

# Installing prerequisites
echo "2) Installing prerequisites:"
sudo apt install git dpkg-dev cmake g++ gcc binutils libx11-dev libxpm-dev libxft-dev libxext-dev sudo gfortran libssl-dev libpcre3-dev libglu1-mesa-dev libftgl-dev libmysqlclient-dev libfftw3-dev libcfitsio-dev libgraphviz-dev libavahi-compat-libdnssd-dev libldap2-dev python2-dev libxml2-dev libkrb5-dev libgsl-dev
echo ""

# Get the ROOT distribution version and unpack it
echo "3) Downloading ROOT v6.28.04:"
wget https://root.cern/download/root_v6.28.10.Linux-ubuntu22-x86_64-gcc11.4.tar.gz
gunzip root_v6.28.10.Linux-ubuntu22-x86_64-gcc11.4.tar.gz
tar -xvf root_v6.28.10.Linux-ubuntu22-x86_64-gcc11.4.tar
echo ""

# Clear and move files
echo "4) Moving ROOT in the HEP_deps directory:"
rm root_v6.28.10.Linux-ubuntu22-x86_64-gcc11.4.tar
mkdir -p HEP_deps
mv root HEP_deps
echo "All done."