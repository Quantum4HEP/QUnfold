#!/bin/bash



#Installing requirements
echo -e "\U27A1 \033[1;36mInstalling requirements\033[0m"
conda install numpy -y


sudo apt install -y tzdata bzip2 libx11-6 libxpm4 libxft2 libxext6 cmake libc6-dev binutils cmake dpkg-dev g++ gcc libssl-dev git libx11-dev libxext-dev libxft-dev libxpm-dev python3 libtbb-dev libgif-dev gfortran libpcre3-dev libglu1-mesa-dev libglew-dev libftgl-dev libfftw3-dev libcfitsio-dev libgraphviz-dev libavahi-compat-libdnssd-dev libldap2-dev  python3-dev python3-numpy libxml2-dev libkrb5-dev libgsl-dev qtwebengine5-dev nlohmann-json3-dev libmysqlclient-dev libgl2ps-dev liblzma-dev libxxhash-dev liblz4-dev libzstd-dev wget
if [ $? -eq 0 ]; then
    echo -e "\033[1;32mDONE \U2705\033[0m"
else
    echo -e "\033[1;31mInstallation Failed! \U274C\033[0m"
fi

#Downloading root and compilation
echo -e "\U27A1 \033[1;36mCreating root directories\033[0m"
mkdir $HOME/root-cern $HOME/root-build $HOME/root-source
wget https://root.cern/download/root_v6.34.02.source.tar.gz
tar -xf root_v6.34.02.source.tar.gz -C $HOME/root-source
if [ $? -eq 0 ]; then
    echo -e "\033[1;32mDONE \U2705\033[0m"
else
    echo -e "\033[1;31mInstallation Failed! \U274C\033[0m"
fi

echo -e "\U27A1 \033[1;36mStarting root compilation and install\033[0m"
cd root-build
cmake -DCMAKE_INSTALL_PREFIX=$HOME/root-cern $HOME/root-source/root-6.34.02
cmake --build . --target install -- -j4
if [ $? -eq 0 ]; then
    echo -e "\033[1;32mDONE \U2705\033[0m"
else
    echo -e "\033[1;31mInstallation Failed! \U274C\033[0m"
fi

echo -e "\U27A1 \033[1;36mStarting RooUnfold Compiling and install\033[0m"
source $HOME/root-cern/bin/thisroot.sh
git clone https://gitlab.cern.ch/RooUnfold/RooUnfold
mv RooUnfold $HOME/RooUnfold
cd $HOME/RooUnfold
mkdir build
cd build
cmake ..
make -j4



conda install -c conda-forge libstdcxx-ng=12 -y

if [ $? -eq 0 ]; then
    echo -e "\033[1;32mDONE \U2705\033[0m"
else
    echo -e "\033[1;31mInstallation Failed! \U274C\033[0m"
fi


echo "source {$HOME}/root-cern/bin/thisroot.sh" >> /home/$USER/.bashrc
echo "source {$HOME}/RooUnfold/build/setup.sh" >> /home/$USER/.bashrc

# Cleaning
rm -r $HOME/root-build
rm -r $HOME/root-source
rm root_v6.34.02.source.tar.gz










