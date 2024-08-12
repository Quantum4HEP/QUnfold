FROM ubuntu:22.04

RUN apt-get update && \
apt-get install -y sudo git wget curl bzip2 libx11-6 libxpm4 libxft2 libxext6 cmake libc6-dev && \
apt-get upgrade -y && \
apt-get clean

RUN useradd -ms /bin/bash qunfold
RUN echo "qunfold:qunfold" | chpasswd
USER qunfold
WORKDIR /home/qunfold

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
bash Miniconda3-latest-Linux-x86_64.sh -b -p /home/qunfold/miniconda3 && \
rm Miniconda3-latest-Linux-x86_64.sh

ENV PATH=/home/qunfold/miniconda3/bin:$PATH
RUN conda create -n qunfold-env python=3.10.14 -y

RUN /bin/bash -c "source activate qunfold-env && \
conda install -c conda-forge root -y"

RUN /bin/bash -c "source activate qunfold-env && \
pip install git+https://gitlab.cern.ch/RooUnfold/RooUnfold"

RUN git clone https://github.com/JustWhit3/QUnfold.git
WORKDIR /home/qunfold/QUnfold
RUN /bin/bash -c "source activate qunfold-env && \
pip install -e .[gurobi] && \
pip install -r requirements-dev.txt && \
pip install -r requirements-docs.txt"

RUN conda clean -a -y

WORKDIR /home/qunfold
RUN echo "source activate qunfold-env" >> /home/qunfold/.bashrc
CMD ["/bin/bash"]
