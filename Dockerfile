FROM ubuntu:22.04

RUN apt-get update && \
apt-get install -y sudo git vim nano wget curl bzip2 libx11-6 libxpm4 libxft2 libxext6 cmake libc6-dev python3 python3-pip && \
apt-get upgrade -y && \
apt-get clean

RUN useradd -m -g sudo -s /bin/bash qunfold
RUN echo "qunfold:qunfold" | chpasswd
USER qunfold
WORKDIR /home/qunfold

RUN git clone https://github.com/Quantum4HEP/QUnfold.git
WORKDIR /home/qunfold/QUnfold
RUN python3 -m pip install --upgrade pip
ENV PATH=$PATH:/home/qunfold/.local/bin

RUN pip3 install -e . 
RUN pip3 install jupyter

RUN pip3 cache purge

WORKDIR /home/qunfold

CMD ["/bin/bash"]
