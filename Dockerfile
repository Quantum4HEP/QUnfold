FROM ubuntu:22.04

RUN apt-get update && \
apt-get install -y sudo git vim wget python3 python3-pip && \
apt-get upgrade -y && \
apt-get clean

RUN useradd -m -g sudo -s /bin/bash qunfold
RUN echo "qunfold:qunfold" | chpasswd
USER qunfold
WORKDIR /home/qunfold

RUN git clone https://github.com/Quantum4HEP/QUnfold.git
WORKDIR /home/qunfold/QUnfold
RUN pip3 install -e . 
RUN pip3 install jupyterlab

RUN pip3 cache purge

WORKDIR /home/qunfold

CMD ["/bin/bash"]
