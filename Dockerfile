FROM ubuntu
SHELL ["/bin/bash", "-c"]
RUN apt update
RUN apt upgrade -y
RUN apt install -y sudo wget git python3 python3-pip
RUN useradd -m -g sudo -s /bin/bash qunfold
RUN echo 'qunfold:qunfold' | chpasswd
USER qunfold
WORKDIR /home/qunfold
RUN pip3 install QUnfold
RUN pip3 cache purge
CMD /bin/bash
