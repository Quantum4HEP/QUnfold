FROM condaforge/miniforge3:24.3.0-0
RUN apt update && apt upgrade -y
RUN apt install -y sudo
RUN apt clean
RUN useradd -ms /bin/bash qunfold
RUN qunfold:qunfold | chpasswd
USER qunfold
WORKDIR /home/qunfold
RUN git clone https://github.com/JustWhit3/QUnfold.git
WORKDIR /home/qunfold/QUnfold
RUN pip install -e . --no-warn-script-location
WORKDIR /home/qunfold/
RUN pip cache purge
RUN conda clean -all -y