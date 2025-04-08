![logo](logo.png)

<p align="center">
	<a href="https://doi.org/10.5281/zenodo.10877157"><img title="DOI" alt="DOI" src="https://zenodo.org/badge/DOI/10.5281/zenodo.8029028.svg">
  <a href="https://qunfold.readthedocs.io/en/latest/"><img title="docs" alt="docs" src="https://readthedocs.org/projects/qunfold/badge/?version=latest"></br>
  <img title="version" alt="version" src="https://img.shields.io/badge/version-v0.3-informational?color=red">
  <img title="license" alt="license" src="https://img.shields.io/badge/license-MIT-informational?color=red">
	<img title="python" alt="python" src="https://img.shields.io/badge/python-≥3.9-informational?color=red">
  <a href="https://skillicons.dev">
    <img src="https://skillicons.dev/icons?i=linux,anaconda,docker" width="80px;"/>
</p>

## Table of contents
- [Introduction](#introduction)
- [Documentation](#documentation)
- [Installation](#installation)
  - [User-mode](#user-mode)
  - [Dev-mode](#dev-mode)
  - [Docker container](#docker-container)
- [Usage example](#usage-example)
- [Unfolding studies](#unfolding-studies)
  - [HEP dependencies](#hep-dependencies)
  - [Performance analysis](#performance-analysis)
- [Main developers](#main-developers)
- [Other contributors](#other-contributors)

## Introduction
This package provides a quantum-based approach to the *statistical unfolding* problem in High-Energy Physics (HEP). The technique is based on the reformulation of this task as a [Quadratic Unconstrained Binary Optimization](https://en.wikipedia.org/wiki/Quadratic_unconstrained_binary_optimization) (QUBO) problem to be solved by [Quantum Annealing](https://en.wikipedia.org/wiki/Quantum_annealing) (QA) on [D-Wave](https://www.dwavesys.com/) quantum devices.

The code is written in Python and relies on [`numpy`](https://numpy.org/) arrays as basic data structures. However, the package also includes simple tools to convert [`ROOT`](https://root.cern/) data to `numpy`, allowing HEP scientists to run the algorithms for their specific use-cases with a minimal effort.
The software is designed leveraging the powerful [D-Wave Ocean SDK](https://docs.ocean.dwavesys.com/en/stable/), which provides several tools to define the QUBO model and run widely-used heuristics for classical computers (e.g. Simulated Annealing) as well as hybrid/quantum solvers running on real QA [D-Wave Systems](https://docs.dwavesys.com/docs/latest/).

The idea for this project was inspired by the work done by K. Cormier, R. Di Sipio, and P. Wittek in 2019 (see [here](https://www.dwavesys.com/resources/application/unfolding-measurement-distributions-in-high-energy-physics-experiments-via-quantum-annealing/) for the links to an oral presentation and the published paper).

:warning: For a comprehensive introduction to the classical problem, see Chapter 11 of the book [*Statistical Data Analysis*](https://academic.oup.com/book/54868) by G. Cowan.

## Documentation
###### (work in progress...)
[Read the Docs](https://qunfold.readthedocs.io/en/latest/): API documentation for all the available features of the package.

## Installation
### *User-mode*
To install the `QUnfold` latest version released on [PyPI](https://pypi.org/project/QUnfold/) in user-mode you can do:
```shell
pip install QUnfold
```

If you also want to enable the classical [Gurobi](https://www.gurobi.com/) solver both for the integer optimization problem and the QUBO problem, you need to install `QUnfold` including this additional requirement:
```shell
pip install QUnfold[gurobi]
```

### *Dev-mode*
To create a dedicated [`conda`](https://docs.conda.io/en/latest/) environment and install `QUnfold` in developer-mode you can do:
```shell
conda create --name qunfold-dev python==3.10.14
conda activate qunfold-dev
git clone https://github.com/Quantum4HEP/QUnfold.git
cd QUnfold
pip install --upgrade -r requirements-dev.txt
pip install -e .[gurobi]
```

### *Docker container*
Two different Docker images are ready to be downloaded from DockerHub to start playing with a containerized version of `QUnfold`:
- [`qunfold`](https://hub.docker.com/r/quantum4hep/qunfold/tags): minimal working version for testing basic functionalities
- [`qunfold-dev`](https://hub.docker.com/r/quantum4hep/qunfold-dev/tags): full version based on the `conda` distribution for Python, including the installation of `ROOT` framework and `RooUnfold` library for expert users in High-Energy Physics

Both the containerized solutions offer the possibility to use `QUnfold` running a JupyterLab web-based environment on your favourite browser. First, run the Docker container with the porting option as follows:
```docker
docker run -itp 8888:8888 qunfold
```
Secondly, once the container has started, launch `jupyter-lab` with the following command:
```shell
jupyter-lab --ip=0.0.0.0
```

## Usage example
Here is a minimal example showing how to use `QUnfold`. The code snippet shows how to create an instance of the unfolder class, initialize the QUBO model, and run the simulated annealing algorithm to solve the problem.
```python
from qunfold import QUnfolder

# Define your input response matrix and measured histogram as numpy arrays
response = ...
measured = ...
binning = ...

# Create the QUnfolder object and initialize the QUBO model
unfolder = QUnfolder(response, measured, binning, lam=0.1)
unfolder.initialize_qubo_model()

# Run one of the available solvers to get the unfolding result
sol, cov = unfolder.solve_simulated_annealing(num_reads=100)
```

The figures show a specific example of a given response matrix as well as the correponding histograms for the case of a *gamma* distribution with Gaussian smearing.
<p align="center">
    <img src="https://github.com/Quantum4HEP/QUnfold/blob/main/examples/simneal_response.png" style="width: 45%;">
    <img src="https://github.com/Quantum4HEP/QUnfold/blob/main/examples/simneal_result.png" style="width: 45%;">
</p>

If you are working in High-Energy Physics, your response matrix might be a `RooUnfoldResponse` object and your measured histogram is probably stored in a `ROOT.TH1`.
The `qunfold.root2numpy` module provides some simple functions to convert these objects to standard numpy arrays:
```python
from qunfold.root2numpy import TH1_to_numpy, TH2_to_numpy

# Convert ROOT.TH1 measured histogram to numpy array
measured = TH1_to_numpy(measured)

# Convert RooUnfoldResponse object to numpy array
response = TH2_to_numpy(response.Hresponse())
```

For a complete example on how to run the `QUnfold` solvers and plot the final results, you can take a look at the [examples](https://github.com/Quantum4HEP/QUnfold/tree/main/examples) directory in the repository.

## Unfolding studies
This section contains instructions to solve the unfolding problem by classical methods (widely used in HEP data analysis) as well as the `QUnfold` quantum-based method. It also provides several tools and examples to compare the results of the two different approaches.

### HEP dependencies
To run the classical unfolding algorithms you need to install the `ROOT` framework developed by CERN (see documentation [here](https://root.cern/doc/v628/)) and the specialized `RooUnfold` library (see documentation [here](http://roounfold.web.cern.ch/index.html)).
On Linux or Mac OS this can be done automatically by running the following script in the root directory of the repository AFTER having activated the python environment you want to use (this can take a long time):
```shell
./install_roounfold.sh
```
This step can be ignored if you are using the `qunfold-dev` Docker container since the corresponding Docker image already includes the required HEP software.

### Performance analysis
The code available in the [studies](https://github.com/Quantum4HEP/QUnfold/tree/main/studies) directory can be used to generate syntetic data samples according to common HEP probability density functions (*normal*, *gamma*, *exponential*, *Breit-Wigner*, *double-peaked*) and apply a smearing to roughly simulate the distortion effects due to limited efficiency, acceptance, and space/time resolution of a given detector.

Then, unfolding is performed by several classical, hybrid, and quantum techniques and the results are studied to compare the performance of the different methods.
In particular, the algorithms currently available are the following:
- `RooUnfold` framework:
  - Matrix Inversion unfolding (MI)
  - Bin-by-Bin unfolding (B2B) 
  - Iterative Bayesian Unfolding (IBU)
  - Tikhonov regularized unfolding (SVD)
- `QUnfold` library:
  - [D-Wave Simulated Annealing](https://docs.ocean.dwavesys.com/en/stable/docs_samplers/reference.html#simulated-annealing) (SA) for QUBO unfolding
  - [D-Wave Hybrid solver](https://docs.ocean.dwavesys.com/en/stable/docs_system/reference/samplers.html#leaphybridsampler) (HYB) for QUBO unfolding
  - [D-Wave Quantum Annealing](https://docs.ocean.dwavesys.com/en/stable/docs_system/reference/samplers.html#dwavesampler) (QA) for QUBO unfolding
  - `Gurobi` solver for integer optimization formulation of the unfolding problem
  - `Gurobi` solver for QUBO unfolding problem

:warning: The HYB and QA solvers require access to a real D-Wave quantum annealer (1 min/month available for free via the [D-Wave Leap](https://cloud.dwavesys.com/leap/login/) cloud platform) while the Gurobi solvers (Python API) require a software license (freely available for 18 months).

***

### Main developers
<table>
  <tr>
    <td align="center"><a href="https://justwhit3.github.io/"><img src="https://avatars.githubusercontent.com/u/48323961?v=4" width="100px;" alt=""/><br /><sub><b>Gianluca Bianco</b></sub></a></td>
    <td align="center"><a href="https://github.com/SimoneGasperini"><img src="https://avatars2.githubusercontent.com/u/71086758?s=400&v=4" width="100px;" alt=""/><br /><sub><b>Simone Gasperini</b></sub></a></td>
  </tr>
</table>

### Other contributors
<table>
  <tr>
    <td align="center"><a href="https://github.com/DrWatt"><img src="https://avatars.githubusercontent.com/u/33673848?v=4" width="60px;" alt=""/><br /><sub><b>Marco Lorusso</b></sub></a></td>
  </tr>
</table>
