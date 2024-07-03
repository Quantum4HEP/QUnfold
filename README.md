<p align="center">
  <img src="https://github.com/JustWhit3/QUnfold/blob/main/logo.png">
</p>

<p align="center">
	<a href="https://doi.org/10.5281/zenodo.10877157"><img title="DOI" alt="DOI" src="https://zenodo.org/badge/DOI/10.5281/zenodo.8029028.svg"></br></a>
  <img title="v0.3" alt="v0.3" src="https://img.shields.io/badge/version-v0.3-informational?color=red"></a>
  <img title="license" alt="license" src="https://img.shields.io/badge/license-MIT-informational?color=red"></a>
	<img title="python" alt="python" src="https://img.shields.io/badge/python-≥3.8-informational?color=red"></br>
  <img title="codeql" alt="codeql" src="https://github.com/JustWhit3/QUnfold/actions/workflows/codeql.yml/badge.svg">
  <a href="https://qunfold.readthedocs.io/en/latest/"><img title="docs" alt="docs" src="https://readthedocs.org/projects/qunfold/badge/?version=latest"></a>
  <a href="https://hub.docker.com/r/marcolorusso/qunfold/tags"><img alt="docker build" src="https://img.shields.io/docker/automated/marcolorusso/qunfold"></a>
</p>

## Table of contents
- [Introduction](#introduction)
- [Documentation](#documentation)
- [Installation](#installation)
  - [User-mode](#user-mode)
  - [Developer-mode](#developer-mode)
- [Usage example](#usage-example)
- [Unfolding studies](#unfolding-studies)
  - [HEP dependencies](#hep-dependencies)
  - [Performance analysis](#performance-analysis)
- [Main developers](#main-developers)
- [Other contributors](#other-contributors)

## Introduction
This package provides an implementation of a quantum-based solver for the [statistical unfolding](https://indico.cern.ch/event/735431/contributions/3275244/attachments/1784103/2904689/PhystatNu_2019.pdf) problem formulated as a [Quadratic Unconstrained Binary Optimization](https://en.wikipedia.org/wiki/Quadratic_unconstrained_binary_optimization) (QUBO) problem.
The code is written in Python and based on [`NumPy`](https://numpy.org/), but it allows to be used on [`ROOT`](https://root.cern/) data for application in High-Energy Physics (HEP).

The idea is inspired by the work done by [Riccardo Di Sipio](https://github.com/rdisipio) et al. which can be found [here](https://github.com/rdisipio/quantum_unfolding).
For a more detailed description of the model, take a look at the [Wiki page](https://github.com/JustWhit3/QUnfold/wiki/Model-description).

The package currently provides the following algorithms to solve the unfolding problem:

- *Simulated annealing*: [D-Wave](https://www.dwavesys.com/) implementation of the standard simulated annealing algorithm running on classical hardware resources
- *Hybrid solver*: complex approach combining both classical computing (for the decomposition of the original problem into smaller sub-problems) and [D-Wave](https://www.dwavesys.com/) quantum annealing on real hardware
- *Quantum annealing*:  quantum approach fully running on [D-Wave](https://www.dwavesys.com/) quantum annealing hardware.

## Documentation
###### (work in progress...)
- [Wiki page](https://github.com/JustWhit3/QUnfold/wiki): description of the theoretical model and examples on how to use the package.
- [Read the Docs](https://qunfold.readthedocs.io/en/latest/): API documentation for all the available features of the package.

## Installation
### *User-mode*
To install the `QUnfold` latest version released on [PyPI](https://pypi.org/project/QUnfold/) in user-mode you can do:
```shell
pip install QUnfold
```

### *Developer-mode*
To create a dedicated [`conda`](https://docs.conda.io/en/latest/) environment and install `QUnfold` in developer-mode you can do:
```shell
conda create --name qunfold-dev python==3.10
conda activate qunfold-dev
git clone https://github.com/JustWhit3/QUnfold.git
cd QUnfold
pip install -r requirements-dev.txt
pip install -e .
```

## Usage example
Here is a simple code example showing how to use `QUnfold`:
```python
from QUnfold import QUnfoldQUBO

# Define your input response matrix and measured histogram as numpy arrays
response = ...
measured = ...

# Create the QUnfoldQUBO object and initialize the QUBO model
unfolder = QUnfoldQUBO(response, measured, lam=0.1)
unfolder.initialize_qubo_model()

# Run one of the available solvers to get the unfolding result
sol, err, cov = unfolder.solve_simulated_annealing(num_reads=100)
```
<p align="center">
    <img src="https://github.com/JustWhit3/QUnfold/blob/main/examples/simneal_response.png" style="width: 45%;">
    <img src="https://github.com/JustWhit3/QUnfold/blob/main/examples/simneal_result.png" style="width: 45%;">
</p>

:warning: The response matrix must be normalized (see [here](https://github.com/JustWhit3/QUnfold/wiki/How-to-use#:~:text=The%20response%20matrix%20must%20be%20normalized)).

If you are working in High-Energy Physics, your response matrix might be a `RooUnfoldResponse` object and your measured histogram is probably stored as a `ROOT.TH1` object.
The `QUnfold.utils` module provides some simple functions to convert these objects to standard numpy arrays:
```python
from QUnfold.utils import TH1_to_numpy, TH2_to_numpy

# Convert ROOT.TH1 measured histogram to numpy array
measured = TH1_to_numpy(measured)

# Convert RooUnfoldResponse object to numpy array
response = TH2_to_numpy(response.Hresponse())
```

For a complete example on how to run the `QUnfold` solvers and plot the final results, you can take a look at the [examples](https://github.com/JustWhit3/QUnfold/tree/main/examples) directory in the repository.

## Unfolding studies
This section contains instructions to solve the unfolding problem by standard classical algotihms (widely used in HEP data analysis) as well as the `QUnfold` quantum-based method. It also provides several tools and examples to compare the results of the two different approached.
Check out the [studies](https://github.com/JustWhit3/QUnfold/tree/main/studies) directory in this repository to learn more.

### HEP dependencies
To run the code you need to install the following HEP frameworks:
- `ROOT v6.28/10` (see documentation [here](https://root.cern/doc/v628/))
- `RooUnfold v3.0.0` (see documentation [here](http://roounfold.web.cern.ch/index.html))

In your Ubuntu system, this can be done automatically by executing the following commands from the root directory of the repository:
```shell
./scripts/fetchROOT.sh
source HEP_deps/root/bin/thisroot.sh
./scripts/fetchRooUnfold.sh
```

### Performance analysis
The code can be used to generate syntetic data samples according to common HEP probability density functions (*normal*, *gamma*, *exponential*, *Breit-Wigner*, *double-peaked*) and apply a smearing to roughly simulate the distortion effects due to limitions efficiency, acceptance, and space/time resolution of a given detector.

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
  - [D-Wave Quantum Annealing](https://docs.ocean.dwavesys.com/en/stable/docs_system/reference/samplers.html#dwavesampler) (QA) for QUBO unfolding (QA)

:warning: The HYB and QA solvers require access to a real D-Wave quantum annealer (1 min/month available for free via the [D-Wave Leap](https://cloud.dwavesys.com/leap/login/) cloud platform).

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
    <td align="center"><a href="https://github.com/DrWatt"><img src="https://avatars.githubusercontent.com/u/33673848?v=4" width="80px;" alt=""/><br /><sub><b>Marco Lorusso</b></sub></a></td>
  </tr>
</table>
