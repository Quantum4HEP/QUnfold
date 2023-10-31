![Logo](https://github.com/JustWhit3/QUnfold/blob/main/img/repository/logo.png)

<h3 align="center">Unfolding statistical distributions using quantum machine learning</h3>
<p align="center">
	<img title="DOI" alt="DOI" src="https://zenodo.org/badge/DOI/10.5281/zenodo.8029028.svg"></br>
  <img title="v0.1" alt="v0.1" src="https://img.shields.io/badge/version-v0.1-informational?style=flat-square&color=red">
  <img title="MIT License" alt="license" src="https://img.shields.io/badge/license-MIT-informational?style=flat-square&color=red">
	<img title="Python" alt="Python" src="https://img.shields.io/badge/Python-≥3.8-informational?style=flat-square&color=red"></br>
  <img title="codeq" alt="codeq" src="https://github.com/JustWhit3/QUnfold/actions/workflows/codeql-analysis.yml/badge.svg">
  <img title="doc" alt="doc" src="https://github.com/JustWhit3/QUnfold/actions/workflows/DocGenerator.yml/badge.svg">
</p>

***

## Table of contents

- [Introduction](#introduction)
- [Documentation](#documentation)
- [How to use](#how-to-use)
  - [NumPy case](#numpy-case)
  - [ROOT case](#root-case)
- [Studies](#studies)
  - [Setup the environment](#setup-the-environment)
  - [Install HEP dependencies](#install-hep-dependencies)
  - [Run the analysis](#run-the-analysis)
  - [Tests](#tests)
- [Credits](#credits)
  - [Main developers](#main-developers)
  - [Other contributors](#other-contributors)
- [Stargazers over time](#stargazers-over-time)

## Introduction

This is an implementation of the [unfolding](https://indico.cern.ch/event/735431/contributions/3275244/attachments/1784103/2904689/PhystatNu_2019.pdf) statistical approach using quantum computation and in particular a [quadratic unconstrain binary optimization](https://en.wikipedia.org/wiki/Quadratic_unconstrained_binary_optimization) approach. To find a detailed implementation of the model you can look at [this](https://www.dropbox.com/scl/fi/umam07m5xiwm3ui335vgr/poster_QUnfold.pdf?rlkey=k5ru4kqkb7ea7g9exvxycbzm3&dl=0) poster. This software is based on [`NumPy`](https://numpy.org/), but you can easily run it also using [`ROOT`](https://root.cern/) framework data (see [below](#root-case)).

For the moment you can use it in two modes:

- Simulated annealing (`solve_simulated_annealing`): which uses simulated quantum computer to compute the result.
- Hybrid solver (`solve_hybrid_sampler`): which uses [D-Wave](https://www.dwavesys.com/) resources to perform hybrid quantum annealing (classical + quantum hardware) to compute the result.

Idea was born by me and [Simone](https://github.com/SimoneGasperini) during a quantum computing school of Cineca and is inspired by the work done by [Riccardo Di Sipio](https://github.com/rdisipio) et al. which can be found [here](https://github.com/rdisipio/quantum_unfolding).

:warning: The project is currently work-in-progress and it is still not ready for production. Some [improvements](https://github.com/JustWhit3/QUnfold/issues) and [issues](https://github.com/JustWhit3/QUnfold/issues/3) should be investigated and solved before releasing and packaging an official version of the software. Any help would be more than welcome! See the [contribution](https://github.com/JustWhit3/QUnfold/blob/main/CONTRIBUTING.md) file if interested.

## How to use

### NumPy case

To run QUnfold on a dataset you can do the following steps:

```python
# Import QUnfold base class and plotter
from QUnfold import QUnfoldQUBO
from QUnfold import QUnfoldPlotter

# Read data from a file or sample them
# NB: data should be in NumPy or list format
truth = ... # truth distribution
meas = ... # measured distribution
response = ... # response matrix
binning = ... # binning of the distributions

# Unfold!
unfolder = QUnfoldQUBO(response, meas, lam=0.1)
unfolded_SA = unfolder.solve_simulated_annealing(num_reads=200) # use solve_hybrid_sampler method to use real quantum computer hardware

# Plot results
plotter = QUnfoldPlotter(
    response=response,
    measured=meas,
    truth=truth,
    unfolded=unfolded_SA,
    binning=binning,
)
plotter.saveResponse("response.png")
plotter.savePlot("comparison.png", "SA")
```

# ROOT case

To use `ROOT` data add the following steps at the beginning of the code:

```python
# Import ROOT converters
# NB: this will be improved soon...
from studies.functions.ROOT_converter import TH1_to_array, TH2_to_array

# Read data as before...
# Convert data
truth = TH1_to_array(truth)
meas = TH1_to_array(meas)
response = TH2_to_array(response.Hresponse()) # Supposing response was a RooUnfold response

# Perform the analysis as before...
```

Look at the [examples](https://github.com/JustWhit3/QUnfold/tree/main/examples) folder for more how-to examples.

## Documentation

Further documentation resources are listed here:

- [Doxygen page](https://justwhit3.github.io/QUnfold/): contains documentation about all the functions and classes of the module.
- [Contributing file](https://github.com/JustWhit3/QUnfold/blob/main/CONTRIBUTING.md): contains instructions about how to contribute.

> :warning: An input filter is applied to the Doxygen generator, in order to convert Python docstrings into Doxygen format. This filter lies in `scripts/py_filter.sh`.

## Studies

This section contains instructions to run unfolding with other packages in order to do comparisons with `QUnfold`. All the code lies under the `studies` directory.

All the dependencies are managed by [tox](https://tox.wiki/en/latest/), except [the ones related to HEP](#install-hep-dependencies).

### Setup the environment

To setup the environment for `QUnfold` development you need two dependencies:

- [`tox`](https://tox.wiki/en/latest/): at least v4.
- [`conda`](https://docs.conda.io/en/latest/).

To setup the `conda` conda environment to work with the repository (only the first time):

```shell
conda create --name qunfold-dev python==3.10
conda activate qunfold-dev
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip cache purge && pip check
```

and every time you open a new shell:

```shell
conda activate qunfold-dev
```

### Install HEP dependencies

To run all the studies you will need to install some HEP dependencies:

- [`ROOT`](https://root.cern/releases/release-62804/): **v6.28/04**. Soon more instructions to install this.

- [`RooUnfold`](https://gitlab.cern.ch/RooUnfold/RooUnfold): **v3.0.0**. See [this user guide](https://statisticalmethods.web.cern.ch/StatisticalMethods/unfolding/RooUnfold_01-Methods_PY/) for a good user guide, the official [Doxygen](http://roounfold.web.cern.ch/index.html) page and the [repository](https://github.com/roofit-dev/RooUnfold).

This dependencies can be easily installed from the `root` directory of the repository using the related scripts:

```shell
./scripts/fetchROOT.sh
source HEP_deps/root/bin/thisroot.sh
./scripts/fetchRooUnfold.sh
```

> :warning: These installers work only for Ubuntu.

They will be installed into the `HEP_deps` directory of the repository.

If you want to use the `ROOT` version of the repo you must do this command every time you plan to run a code which contains the `ROOT` package:

```shell
source HEP_deps/root/bin/thisroot.sh
```

> :warning: If you want to avoid this, install `ROOT` in your computer.

### Run the analysis

To run the whole analysis script:

```shell
cd studies
tox -e analysis
```

Pseudo-data will be generated following common distributions (double-peaked, normal, etc...) which will be unfolded using `RooUnfold` and the 4 classical common methods:

- Matrix inversion
- Iterative Bayesian unfolding (4 iterations)
- SVD (k=3)
- Bin-to-Bin

Comparisons are performed with `QUnfold` and with the following methods:

- Simulated annealing (lambda=0.2, num_reads=100)

The output plots and chi2 for each distribution will be saved into the `img` directory.

### Benchmarks

Benchmarks are performed in order to compare the performances of the various unfolding algorithms. To run them:

```shell
tox -e tests
```

The output data will be saved into the `studies/output/benchmarks` directory, while performance histograms into the `img/benchmarks` directory.

### Tests

Tu run unit tests related to the functions developed for the studies run (optional):

```shell
tox -e tests
```

## Credits

### Main developers

<table>
  <tr>
    <td align="center"><a href="https://justwhit3.github.io/"><img src="https://avatars.githubusercontent.com/u/48323961?v=4" width="100px;" alt=""/><br /><sub><b>Gianluca Bianco</b></sub></a></td>
    <td align="center"><a href="https://github.com/SimoneGasperini"><img src="https://avatars2.githubusercontent.com/u/71086758?s=400&v=4" width="100px;" alt=""/><br /><sub><b>Simone Gasperini</b></sub></a></td>
  </tr>
</table>

### Other contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

Empty for the moment.

## Stargazers over time

[![Stargazers over time](https://starchart.cc/JustWhit3/QUnfold.svg)](https://starchart.cc/JustWhit3/QUnfold)