![Logo](https://github.com/JustWhit3/QUnfold/blob/main/img/repository/logo.png)

<h3 align="center">Unfolding statistical distributions using quantum machine learning</h3>
<p align="center">
	<img title="DOI" alt="DOI" src="https://zenodo.org/badge/DOI/10.5281/zenodo.8029028.svg"></br>
  <img title="v0.0" alt="v0.0" src="https://img.shields.io/badge/version-v0.0-informational?style=flat-square&color=red">
  <img title="MIT License" alt="license" src="https://img.shields.io/badge/license-MIT-informational?style=flat-square&color=red">
	<img title="Python" alt="Python" src="https://img.shields.io/badge/Python-3.8 / 3.9 / 3.10 / 3.11-informational?style=flat-square&color=red"></br>
  <img title="codeq" alt="codeq" src="https://github.com/JustWhit3/QUnfold/actions/workflows/codeql-analysis.yml/badge.svg">
  <img title="doc" alt="doc" src="https://github.com/JustWhit3/QUnfold/actions/workflows/DocGenerator.yml/badge.svg">
</p>

***

## Table of contents

- [Introduction](#introduction)
- [Documentation](#documentation)
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

Idea was born by me and Simone during a quantum computing school of Cineca and is inspired by the work done by [Riccardo Di Sipio](https://github.com/rdisipio) which can be found [here](https://github.com/rdisipio/quantum_unfolding).

The project is currently at its beginning stages.

Work in progress...

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

Pseudo-data will be generated following common distributions (double-peaked, normal, etc...).

Distributions will be unfolded using `RooUnfold` and the 4 classical common methods:

- Matrix inversion
- Iterative Bayesian unfolding (4 iterations)
- SVD (k=3)
- Bin-to-Bin

<div align="center">
  <p><b>Example of `RooUnfold` unfolding for a double-peaked distribution</b></p>
  <div>
    <img src="https://github.com/JustWhit3/QUnfold/blob/main/img/RooUnfold/double-peaked/unfolded_B2B.png" width="350" style="display:inline-block;">
    <img src="https://github.com/JustWhit3/QUnfold/blob/main/img/RooUnfold/double-peaked/unfolded_IBU.png" width="350" style="display:inline-block;">
    <img src="https://github.com/JustWhit3/QUnfold/blob/main/img/RooUnfold/double-peaked/unfolded_MI.png" width="350" style="display:inline-block;">
    <img src="https://github.com/JustWhit3/QUnfold/blob/main/img/RooUnfold/double-peaked/unfolded_SVD.png" width="350" style="display:inline-block;">
  </div>
</div><br>

Distributions will be unfolded using `QUnfold` with the following methods:

- Simulated annealing (lambda=0.2, num_reads=100)

Finally, comparisons among each unfolding method of the previous studies will be performed.

The output plots and chi2 for each distribution will be saved into the `img` directory.

### Benchmarks

Benchmarks are performed in order to compare the performances of the various unfolding algorithms. To run them:

```shell
tox -e tests
```

The output data will be saved into the `studies/output/benchmarks` directory, while performance histograms into the `img/benchmarks` directory.

<div align="center">
  <p><b>Example of benchmarks for different unfolding methods for a double-peaked distribution</b></p>
  <div>
    <img src="https://github.com/JustWhit3/QUnfold/blob/main/img/benchmarks/double-peaked.png" width="450" style="display:inline-block;">
  </div>
</div><br>

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