![Logo](https://github.com/JustWhit3/QUnfold/blob/main/img/repository/logo.png)

<h3 align="center">Unfold statistical distributions using quantum machine learning</h3>
<p align="center">
  <img title="v0.0" alt="v0.0" src="https://img.shields.io/badge/version-v0.0-informational?style=flat-square">
  <img title="MIT License" alt="license" src="https://img.shields.io/badge/license-MIT-informational?style=flat-square">
	<img title="Python" alt="Python" src="https://img.shields.io/badge/Lang-Python-informational?style=flat-square"><br/>
	<img title="Code size" alt="code size" src="https://img.shields.io/github/languages/code-size/JustWhit3/QUnfold?color=red">
	<img title="Repo size" alt="repo size" src="https://img.shields.io/github/repo-size/JustWhit3/QUnfold?color=red">
	<img title="Lines of code" alt="total lines" src="https://img.shields.io/tokei/lines/github/JustWhit3/QUnfold?color=red"></br>
  <img title="codeq" alt="codeq" src="https://github.com/JustWhit3/QUnfold/actions/workflows/codeql-analysis.yml/badge.svg">
  <img title="doc" alt="doc" src="https://github.com/JustWhit3/QUnfold/actions/workflows/DocGenerator.yml/badge.svg">
</p>

***

## Table of contents

- [Introduction](#introduction)
- [Documentation](#documentation)
- [Studies](#studies)
  - [Install HEP dependencies](#install-hep-dependencies)
  - [Generate pseudo-data](#generate-pseudo-data)
  - [`RooUnfold`](#roounfold)
  - [Benchmarks](#benchmarks)
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
> :warning: an input filter is applied to the Doxygen generator, in order to convert Python docstrings into Doxygen format. This filter lies in `scripts/py_filter.sh`.

## Studies

This section contains instructions to run unfolding with other packages in order to do comparisons with `QUnfold`. All the code lies under the `studies` directory.

All the dependencies are managed by [tox](https://tox.wiki/en/latest/), except [the ones related to HEP](#install-hep-dependencies).

### Install HEP dependencies

To run all the studies you will need to install some HEP dependencies:

- [`ROOT`](https://root.cern/releases/release-62804/): **v6.28/04**. Soon more instructions to install this.

- [`RooUnfold`](https://gitlab.cern.ch/RooUnfold/RooUnfold): **v3.0.0**. See [this user guide](https://statisticalmethods.web.cern.ch/StatisticalMethods/unfolding/RooUnfold_01-Methods_PY/) for a good user guide, the official [Doxygen](http://roounfold.web.cern.ch/index.html) page and the [repository](https://github.com/roofit-dev/RooUnfold). 

This dependencies can be easily installed from the `root` directory of the repository using the related scripts:

```shell
./scripts/fetchROOT.sh
./scripts/fetchRooUnfold.sh
```

> :warning: these installers work only for Ubuntu.

They will be installed into the `HEP_deps` directory of the repository.

### Generate pseudo-data

To generate new pseudo-data:

```shell
tox -e generator
```

To modify the generator parameters (samples, distribution...) open the `generator/generator.sh` script and modify it.

Pseudo-data used for testing the unfolding lie into the `data` directory. Each sub-directory contains truth data, measured data and response matrix for each generated distribution.

Current distributions supported for generation:

- [Breit-Wigner](https://en.wikipedia.org/wiki/Relativistic_Breit%E2%80%93Wigner_distribution)
- [Normal](https://en.wikipedia.org/wiki/Normal_distribution)
- Double peaked

After the pseudo-data generation, create a config file with all the generated distributions name, used for the next steps. From the root directory of the repository do:

```shell
./scripts/create_distr_config.sh
```

this will create a Json file into the `config` directory. Re-run this script every time you generate new distributions.

### `RooUnfold`

This section is related to the `RooUnfold` studies. Be sure of being into the `studies` directory before proceeding.

To run classical unfolding example with `RooUnfold:

```shell
tox -e RooUnfold
```

open the `RooUnfold/unfolding.sh` bash script to modify the unfolding parameters.

Data of the unfolded histogram will be saved into the `studies/output/RooUnfold` directory, while comparisons among measured, truth and unfolded (reco) histograms into the `img/RooUnfold` directory.

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