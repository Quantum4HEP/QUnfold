# QUnfold

## Table of contents

- [Introduction](#introduction)
- [Documentation](#documentation)
- [Studies](#studies)
  - [Install HEP dependencies](#install-hep-dependencies)
  - [Generate pseudo-data](#generate-pseudo-data)
  - [`RooUnfold`](#roounfold)
  - [Tests](#tests)
- [Credits](#credits)
  - [Main developers](#main-developers)
  - [Other contributors](#other-contributors)
- [Stargazers over time](#stargazers-over-time)

## Introduction

Idea was born by me and Simone during a quantum computing school of Cineca and is inspired by the work done by [Riccardo Di Sipio](https://github.com/rdisipio) which can be found [here](https://github.com/rdisipio/quantum_unfolding).

Work in progress...

## Documentation

Further documentation resources are listed here:

- [Doxygen page](https://justwhit3.github.io/QUnfold/): contains documentation about all the functions and classes of the module.

## Studies

This section contains instructions to run unfolding with other packages in order to do comparisons with `QUnfold`. All the code lies under the `studies` directory.

All the dependencies are managed by [tox](https://tox.wiki/en/latest/), except [the ones related to HEP](#install-hep-dependencies).

### Generate pseudo-data

Some pseudo-data used for testing the unfolding lie into the `data` directory. Each sub-directory contains truth data, measured data, response matrix and some plots for data visualization.

To generate new pseudo-data:

```shell
tox -e generator
```

To modify the generator parameters (samples, distribution...) open the `generator/generator.sh` script and modify it.

Current distributions supported for generation:

- [Breit-Wigner](https://en.wikipedia.org/wiki/Relativistic_Breit%E2%80%93Wigner_distribution)
- [Normal](https://en.wikipedia.org/wiki/Normal_distribution)
- Double peaked

### Install HEP dependencies

To run all the studies you will need to install some HEP dependencies:

- [`ROOT`](https://root.cern/releases/release-62804/): **v6.28/04**. Soon more instructions to install this.

- [`RooUnfold`](https://gitlab.cern.ch/RooUnfold/RooUnfold): **v3.0.0**. See [this user guide](https://statisticalmethods.web.cern.ch/StatisticalMethods/unfolding/RooUnfold_01-Methods_PY/) for a good user guide, the official [Doxygen](http://roounfold.web.cern.ch/index.html) page and the [repository](https://github.com/roofit-dev/RooUnfold). This dependency can be installed from the `root` directory of the repository using the related script:

```shell
./scripts/fetchRooUnfold.sh
```

### `RooUnfold`

This section is related to the `RooUnfold` studies. Be sure of being into the `studies` directory before proceeding.

To run classical unfolding example with `RooUnfold:

```shell
tox -e RooUnfold
```

open the `RooUnfold/unfolding.sh` bash script to modify the unfolding parameters.

### Tests

Tu run the tests related to the functions developed for the studies run (optional):

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