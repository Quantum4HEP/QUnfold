# QUnfold

## Table of contents

- [Introduction](#introduction)
- [Credits](#credits)
  - [Main developers](#main-developers)
  - [Other contributors](#other-contributors)
- [Studies](#studies)
  - [Comparison with `RooUnfold`](#comparison-with-roounfold)
- [Stargazers over time](#stargazers-over-time)

## Introduction

Idea was born by me and Simone during a quantum computing school of Cineca and is inspired by the work done by [Riccardo Di Sipio](https://github.com/rdisipio) which can be found [here](https://github.com/rdisipio/quantum_unfolding).

Work in progress...

## Studies

This section contains instructions to run unfolding with other packages. All the codes lie under the `studies` directory.

### Comparison with `RooUnfold`

This section is related to the `RooUnfold` comparison studies with `QUnfold`.

Tools version:

- [`RooUnfold`](https://gitlab.cern.ch/RooUnfold/RooUnfold): v3.0.0. See [this user guide](https://statisticalmethods.web.cern.ch/StatisticalMethods/unfolding/RooUnfold_01-Methods_PY/) for a good user guide, the official [Doxygen](http://roounfold.web.cern.ch/index.html) page and the [repository](https://github.com/roofit-dev/RooUnfold).
- [`ROOT`](https://root.cern/releases/release-62804/): v6.28/04
- [`GNU make`](https://www.gnu.org/software/make/): v4.3

First of all enter the `studies/RooUnfold` to fetch and compile `RooUnfold` with this command:

```shell
./fetchRooUnfold.sh
```

this step should be repeated only once.

Tu setup the [tox](https://tox.wiki/en/latest/user_guide.html) environment and run tests (optional):

```shell
tox -e tests
```

To run classical unfolding example:

```shell
tox -e main
```

enter the `unfolding.sh` bash script to modify the unfolding parameters.

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