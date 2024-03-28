<p align="center">
  <img src="https://github.com/JustWhit3/QUnfold/blob/main/logo.png" alt="Logo">
</p>

<p align="center">
	<a href="https://doi.org/10.5281/zenodo.10877157"><img title="DOI" alt="DOI" src="https://zenodo.org/badge/DOI/10.5281/zenodo.8029028.svg"></br></a>
  <a href="https://github.com/JustWhit3/QUnfold/releases"><img title="v0.3" alt="v0.3" src="https://img.shields.io/badge/version-v0.3-informational?style=flat-square&color=red"></a>
  <a href="https://github.com/JustWhit3/QUnfold/blob/main/LICENSE"><img title="MIT License" alt="license" src="https://img.shields.io/badge/license-MIT-informational?style=flat-square&color=red"></a>
	<img title="Python" alt="Python" src="https://img.shields.io/badge/Python-≥3.8-informational?style=flat-square&color=red"></br>
  <img title="codeq" alt="codeq" src="https://github.com/JustWhit3/QUnfold/actions/workflows/codeql-analysis.yml/badge.svg">
  <a href="https://qunfold.readthedocs.io/en/latest/"><img title="doc" alt="doc" src="https://readthedocs.org/projects/qunfold/badge/?version=latest"></a>
  <a href="https://hub.docker.com/r/marcolorusso/qunfold/tags"><img alt="Docker Automated build" src="https://img.shields.io/docker/automated/marcolorusso/qunfold"></a>
</p>

***

## Table of contents

- [Introduction](#introduction)
- [Documentation](#documentation)
- [Developer environment](#developer-environment)
- [Pip installation](#pip-installation)
- [How to use](#how-to-use)
  - [NumPy case](#numpy-case)
  - [ROOT case](#root-case)
- [Tests](#tests)
- [Examples](#examples)
- [Studies](#studies)
  - [Install HEP dependencies](#install-hep-dependencies)
  - [Run the analysis](#run-the-analysis)
  - [Run the paper studies](#run-the-paper-studies)
- [Projects using QUnfold](#projects-using-qunfold)
- [Credits](#credits)
  - [Main developers](#main-developers)

## Introduction

This module consists of an implementation of the [unfolding](https://indico.cern.ch/event/735431/contributions/3275244/attachments/1784103/2904689/PhystatNu_2019.pdf) statistical approach using quantum computation and in particular a [quadratic unconstrain binary optimization](https://en.wikipedia.org/wiki/Quadratic_unconstrained_binary_optimization) approach. To find a detailed implementation of the model you can look at [this](https://github.com/JustWhit3/QUnfold/wiki/Model-description) wiki page. This software is based on [`NumPy`](https://numpy.org/), but you can easily run it also using [`ROOT`](https://root.cern/) framework data (see [below](#root-case)).

For the moment you can use it in two modes:

- Simulated annealing (`solve_simulated_annealing`): which uses simulated quantum computer to compute the result.
- Quantum annealing (`solve_quantum_annealing`): which uses [D-Wave](https://www.dwavesys.com/) resources to perform  quantum annealing to compute the result.
- Hybrid solver (`solve_hybrid_sampler`): which uses [D-Wave](https://www.dwavesys.com/) resources to perform hybrid quantum annealing (classical + quantum hardware) to compute the result.

Idea was born by me and [Simone](https://github.com/SimoneGasperini) during a quantum computing school of Cineca and is inspired by the work done by [Riccardo Di Sipio](https://github.com/rdisipio) et al. which can be found [here](https://github.com/rdisipio/quantum_unfolding).

:warning: The project is currently work-in-progress and it is still not ready for production. Some [improvements and issues](https://github.com/JustWhit3/QUnfold/issues) should be investigated and solved before releasing and packaging an official version of the software. Any help would be more than welcome! See the [contribution](https://github.com/JustWhit3/QUnfold/blob/main/CONTRIBUTING.md) file if interested.

## Documentation

Research documentation:

- [Poster](https://www.dropbox.com/scl/fi/umam07m5xiwm3ui335vgr/poster_QUnfold.pdf?rlkey=k5ru4kqkb7ea7g9exvxycbzm3&dl=0) at the 2023 Julich Summer School.

Further documentation:

- [Wiki pages](https://github.com/JustWhit3/QUnfold/wiki): contain detailed description of each feature and examples.
- [ReadtheDocs page](https://qunfold.readthedocs.io/en/latest/): contains documentation about all the functions and classes of the module.
- [Contributing file](https://github.com/JustWhit3/QUnfold/blob/main/CONTRIBUTING.md): contains instructions about how to contribute.

## Developer environment

To setup the environment for `QUnfold` development you need [`conda`](https://docs.conda.io/en/latest/):

```shell
conda create --name qunfold-dev python==3.10
conda activate qunfold-dev
pip install -r requirements.txt
pip install -e src/
```

and every time you open a new shell:

```shell
conda activate qunfold-dev
```

## Pip installation

To install the latest released version of `QUnfold` from [PyPI](https://pypi.org/project/QUnfold/) you can do:

```shell
pip install QUnfold
```

## How to use

### NumPy case

To run QUnfold on a dataset you can do the following steps:

```python
# Import QUnfold base class and plotter
from QUnfold import QUnfoldQUBO
from QUnfold import QUnfoldPlotter

# Read numpy data from a file or sample them
truth = ... # truth distribution
measured = ... # measured distribution
response = ... # response matrix
binning = ... # binning of the distributions

# Run unfolding
unfolder = QUnfoldQUBO(response, measured, lam=0.1)
unfolder.initialize_qubo_model()
unfolded_SA, error_SA, cov_matrix_SA, corr_matrix_SA = unfolder.solve_simulated_annealing(num_reads=10, num_toys=100)

# Plot results
plotter = QUnfoldPlotter(
    response=response,
    measured=measured,
    truth=truth,
    unfolded=unfolded_SA,
    error=error_SA,
    binning=binning,
    chi2=compute_chi2(unfolded, truth, cov_matrix),
)
plotter.plotResponse()
plotter.plot()
```

which will produce a similar result to this unfolded normal distribution:

<p align="center">
    <img src="https://github.com/JustWhit3/QUnfold/blob/main/examples/simneal_response.png" style="width: 45%;">
    <img src="https://github.com/JustWhit3/QUnfold/blob/main/examples/simneal_result.png" style="width: 45%;">
</p>

:warning: The response matrix must be normalized in order to keep the correct binary encoding of input data. The procedure is described in [this](https://github.com/JustWhit3/QUnfold/wiki/How-to-use#:~:text=The%20response%20matrix%20must%20be%20normalized%20before%20unfolding) twiki page.

:warning: To get more information about how the error and covariance/correlation matrices are compute see [this](https://github.com/JustWhit3/QUnfold/wiki/How-to-use#error-computation) twiki page.

### ROOT case

To use `ROOT` data add the following steps at the beginning of the code:

```python
# Import ROOT converters
from QUnfold.utility import TH1_to_array, TH2_to_array

# Read data as before...
# Convert data from ROOT to numpy
truth = TH1_to_array(truth)
measured = TH1_to_array(measured)
response = TH2_to_array(response.Hresponse())

# Run analysis as before...
```

## Tests

Tests are performed using [`pytest`](https://docs.pytest.org/en/7.4.x/) in verbose mode. To run them:

```shell
python -m pytest tests
```

## Examples

Look at the [examples](https://github.com/JustWhit3/QUnfold/tree/main/examples) folder for more details. To run the example:

```bash
python examples/simneal_script.py
```

Results are saved into the `/examples` directory.

## Studies

This section contains instructions to run unfolding with other packages in order to do comparisons with `QUnfold`. All the code lies under the `studies` directory.

### Install HEP dependencies

To run all the studies you will need to install some HEP dependencies:

- [`ROOT`](https://root.cern/releases/release-62804/): **v6.28/10**.

- [`RooUnfold`](https://gitlab.cern.ch/RooUnfold/RooUnfold): **v3.0.0**. See [this user guide](https://statisticalmethods.web.cern.ch/StatisticalMethods/unfolding/RooUnfold_01-Methods_PY/) for a good user guide, the official [Doxygen](http://roounfold.web.cern.ch/index.html) page and the [repository](https://github.com/roofit-dev/RooUnfold).

This dependencies can be easily installed from the `root` directory of the repository using the related scripts:

```shell
./scripts/fetchROOT.sh
source HEP_deps/root/bin/thisroot.sh
./scripts/fetchRooUnfold.sh
```

They will be installed into the `HEP_deps` directory of the repository.

If you want to use the `ROOT` version of the repo you must do this command every time you plan to run a code which contains the `ROOT` package:

```shell
source HEP_deps/root/bin/thisroot.sh
```

### Run the analysis

To run the whole analysis script:

```shell
python studies/analysis.py
```

Pseudo-data will be generated following common distributions (normal, exponential, etc...) which will be unfolded using `RooUnfold` and the following common classical methods:

- Iterative Bayesian unfolding (4 iters)
- Tikhonov regularized SVD unfolding (K=2)

Comparisons are performed with the following `QUnfold` methods:

- Simulated Annealing - QUBO unfolding
- D-Wave Hybrid solver - QUBO unfolding

The output plots and chi2 for each distribution will be saved into an external `img` directory.

:warning: An error related to `RooUnfold` may appear, you can ignore it since it is a false positive.

### Run the paper studies

To run the code related to the technical QUnfold paper we are developing:

```shell
python studies/paper.py
```

## Projects using QUnfold

List of projects which import and use `QUnfold`:

- [PyXSec](https://github.com/JustWhit3/PyXSec): Python framework to measure differential cross-section of particle physics processes using classical- and quantum-computing based techniques.

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
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/DrWatt"><img src="https://avatars.githubusercontent.com/u/33673848?v=4" width="100px;" alt=""/><br /><sub><b>DrWatt</b></sub></a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
