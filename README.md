![](logo.png)

<h3 align="center">Unfolding statistical distributions using quantum annealing</h3>
<p align="center">
	<img title="DOI" alt="DOI" src="https://zenodo.org/badge/DOI/10.5281/zenodo.8029028.svg"></br>
  <img title="v0.0" alt="v0.0" src="https://img.shields.io/badge/version-v0.0-informational?style=flat-square&color=red">
  <img title="MIT License" alt="license" src="https://img.shields.io/badge/license-MIT-informational?style=flat-square&color=red">
	<img title="Python" alt="Python" src="https://img.shields.io/badge/Python-3.10 / 3.11-informational?style=flat-square&color=red"></br>
</p>

***


## Introduction
The idea was born by me and Simone during a quantum computing school at CINECA and it is inspired by the work done by [Riccardo Di Sipio](https://github.com/rdisipio) which can be found [here](https://github.com/rdisipio/quantum_unfolding).

The project is currently at its beginning stages: work in progress...


### Setup DEV environment
To setup the environment for `QUnfold` development you need to install [`conda`](https://docs.conda.io/en/latest/) and create a separate environment:

```shell
conda create --name qunfold-dev python==3.10
conda activate qunfold-dev
pip install -r requirements.txt
```

> :warning: remember to activate the `conda` environment every time you open a new shell.


### Install HEP dependencies
To run all the studies you need to install some HEP dependencies:

- [`ROOT`](https://root.cern/releases/release-62804/): **v6.28/04**.
- [`RooUnfold`](https://gitlab.cern.ch/RooUnfold/RooUnfold): **v3.0.0**.

These dependencies can be easily downloaded and installed in the `HEP_deps` directory of the repository by running the related scripts (they work on Ubuntu only):

```shell
./scripts/fetchROOT.sh
source HEP_deps/root/bin/thisroot.sh

./scripts/fetchRooUnfold.sh
```

> :warning: remember to source the `ROOT` framework every time you open a new shell.


### Launch analysis
To launch the full analysis you can simply run the related Python script in the `studies` directory. All the output plots will be saved into the `results` directory of the repository.

```shell
cd studies
python analysis.py
```

Pseudo-data will be generated following common distributions (normal, breit-wigner, exponential, double-peaked).

Distributions will be unfolded using `RooUnfold` by the following classical methods:

- Response Matrix Inversion (RMI)
- Iterative Bayesian Unfolding (IBU), with 4 iterations
- SVD Tikhonov unfolding (SVD), with K=2
- Bin-by-Bin unfolding (B2B)

Distributions will be unfolded using `QUnfold` by the following methods:

- Simulated annealing unfolding (lambda=0.1, num_reads=100)


## Credits
<table>
  <tr>
    <td align="center"><a href="https://justwhit3.github.io/"><img src="https://avatars.githubusercontent.com/u/48323961?v=4" width="100px;" alt=""/><br /><sub><b>Gianluca Bianco</b></sub></a></td>
    <td align="center"><a href="https://github.com/SimoneGasperini"><img src="https://avatars2.githubusercontent.com/u/71086758?s=400&v=4" width="100px;" alt=""/><br /><sub><b>Simone Gasperini</b></sub></a></td>
  </tr>
</table>
