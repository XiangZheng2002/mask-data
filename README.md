# Learning to Control PDEs with Differentiable Physics

Code for the [ICLR 2020 paper](https://ge.in.tum.de/publications/2020-iclr-holl/).

*Authors:* Philipp Holl, Vladlen Koltun, Nils Thuerey.

This project is based on the differentiable simulation framework [Φ<sub>Flow</sub>](https://github.com/tum-pbs/PhiFlow).

## Installation

To run the code, you will need Python 2.7, 3.6 or newer.

Clone this repository including Φ<sub>Flow</sub> by running the following command.

```bash
$ git clone --recursive https://github.com/holl-/PDE-Control.git
```

Install the dependencies by executing the below command inside the root directory of your cloned repository.

```bash
$ pip install PhiFlow/[gui] jupyterlab
```


## Reproducing the experiments

This repository contains a revised version of the code used in the ICLR paper, now based on Φ<sub>Flow</sub> 1.0.
The new version features well-documented Jupyter notebooks to reproduce the experiments.

The original code is based on an old version of Φ<sub>Flow</sub> and will be available in `/legacy` before the ICLR 2020 conference.

**Experiment 1: Burgers equation**

Launch Jupyter notebook by executing the following command in the root directory of the repository.
```bash
$ jupyter notebook
```

In the browser, navigate into the `notebooks` directory and open `Control Burgers.ipynb`.

**Experiment 2: Shape transition**

Coming soon.