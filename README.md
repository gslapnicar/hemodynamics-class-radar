# What is this?
Prototype code used in the paper "Classification of Hemodynamics Scenarios from a Public RadarDataset using a Deep Learning Approach" by Slapničar, Wang and Luštrek (currently submitted to Sensors journal and is subject to review).

This code has a heavy research/experimental focus and as such might contain some bugs and is not completely optimized. It is more result rather than production oriented.

# Dataset
This code relies on a publicly available dataset described [here](https://www.nature.com/articles/s41597-020-00629-5). It contains labelled contact sensor data alongside radar data, which are all labelled with 5 possible scenarios: Valsalva maneuver, Apnea simulation, Tilt up and Tilt down movements (on a tilting table) and Resting. It totals 24h worth of signals coming from 30 subjects.

# Requirements
This work was done using two programming languages: MATLAB and Python 3.

## MATLAB
For the MATLAB part we used `MATLAB R2020a`.

## Python 3
There are two requirement files available in the `python/` directory, one created with `conda`, and the other with `pip`. `conda 4.9.2` was primarily used to create this environment from a `python 3.8.5` base environment on a linux `Ubuntu 20.04 LTS` system.

***IMPORTANT:*** This environment setup is **not** minimal, but it is an evolving setup, which will hopefully allow for re-running all the python code. We initially started development in Jupyter notebooks, however, we later ported to individual python scripts in Pycharm.

To re-create this python environment, run:
`conda create --name <env> --file <env_file>`

# Directory structure
