# What is this?
Prototype code used in the paper "_Classification of Hemodynamics Scenarios from a Public Radar Dataset using a Deep Learning Approach_" by _Slapničar_, _Wang_ and _Luštrek_ (currently submitted to Sensors journal and is subject to review).

This code has a heavy research/experimental focus and as such might contain some bugs and is not completely optimized. It is more result rather than production oriented. For any questions, please contact _gasper [dot] slapnicar [at] ijs [dot] si_.

# Dataset
This code relies on a publicly available dataset described [here](https://www.nature.com/articles/s41597-020-00629-5). It contains labelled contact sensor data alongside radar data, which are all labelled with 5 possible scenarios: Valsalva maneuver, Apnea simulation, Tilt up and Tilt down movements (on a tilting table) and Resting. It totals 24h worth of signals coming from 30 subjects.

# Requirements
This work was done using two programming languages: MATLAB and Python 3.

## MATLAB
For the MATLAB part we used `MATLAB R2020a`.

## Python 3
There are two requirement files available in the `python/requirements/` directory, one created with `conda`, and the other with `pip`. `conda 4.9.2` was primarily used to create this environment from a `python 3.8.5` base environment on a linux `Ubuntu 20.04 LTS` system.

***IMPORTANT:*** This environment setup is **not** minimal, but it is an evolving setup, which will hopefully allow for re-running all the python code. We initially started development in Jupyter notebooks, however, we later ported to individual python scripts in PyCharm.

To re-create this python environment, run:
`conda create --name <env> --file <env_file.yml>`

# Directory structure
There are 3 folders in the root path: `data/`, `matlab/` and `python/`. The first should contain the data as provided by the original authors while the 2nd and 3rd contain MATLAB and Python code respectively, assuming that data is located in the first directory. Paths are always tricky and should be checked.

## MATLAB
In the matlab directory, the most important script is `matlab/existing_code_starting_point/prep_for_ML.m`. Some of the code here is modified code provided by the original paper authors [here](https://gitlab.com/sven_schellenberger/scidata_phase1).

## Python
In the python directory, the most important script is `python/src/main.py`. It contains the basic pipeline which does the experiments described in the paper. There are many (hyper)parameters to be set, so you should play around and consider those.

Many evaluation results are saved in the corresponding directories encoded with runtime timestamps, including ANN architectures, accuracies across folds and overall hyperparameter/accuracy combinations. 
