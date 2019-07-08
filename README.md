# Detecting breast cancer metastases _by OWKIN_

## Getting started

This repository can be seen in two ways: 

### Experiment with the dataset

This repository is a python library of utilitary functions 
allowing the loading and manipulation of the dataset provided
by OWKIN on [https://challengedata.ens.fr](https://challengedata.ens.fr/participants/challenges/18/).

Main functionalities include:

* the loading of the training dataset with its weak labels; 
* the loading of the annotated tiles training dataset with its 
labels;
* the loading of the public test dataset
* the generation of a prediction `csv` file as expected by the 
submission form of [https://challengedata.ens.fr](https://challengedata.ens.fr/participants/challenges/18/submit)

A set of [jupyter](https://jupyter.org/) notebooks are provided
in the [`notebooks/`](./notebooks/) folder; each of them 
containing an experiment.

### Command line tool

From the repository directory, you can launch the command line 
tool by typing the following command:

```bash
python -m owkin
```
