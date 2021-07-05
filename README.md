# Erdre

Erroneous data repair for Industry 4.0.

This is a project using [DVC](https://dvc.org/) for setting up a flexible and
robust pipeline for machine learning experiments.

This README explains how to use the pipeline to create predictive models from
scratch:

1. [Installation](#1-installation)
2. [Setup](#2-setup)
3. [Add data](#3-add-data)
4. [Specify parameters](#4-specify-parameters)
5. [Run experiments](#5-run-experiments)

An example of results can be seen here:

- [Example of results](#example-of-results)

## 1. Installation

Developed using Python3.8. You can install the required modules by creating a
virtual environment and install the `requirements.txt`-file (run these commands
from the main folder):

```
mkdir venv
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

As an alternative you can install the required modules by running:

```
pip3 install numpy pandas pandas_profiling matplotlib tensorflow sklearn plotly pyyaml dvc nonconformist
```

To get a plot of the neural network architecture, the following software needs
to be installed: [Graphviz](https://graphviz.org/about/).

## 2. Setup

Initialize DVC by running:

```
dvc init --no-scm
```

The `--no-scm` option specifies that we do not want the DVC-related files to be
tracked by git. This is because this repository only containes the files that
consitutes the framework for creating a machine learning pipeline, and should
be agnostic to project specific things, such as data files, plots, metrics and
so on.

If you have a fork of this repository, and want to track project specific
files you have to:

1. Remove the last lines under `# project-specific files` in .gitignore, and
2. Initialize DVC with `dvc init`.

## 3. Add data

To add your data to the pipeline, perform the following steps:

1. Place the data files in the folder `assets/data/raw/[dataset]`, where
   `dataset` is your chosen name of the data set.
2. In `params.yaml`, set the parameter `dataset` to the name of your data set.

Currently only .csv-files are supported.

Example with a data set called `data01`:

```
assets/
├── data/
|   └── raw/
|       ├── data01/
|       |   ├── experiment_01.csv
|       |   ├── experiment_02.csv
|       |   ├── experiment_03.csv
|       |   ├── ...
|       |   ├── experiment_17.csv
|       |   └── experiment_18.csv
|       └─── README.md
├── metrics/
├── plots/
└── profiling/
```

The data set name is then specified in `params.yaml`:

```
profile:
  dataset: data01

...

```

It is also possible to keep your data files directly inside `assets/data/raw/`,
in which case you will need to let the `dataset` parameter in `params.yaml` be
empty. This will however limit the flexibility of easily swapping data sets
without having to move files around.


## 4. Specify parameters

Adjust the parameters in `params.yaml` to customize the stages of preprocessing
and machine learning. Documentation of the parameters is found in
[`params.yaml`](https://github.com/SINTEF-9012/Erdre/blob/master/params.yaml).

## 5. Running experiments

Run experiments by executing this command on the main directory:

```
dvc repro
```

On some systems, `dvc` cannot be run as a standalone command, in which case you
will need to run:

```
python3 -m dvc repro
```


To run single stages of the pipeline, run:
```
dvc repro [STAGE NAME]
```

For example:

```
dvc repro profile     # will only run the profiling stage
dvc repro featurize   # will only run the featurize stage
```


## Example of results

When the trained model is evaluated, the program produces two plots that
visualizes the predictions on the test set. These plots are placed on the
folder `assets/plots/`, and show the following data:

- `prediction.html`: The true target values compared to the predicted target
  values. Only the first value of each target sequence is used in the plot, and
  all of this values are connected and plotted as a line. The features used in
  the model are also plotted.

![Example of prediction.](img/prediction_example.png)

- `prediction_sequences.html`: Individual predicted target sequences are
  plotted against the true values. The predicted sequences are shown in
  different colors to easily distinguish between them. Only a subset of the
  predicted target sequences are shown, in order to avoid overlapping and make
  the plot easier to interpret.

![Example of sequence predictions.](img/prediction_sequences_example.png)

