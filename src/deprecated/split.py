#!/usr/bin/env python3
"""Split data into training and test set.

This module combines data from multiple workouts, and the splits the data into
a training and test set.

Author:
    Erik Johannes Husom

Date:
    2020-09-16

"""
import sys

import numpy as np
import yaml

from config import DATA_SPLIT_PATH


def combine(filepaths):
    """Combine data from multiple workouts into one dataset.

    Args:
        filepaths (list of str): A list of paths to files containing
            sequentialized data.

    Returns:
        X (array): Input array.
        y ( array): Output/target array.

    """

    # If filepaths is a string (e.g. only one filepath), wrap this in a list
    if isinstance(filepaths, str):
        filepaths = [filepaths]

    inputs = []
    outputs = []

    for filepath in filepaths:
        infile = np.load(filepath)

        inputs.append(infile["X"])
        outputs.append(infile["y"])

    X = np.concatenate(inputs)
    y = np.concatenate(outputs)

    return X, y


def split(X, y):
    """Split data into train and test set.

    Args:
        X (array): Input array.
        y (array): Output/target array.

    """

    DATA_SPLIT_PATH.mkdir(parents=True, exist_ok=True)

    params = yaml.safe_load(open("params.yaml"))["split"]
    train_split = params["train_split"]
    train_elements = int(X.shape[0] * train_split)

    # Split X and y into train and test
    X_train, X_test = np.split(X, [train_elements])
    y_train, y_test = np.split(y, [train_elements])

    # Save train and test data into a binary file
    np.savez(DATA_SPLIT_PATH / "train.npz", X=X_train, y=y_train)
    np.savez(DATA_SPLIT_PATH / "test.npz", X=X_test, y=y_test)


if __name__ == "__main__":

    X, y = combine(sys.argv[1:])
    split(X, y)
