#!/usr/bin/env python3
"""Split data into sequences.

Prepare the data for input to a neural network. A sequence with a given history
size is extracted from the input data, and matched with the appropriate target
value(s).

Author:
    Erik Johannes Husom

Date:
    2020-09-16

"""
import os
import sys

import numpy as np
import yaml

from config import DATA_SEQUENTIALIZED_PATH
from preprocess_utils import flatten_sequentialized, read_csv
from preprocess_utils import split_sequences


def sequentialize(dir_path):

    filepaths = []

    for f in os.listdir(dir_path):
        if f.endswith(".npz"):
            filepaths.append(dir_path + "/" + f)

    DATA_SEQUENTIALIZED_PATH.mkdir(parents=True, exist_ok=True)

    params = yaml.safe_load(open("params.yaml"))["sequentialize"]
    net = yaml.safe_load(open("params.yaml"))["train"]["net"]

    hist_size = params["hist_size"]
    target_size = params["target_size"]

    if target_size > hist_size:
        raise ValueError("target_size cannot be larger than hist_size.")

    for filepath in filepaths:

        infile = np.load(filepath)

        X = infile["X"]
        y = infile["y"]

        # Combine y and X to get correct format for sequentializing
        data = np.hstack((y, X))

        # Split into sequences
        X, y = split_sequences(data, hist_size, target_size=target_size)

        if net == "dnn":
            X = flatten_sequentialized(X)

        # Save X and y into a binary file
        np.savez(
            DATA_SEQUENTIALIZED_PATH
            / (
                os.path.basename(filepath).replace(
                    "scaled.csv", "sequentialized.npz"
                )
            ),
            X=X,
            y=y,
        )


if __name__ == "__main__":

    np.random.seed(2020)

    sequentialize(sys.argv[1])
