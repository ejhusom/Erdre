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
import pandas as pd
import yaml

from config import DATA_PATH, DATA_SEQUENTIALIZED_PATH
from preprocess_utils import flatten_sequentialized, read_csv
from preprocess_utils import find_files, split_sequences


def sequentialize(dir_path):

    filepaths = find_files(dir_path, file_extension=".npz")

    DATA_SEQUENTIALIZED_PATH.mkdir(parents=True, exist_ok=True)

    params = yaml.safe_load(open("params.yaml"))["sequentialize"]
    net = yaml.safe_load(open("params.yaml"))["train"]["net"]
    classification = yaml.safe_load(open("params.yaml"))["clean"]["classification"]

    hist_size = params["hist_size"]

    if classification:
        target_size = 1
    else:
        target_size = params["target_size"]

    if target_size > hist_size:
        raise ValueError("target_size cannot be larger than hist_size.")

    output_columns = np.array(
            pd.read_csv(DATA_PATH / "output_columns.csv", index_col=0)
    ).reshape(-1)

    n_output_cols = len(output_columns)

    for filepath in filepaths:

        infile = np.load(filepath)

        X = infile["X"]
        y = infile["y"]

        # Combine y and X to get correct format for sequentializing
        data = np.hstack((y, X))

        # Split into sequences
        X, y = split_sequences(data, hist_size, target_size=target_size,
                n_target_columns=n_output_cols)

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
