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
from preprocess_utils import read_csv, split_sequences


def sequentialize(filepaths):

    # If filepaths is a string (e.g. only one filepath), wrap this in a list
    if isinstance(filepaths, str):
        filepaths = [filepaths]

    DATA_SEQUENTIALIZED_PATH.mkdir(parents=True, exist_ok=True)

    params = yaml.safe_load(open("params.yaml"))["sequentialize"]

    hist_size = params["hist_size"]
    use_elements = params["use_elements"]
    target_mean_window = params["target_mean_window"]

    for filepath in filepaths:

        df, index = read_csv(filepath)

        # Convert to numpy
        data = df.to_numpy()

        # Split into input (X) and output/target (y)
        X = data[:, 1:].copy()
        y = data[:, 0].copy().reshape(-1, 1)

        if use_elements > 1:
            X = X[::use_elements]
            y = y[::use_elements]

        # Combine y and X to get correct format for sequentializing
        data = np.hstack((y, X))

        # Split into sequences
        X, y = split_sequences(data, hist_size, target_mean_window)

        # Save X and y into a binary file
        np.savez(
            DATA_SEQUENTIALIZED_PATH
            / (
                os.path.basename(filepath).replace(
                    "featurized.csv", "sequentialized.npz"
                )
            ),
            X=X,
            y=y,
        )


if __name__ == "__main__":

    sequentialize(sys.argv[1:])
