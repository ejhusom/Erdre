#!/usr/bin/env python3
"""Scaling the inputs of the data set.

Possible scaling methods

TODO:
    Implement scaling.


Author:   
    Erik Johannes Husom

Created:  
    2020-09-16

"""
import sys

import numpy as np
import yaml

from config import DATA_SCALED_PATH
from preprocess_utils import scale_data


def scale(train_file, test_file):
    """Scale train and test set.

    Args:
        train_file (str): Path to file containing train set.
        test_file (str): Path to file containing test set.

    """

    DATA_SCALED_PATH.mkdir(parents=True, exist_ok=True)

    params = yaml.safe_load(open("params.yaml"))["scale"]

    # Load training and test files
    train = np.load(train_file)
    test = np.load(test_file)

    X_train = train["X"]
    X_test = test["X"]
    y_train = train["y"]
    y_test = test["y"]

    # X_train, X_test = scale_data(X_train, X_test, scaler_type=params["method"])

    np.savez(DATA_SCALED_PATH / "train.npz", X=X_train, y=y_train)
    np.savez(DATA_SCALED_PATH / "test.npz", X=X_test, y=y_test)


if __name__ == "__main__":

    if len(sys.argv) == 3:
        train_file = sys.argv[1]
        test_file = sys.argv[2]
        scale(train_file, test_file)
    else:
        raise ValueError("Give 2 cmd line args: [train_file] [test_file]")
