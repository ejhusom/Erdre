#!/usr/bin/env python3
"""Clean up inputs and add features to data set.

Author:
    Erik Johannes Husom

Date:
    2020-09-16

"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from config import DATA_FEATURIZED_PATH, DATA_PATH
from preprocess_utils import move_column


def featurize(dir_path):
    """Clean up inputs and add features to data set.

    Args:
        dir_path (str): Path to directory containing files.

    """

    # Load parameters
    params = yaml.safe_load(open("params.yaml"))["featurize"]

    dataset = params["dataset"]
    """Name of data set, which must be the name of subfolder of
    'assets/data/raw', in where to look for data. If no name of data set is
    given, all files present in 'assets/data/raw' will be used."""

    if dataset != None:
        dir_path += "/" + dataset

    filepaths = []

    for f in os.listdir(dir_path):
        if f.endswith(".csv"):
            filepaths.append(dir_path + "/" + f)

    DATA_FEATURIZED_PATH.mkdir(parents=True, exist_ok=True)

    if len(filepaths) == 0:
        raise ValueError(f"Could not find any data files in {dir_path}.")

    features = params["features"]
    """Features to include in data set."""

    target = params["target"]
    """Variable to use as target."""

    for filepath in filepaths:

        # Read csv
        df = pd.read_csv(filepath)

        # Move target column to the beginning of dataframe
        df = move_column(df, column_name=target, new_idx=0)

        # Check if wanted features from params.yaml exists in the data
        for feature in features:
            if feature not in df.columns:
                print(f"Feature {feature} not found!")

        # TODO: Engineer features. At the moment no engineered features exists!
        df = add_features(df, features)

        # Remove feature from input. This is useful in the case that a raw
        # feature is used to engineer a feature, but the raw feature itself
        # should not be a part of the input.
        for col in df.columns:
            if col not in features and col != target:
                del df[col]

        df.dropna(inplace=True)

        # Save data
        df.to_csv(
            DATA_FEATURIZED_PATH
            / (os.path.basename(filepath).replace(".", "-featurized."))
        )

    # Save list of features used
    pd.DataFrame(df.columns).to_csv(DATA_PATH / "input_columns.csv")

def add_features(df, features):
    """
    This function adds features to the input data, based on the arguments
    given in the features-list.

    Args:
    df (pandas DataFrame): Data frame to add features to.
    features (list): A list containing keywords specifying which features to
        add.

    Returns:
        df (pandas DataFrame): Data frame with added features.

    """

    # Stop function of features is not a list
    if features == None:
        return 0

    # TODO: Add som engineered features
    # if "frequency" in features:
        # df["frequency"] = 0

    return df 

if __name__ == "__main__":

    np.random.seed(2020)

    featurize(sys.argv[1])
