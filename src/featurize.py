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
from pandas.api.types import is_numeric_dtype
import yaml

from config import DATA_FEATURIZED_PATH, DATA_PATH
from preprocess_utils import move_column, find_files


def featurize(dir_path):
    """Clean up inputs and add features to data set.

    Args:
        dir_path (str): Path to directory containing files.

    """

    # Load parameters
    params = yaml.safe_load(open("params.yaml"))["featurize"]

    features = params["features"]
    """Features to include in data set."""

    target = yaml.safe_load(open("params.yaml"))["clean"]["target"]
    """Variable to use as target."""

    filepaths = find_files(dir_path, file_extension=".csv")

    DATA_FEATURIZED_PATH.mkdir(parents=True, exist_ok=True)

    for filepath in filepaths:

        # Read csv
        df = pd.read_csv(filepath)

        # Move target column to the beginning of dataframe
        df = move_column(df, column_name=target, new_idx=0)

        # If no features are specified, use all columns as features
        # TODO: Maybe not the most robust way to test this
        if type(params["features"]) != list:  
            features = df.columns

        # Check if wanted features from params.yaml exists in the data
        for feature in features:
            if feature not in df.columns:
                print(f"Feature {feature} not found!")

        # TODO: Engineer features. At the moment no engineered features exists!
        df = add_features(df, features)

        for col in df.columns:
            # Remove feature from input. This is useful in the case that a raw
            # feature is used to engineer a feature, but the raw feature itself
            # should not be a part of the input.
            if col not in features and col != target:
                del df[col]
            
            # Remove feature if it is non-numeric
            elif not is_numeric_dtype(df[col]):
                del df[col]

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
    if type(features) != list:
        return df

    # TODO: Add som engineered features
    # if "frequency" in features:
        # df["frequency"] = 0

    return df 

if __name__ == "__main__":

    np.random.seed(2020)

    featurize(sys.argv[1])
