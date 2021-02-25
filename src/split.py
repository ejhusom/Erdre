#!/usr/bin/env python3
"""Split data into training and test set.

Author:
    Erik Johannes Husom

Date:
    2021-02-24

"""
import os
import sys

import numpy as np
import pandas as pd
import yaml

from config import DATA_SPLIT_PATH
from preprocess_utils import read_csv


def split(dir_path):
    """Split data into train and test set.

    Training files and test files are saved to different folders.

    Args:
        dir_path (str): Path to directory containing files.

    """

    filepaths = []

    for f in os.listdir(dir_path):
        if f.endswith(".csv"):
            filepaths.append(dir_path + "/" + f)

    # Handle special case where there is only one workout file.
    if isinstance(filepaths, str) or len(filepaths) == 1:
        raise NotImplementedError("Cannot handle only one input file.")

    DATA_SPLIT_PATH.mkdir(parents=True, exist_ok=True)

    params = yaml.safe_load(open("params.yaml"))["split"]

    # Parameter 'train_split' is used to find out no. of files in training set
    file_split = int(len(filepaths) * params["train_split"])

    training_files = filepaths[:file_split]
    test_files = filepaths[file_split:]


    for filepath in filepaths:

        df = pd.read_csv(filepath, index_col=0)

        if filepath in training_files:
            df.to_csv(
                DATA_SPLIT_PATH
                / (os.path.basename(filepath).replace("featurized", "train"))
            )
        elif filepath in test_files:
            df.to_csv(
                DATA_SPLIT_PATH
                / (os.path.basename(filepath).replace("featurized", "test"))
            )


if __name__ == "__main__":

    np.random.seed(2020)

    split(sys.argv[1])
