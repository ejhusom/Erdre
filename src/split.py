#!/usr/bin/env python3
"""Split data into training and test set.

Author:
    Erik Johannes Husom

Date:
    2021-02-24

"""
import os
import random
import shutil
import sys

import numpy as np
import pandas as pd
import yaml

from config import DATA_SPLIT_PATH
from preprocess_utils import find_files


def split(dir_path):
    """Split data into train and test set.

    Training files and test files are saved to different folders.

    Args:
        dir_path (str): Path to directory containing files.

    """

    params = yaml.safe_load(open("params.yaml"))["split"]
    shuffle_files = params["shuffle_files"]

    DATA_SPLIT_PATH.mkdir(parents=True, exist_ok=True)

    filepaths = find_files(dir_path, file_extension=".npy")

    # Handle special case where there is only one data file.
    if isinstance(filepaths, str) or len(filepaths) == 1:
        filepath = filepaths[0]

        data = np.load(filepath)

        train_size = int(len(data) * params["train_split"])

        # This is used when using conformal predictors.
        # It specifies the calibration set size.
        # Set to 0 in params.yml if no calibration is to be done.
        calibrate_size = int(len(data) * params["calibrate_split"])

        data_train = None
        data_test = None
        data_calibrate = None

        if params["calibrate_split"] == 0:
            data_train = data[:train_size, :]
            data_test = data[train_size:, :]
        else:
            data_train = data[:train_size, :]
            data_calibrate = data[train_size : train_size + calibrate_size, :]
            data_test = data[train_size + calibrate_size :, :]

        np.save(
            DATA_SPLIT_PATH / os.path.basename(filepath).replace("featurized", "train"),
            data_train,
        )

        np.save(
            DATA_SPLIT_PATH / os.path.basename(filepath).replace("featurized", "test"),
            data_test,
        )

        if params["calibrate_split"] != 0:
            np.save(
                DATA_SPLIT_PATH
                / os.path.basename(filepath).replace("featurized", "calibrate"),
                data_calibrate,
            )

    else:

        if shuffle_files:
            random.shuffle(filepaths)

        # Parameter 'train_split' is used to find out no. of files in training set
        file_split = int(len(filepaths) * params["train_split"])
        file_split_calibrate = int(len(filepaths) * params["calibrate_split"])

        training_files = []
        test_files = []
        calibrate_files = []

        if file_split_calibrate == 0:
            training_files = filepaths[:file_split]
            test_files = filepaths[file_split:]
        else:
            training_files = filepaths[:file_split]
            calibrate_files = filepaths[file_split : file_split + file_split_calibrate]
            test_files = filepaths[file_split + file_split_calibrate :]

        for filepath in filepaths:

            if filepath in training_files:
                shutil.copyfile(
                    filepath,
                    DATA_SPLIT_PATH
                    / os.path.basename(filepath).replace("featurized", "train"),
                )

            elif filepath in test_files:
                shutil.copyfile(
                    filepath,
                    DATA_SPLIT_PATH
                    / os.path.basename(filepath).replace("featurized", "test"),
                )
            elif filepath in calibrate_files:
                shutil.copyfile(
                    filepath,
                    DATA_SPLIT_PATH
                    / os.path.basename(filepath).replace("featurized", "calibrate"),
                )


if __name__ == "__main__":

    np.random.seed(2029)

    split(sys.argv[1])
