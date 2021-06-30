#!/usr/bin/env python3
"""Scaling the inputs of the data set.

Possible scaling methods

TODO:
    Implement scaling when there is only one workout file.

Author:   
    Erik Johannes Husom

Created:  
    2020-09-16

"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import yaml

from config import DATA_SCALED_PATH
from preprocess_utils import find_files, scale_data

def scale(dir_path):
    """Scale training and test data.

    Args:
        dir_path (str): Path to directory containing files.

    """

    filepaths = find_files(dir_path, file_extension=".csv")

    DATA_SCALED_PATH.mkdir(parents=True, exist_ok=True)

    params = yaml.safe_load(open("params.yaml"))["scale"]
    input_method = params["input"]
    output_method = params["output"]
    
    if input_method == "standard":
        scaler = StandardScaler()
    elif input_method == "minmax":
        scaler = MinMaxScaler()
    elif input_method == "robust":
        scaler = RobustScaler()
    elif input_method == "none":
        scaler = StandardScaler()
    else:
        raise NotImplementedError(f"{input_method} not implemented.")

    if output_method == "standard":
        output_scaler = StandardScaler()
    elif output_method == "minmax":
        output_scaler = MinMaxScaler()
    elif output_method == "robust":
        output_scaler = RobustScaler()
    elif output_method == "none":
        output_scaler = StandardScaler()
    else:
        raise NotImplementedError(f"{output_method} not implemented.")

    train_inputs = []
    train_outputs = []

    data_overview = {}

    for filepath in filepaths:

        df = pd.read_csv(filepath, index_col=0)
        
        # Convert to numpy
        data = df.to_numpy()

        # Split into input (X) and output/target (y)
        X = data[:, 1:].copy()
        y = data[:, 0].copy().reshape(-1, 1)


        if "train" in filepath:
            train_inputs.append(X)
            train_outputs.append(y)
            category = "train"
        elif "test" in filepath:
            category = "test"
        elif "calibrate" in filepath:
            category = "calibrate"
            
        data_overview[filepath] = {"X": X, "y": y, "category": category}

    X_train = np.concatenate(train_inputs)
    y_train = np.concatenate(train_outputs)

    # Fit a scaler to the training data
    scaler = scaler.fit(X_train)
    output_scaler = output_scaler.fit(y_train)

    for filepath in data_overview:

        # Scale inputs
        if input_method == "none":
            X=data_overview[filepath]["X"]
        else:
            X = scaler.transform(data_overview[filepath]["X"])

        # Scale outputs
        if output_method == "none":
            y = data_overview[filepath]["y"]
        else:
            y = output_scaler.transform(data_overview[filepath]["y"])

        # Save X and y into a binary file
        np.savez(
            DATA_SCALED_PATH
            / (
                os.path.basename(filepath).replace(
                    data_overview[filepath]["category"] + ".csv", 
                    data_overview[filepath]["category"] + "-scaled.npz"
                )
            ),
            #X=data_overview[filepath]["X"],
            X = X, 
            # y = data_overview[filepath]["y"]
            y = y
        )

if __name__ == "__main__":

    np.random.seed(2020)

    scale(sys.argv[1])

