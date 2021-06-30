#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Clean up data.

Author:   
    Erik Johannes Husom

Created:  
    2021-06-30

"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import yaml

from config import DATA_CLEANED_PATH, DATA_PATH
from preprocess_utils import move_column, find_files

def clean(dir_path):
    """Clean up inputs.

    Args:
        dir_path (str): Path to directory containing files.

    """

    # Load parameters
    params = yaml.safe_load(open("params.yaml"))["clean"]

    dataset = params["dataset"]
    """Name of data set, which must be the name of subfolder of
    'assets/data/raw', in where to look for data.""" 

    combine_files = params["combine_files"]

    # target = params["target"]
    # """Variable to use as target."""

    # If no name of data set is given, all files present in 'assets/data/raw'
    # will be used.
    if dataset != None:
        dir_path += "/" + dataset

    filepaths = find_files(dir_path, file_extension=".csv")

    if len(filepaths) == 0:
        raise ValueError(f"Could not find any data files in {dir_path}.")

    DATA_CLEANED_PATH.mkdir(parents=True, exist_ok=True)

    combined_df = pd.DataFrame()
    dfs = []

    for filepath in filepaths:

        # Read csv
        df = pd.read_csv(filepath)

        # number_of_nans_per_row = df.isnull().sum(axis=0)
        # number_of_samples
        # print(filepath)
        # print(number_of_nans_per_row)
        # print(number_of_nans_per_row[2])
        
        df.dropna(inplace=True)

        dfs.append(df)

    if combine_files:
        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df.to_csv(
            DATA_CLEANED_PATH
            / (os.path.basename(dataset + "-cleaned.csv"))
        )
    else:
        for filepath, df in zip(filepaths, dfs):
            df.to_csv(
                DATA_CLEANED_PATH
                / (os.path.basename(filepath).replace(".", "-cleaned."))
            )



if __name__ == "__main__":

    np.random.seed(2020)

    clean(sys.argv[1])


