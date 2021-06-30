#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Clean up data.

TODO: Remove features with high correlation

Author:   
    Erik Johannes Husom

Created:  
    2021-06-30

"""
import os
import sys

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import yaml

from config import DATA_CLEANED_PATH, DATA_PATH, PROFILE_PATH
from preprocess_utils import move_column, find_files
from profile import profile

def clean(dir_path):
    """Clean up inputs.

    Args:
        dir_path (str): Path to directory containing files.

    """

    # Load parameters
    dataset = yaml.safe_load(open("params.yaml"))["profile"]["dataset"]
    """Name of data set, which must be the name of subfolder of
    'assets/data/raw', in where to look for data.""" 

    combine_files = yaml.safe_load(open("params.yaml"))["clean"]["combine_files"]

    # If no name of data set is given, all files present in 'assets/data/raw'
    # will be used.
    if dataset != None:
        dir_path += "/" + dataset

    filepaths = find_files(dir_path, file_extension=".csv")

    DATA_CLEANED_PATH.mkdir(parents=True, exist_ok=True)

    dfs = []

    for filepath in filepaths:

        # Read csv
        df = pd.read_csv(filepath)

        removable_columns = parse_profile_warnings()

        for c in removable_columns:
            del df[c]
        
        df.dropna(inplace=True)

        dfs.append(df)
        
    combined_df = pd.concat(dfs, ignore_index=True)

    if combine_files:
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

    # Save list of features used
    pd.DataFrame(df.columns).to_csv(DATA_PATH / "input_columns.csv")

def parse_profile_warnings():
    """Read profile warnings and find which columns to delete.

    Returns:
        removable_columns (list): Which columns to delete from data set.

    """

    profile_json = json.load(open(PROFILE_PATH / "profile.json"))
    messages = profile_json["messages"]

    removable_columns = []

    p_zeros_threshold = 0.5

    for m in messages:
        m = m.split()

        if m[0] == "[CONSTANT]":
            removable_columns.append(m[-1])
        if m[0] == "[ZEROS]":
            p_zeros = profile_json["variables"][m[-1]]["p_zeros"]
            if p_zeros > p_zeros_threshold:
                removable_columns.append(m[-1])

    return removable_columns

if __name__ == "__main__":

    np.random.seed(2020)

    clean(sys.argv[1])

    # parse_profile_warnings()


