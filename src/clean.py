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

    # Find removable variables from profiling report
    removable_variables = parse_profile_warnings()

    dfs = []

    for filepath in filepaths:

        # Read csv
        df = pd.read_csv(filepath, index_col=0)

        for c in removable_variables:
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


def parse_profile_warnings():
    """Read profile warnings and find which columns to delete.

    Returns:
        removable_variables (list): Which columns to delete from data set.

    """

    params = yaml.safe_load(open("params.yaml"))["clean"]
    correlation_metric = params["correlation_metric"]
    target = params["target"]

    profile_json = json.load(open(PROFILE_PATH / "profile.json"))
    messages = profile_json["messages"]
    variables = list(profile_json["variables"].keys())
    correlations = profile_json["correlations"]["pearson"]

    removable_variables = []

    percentage_zeros_threshold = params["percentage_zeros_threshold"]
    correlation_threshold = params["correlation_threshold"]

    for m in messages:
        m = m.split()
        warning = m[0]
        variable = m[-1]

        if warning == "[CONSTANT]":
            removable_variables.append(variable)
        if warning == "[ZEROS]":
            p_zeros = profile_json["variables"][variable]["p_zeros"]
            if p_zeros > percentage_zeros_threshold:
                removable_variables.append(variable)
        if warning == "[HIGH_CORRELATION]":
            try:
                correlation_scores = correlations[variables.index(variable)]
                for correlated_variable in correlation_scores:
                    if (correlation_scores[correlated_variable] > correlation_threshold and
                        variable != correlated_variable and
                        variable not in removable_variables):

                        removable_variables.append(correlated_variable)
                        # print(f"{variable} is correlated with {correlated_variable}: {correlation_scores[correlated_variable]}")
            except:
                # Pandas profiling might not be able to compute correlation
                # score for some variables, for example some categorical
                # variables.
                pass
                # print(f"{variable}: Could not find correlation score.")

    removable_variables = list(set(removable_variables))

    if target in removable_variables:
        print("Warning related to target variable. Check profile for details.")
        removable_variables.remove(target)

    return removable_variables

if __name__ == "__main__":

    np.random.seed(2020)

    clean(sys.argv[1])

