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
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.utils import shuffle
import yaml

from config import DATA_CLEANED_PATH, DATA_PATH, PROFILE_PATH
from preprocess_utils import move_column, find_files
from profiling import profile

def clean(dir_path):
    """Clean up inputs.

    Args:
        dir_path (str): Path to directory containing files.

    """

    # Load parameters
    dataset = yaml.safe_load(open("params.yaml"))["profile"]["dataset"]
    """Name of data set, which must be the name of subfolder of
    'assets/data/raw', in where to look for data.""" 

    params = yaml.safe_load(open("params.yaml"))["clean"]
    combine_files = params["combine_files"]
    target = params["target"]
    classification = params["classification"]
    onehot_encode_target = params["onehot_encode_target"]

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
        df = pd.read_csv(filepath)

        # If the first column is an index column, remove it.
        if df.iloc[:,0].is_monotonic:
            df = df.iloc[:,1:]

        # for c in removable_variables:
        #     del df[c]
        
        df.dropna(inplace=True)

        dfs.append(df)
        
    combined_df = pd.concat(dfs, ignore_index=True)

    if classification:

        if onehot_encode_target:
            encoder = LabelBinarizer()
        else:
            encoder = LabelEncoder()

        target_col = np.array(combined_df[target]).reshape(-1)
        encoder.fit(target_col)
        print(f"Classes: {encoder.classes_}")
        print(f"Encoded classes: {encoder.transform(encoder.classes_)}")

        combined_df, output_columns = encode_target(encoder, combined_df, target)

        for i in range(len(dfs)):
            dfs[i], _ = encode_target(encoder, dfs[i], target)

    else:
        output_columns = [target]

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

    pd.DataFrame(output_columns).to_csv(DATA_PATH / "output_columns.csv")

def encode_target(encoder, df, target):
    """Encode a target variable based on a fitted encoder.

    Args:
        encoder: A fitted encoder.
        df (DataFrame): DataFrame containing the target variable.
        target (str): Name of the target variable.

    Returns:
        df (DataFrame): DataFrame with the original target variable removed,
            substituted by a onehot encoding of the variable.
        output_columns (list): List of the names of the target columns.

    """

    output_columns = []

    target_col = np.array(df[target]).reshape(-1)
    target_encoded = encoder.transform(target_col)

    del df[target]
    
    if len(target_encoded.shape) > 1:
        for i in range(target_encoded.shape[-1]):
            column_name = f"{target}_{i}"
            df[column_name] = target_encoded[:,i]
            output_columns.append(column_name)
    else:
        df[target] = target_encoded
        output_columns.append(target)

    return df, output_columns


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
    input_max_correlation_threshold = params["input_max_correlation_threshold"]

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
                    if (correlation_scores[correlated_variable] > input_max_correlation_threshold and
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

