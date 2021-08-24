#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Clean up data.

TODO: Remove features with high correlation

Author:
    Erik Johannes Husom

Created:
    2021-06-30

"""
import json
import os
import sys

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import LabelBinarizer, LabelEncoder

from config import DATA_CLEANED_PATH, DATA_PATH, PROFILE_PATH
from preprocess_utils import find_files


def clean(dir_path):
    """Clean up inputs.

    Args:
        dir_path (str): Path to directory containing files.

    """

    # Load parameters
    dataset = yaml.safe_load(open("params.yaml"))["profile"]["dataset"]
    params = yaml.safe_load(open("params.yaml"))
    combine_files = params["clean"]["combine_files"]
    target = params["clean"]["target"]
    classification = params["clean"]["classification"]
    onehot_encode_target = params["clean"]["onehot_encode_target"]

    # If no name of data set is given, all files present in 'assets/data/raw'
    # will be used.
    if dataset is not None:
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
        if df.iloc[:, 0].is_monotonic:
            df = df.iloc[:, 1:]

        for column in removable_variables:
            del df[column]

        df.dropna(inplace=True)

        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)

    if classification:

        if onehot_encode_target and len(np.unique(combined_df[target])) > 2:
            encoder = LabelBinarizer()
        else:
            if onehot_encode_target:
                raise ValueError(
                    "Parameter 'onehot_encode_target' is set to True, but target is binary. Change parameter to False in order to use this pipeline."
                )
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
            DATA_CLEANED_PATH / (os.path.basename("data-cleaned.csv"))
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
            df[column_name] = target_encoded[:, i]
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

    for message in messages:
        message = message.split()
        warning = message[0]
        variable = message[-1]

        if warning == "[CONSTANT]":
            removable_variables.append(variable)
            print(f"Removed variable '{variable}' because it is constant.")
        if warning == "[ZEROS]":
            p_zeros = profile_json["variables"][variable]["p_zeros"]
            if p_zeros > percentage_zeros_threshold:
                removable_variables.append(variable)
                print(
                    f"Removed variable '{variable}' because % of zeros exceeds {percentage_zeros_threshold*100}%."
                )
        if warning == "[HIGH_CORRELATION]":
            try:
                correlation_scores = correlations[variables.index(variable)]
                for correlated_variable in correlation_scores:
                    if (
                        correlation_scores[correlated_variable]
                        > input_max_correlation_threshold
                        and variable != correlated_variable
                        and variable != target
                        and correlated_variable != target
                        and variable not in removable_variables
                    ):

                        removable_variables.append(correlated_variable)
                        print(
                            f"Removed variable '{correlated_variable}' because of high correlation ({correlation_scores[correlated_variable]:.2f}) with variable '{variable}'."
                        )
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

    clean(sys.argv[1])
