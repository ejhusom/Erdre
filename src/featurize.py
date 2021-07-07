#!/usr/bin/env python3
"""Clean up inputs and add features to data set.

Author:
    Erik Johannes Husom

Date:
    2020-09-16

"""
import os
import sys

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import OneHotEncoder
import yaml

from config import DATA_FEATURIZED_PATH, DATA_PATH, PROFILE_PATH
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

    output_columns = np.array(
            pd.read_csv(DATA_PATH / "output_columns.csv", index_col=0)
    ).reshape(-1)

    #===============================================
    # Read all data to fit one-hot encoder
    # dfs = []

    # for filepath in filepaths:
    #     df = pd.read_csv(filepath)
    #     dfs.append(df)
        
    # combined_df = pd.concat(dfs, ignore_index=True)
    # categorical_variables = find_categorical_variables()

    # print(f"Columns: {combined_df.columns}")
    # print(f"Cat: {categorical_variables}")

    # Check if some categorical variables have been removed in the cleaning
    # process, and if so, remove them from the list
    # removables = []
    # for v in categorical_variables:
    #     if v not in combined_df.columns:
    #         removables.append(v)
    #         # categorical_variables.remove(v)
    # print(removables)
    # categorical_variables.remove(removables)

    # print(f"Cat: {categorical_variables}")
    # print(combined_df[categorical_variables])
    # categorical_encoder = OneHotEncoder()
    # categorical_encoder.fit(combined_df)
    #===============================================


    for filepath in filepaths:

        # Read csv
        df = pd.read_csv(filepath, index_col=0)

        # Move target column(s) to the beginning of dataframe
        for col in output_columns[::-1]:
            df = move_column(df, column_name=col, new_idx=0)

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

# def filter_inputs_by_correlation():
#     """Filter the input features based on the correlation between the features
#     and the target variable.

#     Returns:
#         removable_variables (list): Which columns to delete from data set.

#     """

#     params_clean = yaml.safe_load(open("params.yaml"))["clean"]
#     correlation_metric = params_clean["correlation_metric"]
#     target = params_clean["target"]

#     params_featurize = yaml.safe_load(open("params.yaml"))["featurize"]
#     target_min_correlation_threshold = params["target_min_correlation_threshold"]

#     profile_json = json.load(open(PROFILE_PATH / "profile.json"))
#     messages = profile_json["messages"]
#     variables = list(profile_json["variables"].keys())
#     correlations = profile_json["correlations"]["pearson"]

#     removable_variables = []


#     for m in messages:
#         m = m.split()
#         warning = m[0]
#         variable = m[-1]

#         if warning == "[CONSTANT]":
#             removable_variables.append(variable)
#         if warning == "[ZEROS]":
#             p_zeros = profile_json["variables"][variable]["p_zeros"]
#             if p_zeros > percentage_zeros_threshold:
#                 removable_variables.append(variable)
#         if warning == "[HIGH_CORRELATION]":
#             try:
#                 correlation_scores = correlations[variables.index(variable)]
#                 for correlated_variable in correlation_scores:
#                     if (correlation_scores[correlated_variable] > input_max_correlation_threshold and
#                         variable != correlated_variable and
#                         variable not in removable_variables):

#                         removable_variables.append(correlated_variable)
#                         # print(f"{variable} is correlated with {correlated_variable}: {correlation_scores[correlated_variable]}")
#             except:
#                 # Pandas profiling might not be able to compute correlation
#                 # score for some variables, for example some categorical
#                 # variables.
#                 pass
#                 # print(f"{variable}: Could not find correlation score.")

#     removable_variables = list(set(removable_variables))

#     return removable_variables

def find_categorical_variables():
    """Find categorical variables based on profiling report.

    Returns:
        categorical_variables (list): List of categorical variables.

    """

    params = yaml.safe_load(open("params.yaml"))["clean"]
    target = params["target"]

    profile_json = json.load(open(PROFILE_PATH / "profile.json"))
    variables = list(profile_json["variables"].keys())
    correlations = profile_json["correlations"]["pearson"]

    categorical_variables = []

    for v in variables:

        try:
            n_categories = profile_json["variables"][v]["n_category"]
            categorical_variables.append(v)
        except:
            pass


    # categorical_variables = list(set(categorical_variables))

    # if target in categorical_variables:
    #     print("Warning related to target variable. Check profile for details.")
    #     categorical_variables.remove(target)

    return categorical_variables


if __name__ == "__main__":

    np.random.seed(2020)

    featurize(sys.argv[1])

