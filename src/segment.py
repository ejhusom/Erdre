#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Segmentation of time-series data.

Author:   
    Erik Johannes Husom

Created:  
    2021-07-07

"""
import sys

import numpy as np
import pandas as pd
import yaml
from tsfresh import extract_relevant_features
from tsfresh.utilities.dataframe_functions import roll_time_series

from config import DATA_PATH
from preprocess_utils import find_files, move_column


def segment(dir_path):
    """Create segments of time series."""

    target = yaml.safe_load(open("params.yaml"))["clean"]["target"]

    filepaths = find_files(dir_path, file_extension=".csv")

    output_columns = np.array(
        pd.read_csv(DATA_PATH / OUTPUT_FEATURES_PATH, index_col=0)
    ).reshape(-1)

    dfs = []

    for filepath in filepaths:
        df = pd.read_csv(filepath, index_col=0)
        # df = df.iloc[10000:90000,:]
        # df = df.iloc[:,:-1]
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df = combined_df[::10]
    print(combined_df)

    n_rows = len(combined_df)
    segment_size = 100
    n_segments = int(n_rows / segment_size)
    ids = np.arange(1, n_segments + 1, 1)

    idlist = np.ones(segment_size)

    for i in ids[1:]:
        idlist = np.concatenate((idlist, np.ones(segment_size) * i))

    idlist = np.array(idlist, dtype=np.int32)

    # combined_df = combined_df.iloc[:len(idlist),:]
    # combined_df["id"] = idlist
    combined_df["id"] = np.ones(n_rows)

    # y = []

    # for i in ids:
    #     target_value = combined_df[combined_df["id"] == i][target].iloc[-1]
    #     y.append(target_value)

    # y = pd.Series(y)
    # y.index = y.index + 1
    # combined_df.index.name = "index"
    # print(y)
    print(combined_df)
    # print(np.unique(y))

    df_rolled = roll_time_series(combined_df, column_id="id", column_sort=None)
    print(df_rolled)

    # features_filtered_direct = extract_relevant_features(
    #         combined_df,
    #         y,
    #         column_id='id',
    #         # column_sort='index'
    # )

    # print(features_filtered_direct)


def try_cesium(df):

    from cesium import featurize

    features_to_use = [
        "amplitude",
        "percent_beyond_1_std",
        "maximum",
        "max_slope",
        "median",
        "median_absolute_deviation",
        "percent_close_to_median",
        "minimum",
        "skew",
        "std",
        "weighted_average",
    ]
    fset_cesium = featurize.featurize_time_series(
        times=eeg["times"],
        values=eeg["measurements"],
        errors=None,
        features_to_use=features_to_use,
    )
    print(fset_cesium.head())


if __name__ == "__main__":

    segment(sys.argv[1])
