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
from tsfresh import extract_relevant_features
import yaml

from config import DATA_PATH
from preprocess_utils import move_column, find_files


def segment(dir_path):
    """Create segments of time series.

    """

    target = yaml.safe_load(open("params.yaml"))["clean"]["target"]

    filepaths = find_files(dir_path, file_extension=".csv")

    output_columns = np.array(
            pd.read_csv(DATA_PATH / "output_columns.csv", index_col=0)
    ).reshape(-1)

    dfs = []

    for filepath in filepaths:
        df = pd.read_csv(filepath, index_col=0)
        # df = df.iloc[10000:90000,:]
        df = df.iloc[:,:-1]
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)

    n_rows = len(combined_df)
    segment_size = 100
    n_segments = int(n_rows / segment_size)
    ids = np.arange(1, n_segments+1, 1)

    idlist = np.ones(segment_size)

    for i in ids[1:]:
        idlist = np.concatenate((idlist, np.ones(segment_size)*i))

    idlist = np.array(idlist, dtype=np.int32)

    combined_df = combined_df.iloc[:len(idlist),:]
    combined_df["id"] = idlist

    y = []

    for i in ids:
        target_value = combined_df[combined_df["id"] == i][target].iloc[-1]
        y.append(target_value)

    y = pd.Series(y)
    y.index = y.index + 1
    # combined_df.index.name = "index"
    print(y)
    print(combined_df)
    print(np.unique(y))

    features_filtered_direct = extract_relevant_features(
            combined_df,
            y,
            column_id='id',
            # column_sort='index'
    )

    print(features_filtered_direct)


if __name__ == '__main__':

    segment(sys.argv[1])
