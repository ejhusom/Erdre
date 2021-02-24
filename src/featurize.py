#!/usr/bin/env python3
"""Clean up inputs and add features to data set.

Author:
    Erik Johannes Husom

Date:
    2020-09-16

"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import yaml

from config import DATA_FEATURIZED_PATH, DATA_PATH
from preprocess_utils import read_csv, move_column


def featurize(filepaths):
    """Clean up inputs and add features to data set.

    Args:
        filepaths (list of str): List of paths to files to process.

    """

    # If filepaths is a string (e.g. only one filepath), wrap this in a list
    if isinstance(filepaths, str):
        filepaths = [filepaths]

    DATA_FEATURIZED_PATH.mkdir(parents=True, exist_ok=True)

    # Load parameters
    params = yaml.safe_load(open("params.yaml"))["featurize"]

    features = params["features"]
    """Features to be engineered."""

    delete_features = params["delete"]
    """Features to be deleted before adding new features."""

    remove_features = params["remove"]
    """Features to be removed after adding new features, since they are used in
    feature engineering.
    """

    scale = params["scale"]
    """Whether to scale the input features before feature engineering."""

    diff_targets = params["diff_targets"]
    """Whether to use the change in power as target, as opposed to the power
    value itself.
    """

    for filepath in filepaths:

        # Read csv, and delete specified columns
        df, index = read_csv(
            filepath,
            delete_columns=delete_features
        )

        # Move target column to the beginning of dataframe
        df = move_column(df, column_name="power", new_idx=0)

        if scale:
            df = scale_inputs(df)

        add_features(df, features, 
                range_window=params["range_window"],
                range_smoothing=params["range_smoothing"],
                slope_shift=params["slope_shift"],
                slope_smoothing=params["slope_smoothing"],
        )

        # Remove columns from input. Check first if it is a list, to avoid
        # error if empty.
        if isinstance(remove_features, list):
            for col in remove_features:
                del df[col]

        if diff_targets > 0:
            df["power"] = df["power"].diff(diff_targets)
            df["power"].fillna(0, inplace=True)

        # Save data
        df.to_csv(
            DATA_FEATURIZED_PATH
            / (os.path.basename(filepath).replace("restructured", "featurized"))
        )

    # Save list of features used
    pd.DataFrame(df.columns).to_csv(DATA_PATH / "input_columns.csv")

def scale_inputs(df):
    """Scale input features.

    Args:
        df (DataFrame): Data frame containing data.

    Returns:
        scaled_df (DataFrame): Data frame containing scaled data.

    """

    # Load scaling parameters
    params = yaml.safe_load(open("params.yaml"))["featurize"]
    heartrate_min = params["heartrate_min"]
    heartrate_max = params["heartrate_max"]
    breathing_min = params["breathing_min"]
    breathing_max = params["breathing_max"]
    
    heartrate_range = heartrate_max - heartrate_min
    breathing_range = breathing_max - breathing_min

    if "heartrate" in df.columns:
        df["heartrate"] = (df["heartrate"] - heartrate_min)/heartrate_range

    if "ribcage" in df.columns:
        df["ribcage"] = (df["ribcage"] - breathing_min)/breathing_range

    if "abdomen" in df.columns:
        df["abdomen"] = (df["abdomen"] - breathing_min)/breathing_range

    return df

def add_features(df, features, 
        range_window=100,
        range_smoothing=1,
        slope_shift=2,
        slope_smoothing=1,
    ):
    """
    This function adds features to the input data, based on the arguments
    given in the features-list.

    Args:
    df (pandas DataFrame): Data frame to add features to.
    features (list): A list containing keywords specifying which features to
        add.
    range_window (int): How many time steps to use when calculating range.
    range_smoothing (int): Rolling mean window for smoothing range.
    slope_shift (int): How many time steps to use when calculating slope.
    slope_smoothing (int): Rolling mean window for smoothing slope.

    Returns:
        df (pandas DataFrame): Data frame with added features.

    """

    # Stop function of features is not a list
    if features == None:
        return 0

    if "ribcage_min" in features:
        ribcage_min = df["ribcage"].rolling(range_window).min()

        df["ribcage_min"] = ribcage_min

    if "ribcage_max" in features:
        ribcage_max = df["ribcage"].rolling(range_window).max()

        df["ribcage_max"] = ribcage_max

    if "ribcage_range" in features:

        ribcage_min = df["ribcage"].rolling(range_window).min()
        ribcage_max = df["ribcage"].rolling(range_window).max()
        ribcage_range = ribcage_max - ribcage_min

        df["ribcage_range"] = ribcage_range.rolling(range_smoothing).mean()

    if "abdomen_min" in features:
        abdomen_min = df["abdomen"].rolling(range_window).min()

        df["abdomen_min"] = abdomen_min

    if "abdomen_max" in features:
        abdomen_max = df["abdomen"].rolling(range_window).max()

        df["abdomen_max"] = abdomen_max

    if "abdomen_range" in features:

        abdomen_min = df["abdomen"].rolling(range_window).min()
        abdomen_max = df["abdomen"].rolling(range_window).max()
        abdomen_range = abdomen_max - abdomen_min

        df["abdomen_range"] = abdomen_range.rolling(range_smoothing).mean()

    if "ribcage_gradient" in features:
        df["ribcage_gradient"] = np.gradient(df["ribcage"])

    if "abdomen_gradient" in features:
        df["abdomen_gradient"] = np.gradient(df["abdomen"])
        
    if "heartrate_gradient" in features:
        df["heartrate_gradient"] = np.gradient(df["heartrate"])

    if "ribcage_frequency" in features:
        df["ribcage_frequency"] = calculate_frequency(df["ribcage"])

    if "abdomen_frequency" in features:
        df["abdomen_frequency"] = calculate_frequency(df["abdomen"])

    if "ribcage_slope" in features:
        df["ribcage_slope"] = calculate_slope(
                df["ribcage"], slope_shift, slope_smoothing
        )

    if "abdomen_slope" in features:
        df["abdomen_slope"] = calculate_slope(
                df["abdomen"], slope_shift, slope_smoothing
        )

    if "heartrate_slope" in features:
        df["heartrate_slope"] = calculate_slope(
                df["heartrate"], slope_shift, slope_smoothing
        )

    if "ribcage_slope_cyclic" in features:
        df["ribcage_slope_sin"] = np.sin(
                calculate_slope(df["ribcage"], slope_shift,
                    rolling_mean_window=1, absvalue=False)
        )
        df["ribcage_slope_cos"] = np.cos(
                calculate_slope(df["ribcage"], slope_shift,
                    rolling_mean_window=1, absvalue=False)
        )

    if "abdomen_slope_cyclic" in features:
        df["abdomen_slope_sin"] = np.sin(
                calculate_slope(df["abdomen"], slope_shift,
                    rolling_mean_window=1, absvalue=False)
        )
        df["abdomen_slope_cos"] = np.cos(
                calculate_slope(df["abdomen"], slope_shift,
                    rolling_mean_window=1, absvalue=False)
        )

    if "heartrate_slope_cyclic" in features:
        df["heartrate_slope_sin"] = np.sin(
                calculate_slope(df["heartrate"], slope_shift,
                    rolling_mean_window=1, absvalue=False)
        )
        df["heartrate_slope_cos"] = np.cos(
                calculate_slope(df["heartrate"], slope_shift,
                    rolling_mean_window=1, absvalue=False)
        )

def calculate_frequency(data, rolling_mean_window=100):
    """Calculate frequency based on peaks.

    Args:
        data (array): Data for frequency calculation.
        rolling_mean_window (int): Window for calculating rolling mean.

    Returns:
        freq (array): Array of frequency.

    """

    peaks_indices = find_peaks(data, distance=5)[0]
    peaks = np.zeros(len(data))
    peaks[peaks_indices] = 1

    freq = []
    f = 0
    counter = 0

    for i, p in enumerate(peaks):

        if p == 1:
            f = 10 / counter
            counter = 0
        else:
            counter += 1

        freq.append(f)

    freq = np.array(freq)*60
    freq = pd.Series(freq).rolling(rolling_mean_window).mean()

    return freq

def calculate_slope(data, shift=2, rolling_mean_window=1, absvalue=False):
    """Calculate slope.

    Args:
        data (array): Data for slope calculation.
        shift (int): How many steps backwards to go when calculating the slope.
            For example: If shift=2, the slope is calculated from the data
            point two time steps ago to the data point at the current time
            step.
        rolling_mean_window (int): Window for calculating rolling mean.

    Returns:
        slope (array): Array of slope angle.

    """

    v_dist = data - data.shift(shift)
    h_dist = 0.1 * shift

    slope = np.arctan(v_dist / h_dist)

    if absvalue:
        slope = np.abs(slope)

    slope = slope.rolling(rolling_mean_window).mean()

    return slope

if __name__ == "__main__":

    np.random.seed(2020)

    featurize(sys.argv[1:])
