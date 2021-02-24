#!/usr/bin/env python3
# ============================================================================
# File:     restructure.py
# Author:   Erik Johannes Husom
# Created:  2020-01-27
# ----------------------------------------------------------------------------
# Description:
# Process and plot raw data from BreathZpot FLOW and PM5 monitor.
# ============================================================================
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

from config import *


def pad_data(df):
    """Pad data such that there is a datapoint for every 0.1 second.

    Parameters
    ----------
    df : DataFrame
        The dataframe that contains the data to be padded. Dataframe is
        required to have a column that is called 'writetime'.

    Returns
    -------
    df : DataFrame
        The padded dataframe.

    """

    print(f"Data size before padding: {df.shape[0]} rows.")
    df.index = np.around(np.array(df["writetime"]), decimals=-2)

    df = (
        df.reset_index()
        .drop_duplicates(subset="index", keep="first")
        .set_index("index")
    )

    # Pad data such that every tenth of a second has a data point
    t = np.arange(df.index[0], df.index[-1] + 100, 100)
    df = df.reindex(t, method="nearest")
    df["time"] = df.index / 1000

    print(f"Data size after padding: {df.shape[0]} rows.")

    return df


def check_missing_points(df):

    diff = df.time.diff(1)
    print(f" Max timestep (should be 0.1): {np.max(diff)}")
    print(f" Min timestep (should be 0.1): {np.min(diff)}")
    print(f"Duplicate rows: {df['time'].duplicated().any()}")


def explore_timestamps(df):

    try:
        plt.plot(df.writediff)
        plt.title(f"{df.name}: Intervals of writetime.")
        plt.show()
    except:
        print(f"{df.name} lacks necessary columns to plot write time difference.")

    try:
        plt.plot(df.timediff)
        plt.title(f"{df.name}: Time interval of processed data.")
        plt.show()
    except:
        print(f"{df.name} lacks necessary columns to plot time intervals.")


def structure_breathing_data(data, placement, t0):
    """Process breathing data from the FLOW sensor.

    Parameters
    ----------
    placement : string
        Placement of the FLOW sensor. Usually 'ribcage' or 'abdomen'.

    Returns
    -------
    df : pandas DataFrame
        Dataframe containing the structured raw data.

    """

    print(f"PROCESSING: {placement}")

    # Isolate the breathing data of current placement
    df = data[data["datatype"] == placement][
        ["writetime", "value", "value2"]
    ].reset_index(drop=True)
    df["value2"] -= t0 - 700
    df.rename(columns={"value2": "milliseconds"}, inplace=True)
    if df.empty:
        print(f"No data from {placement}.")
        df.name = placement
        return df

    print(f"Original data size for {placement}: {df.shape[0]} rows.")

    df.index = np.around(df.milliseconds, decimals=-2)
    df.index.names = ["index"]
    dups_time = df.pivot_table(index=df.index, aggfunc="size")
    dups_time = dups_time[dups_time > 1]

    print(
        f"Found {dups_time.shape[0]} points with duplicate timestamp for {placement}."
    )

    df = df.loc[~df.index.duplicated(keep="first")]

    # Pad data such that every tenth of a second has a data point
    t = np.arange(df.index[0], df.index[-1] + 100, 100)
    df = df.reindex(t).ffill()

    df["time"] = df.index / 1000

    print(f"Processed data size for {placement}: {df.shape[0]} rows.")
    df.name = placement

    # Add columns with timestep difference, for easy inspection
    df["writediff"] = df.writetime.diff(1)
    df["timediff"] = df.time.diff(1)

    return df


def structure_power_data(data):
    # POWER DATA
    power = data[data["datatype"] == "power"][
        ["writetime", "value", "value2"]
    ].reset_index(drop=True)

    # Rename column that contains elapsed time:
    power.rename(columns={"value2": "elapsedtime"}, inplace=True)

    if not power.empty:
        power = pad_data(power)

    power["writediff"] = power.writetime.diff(1)

    power.name = "power"

    return power


def structure_calories_data(data):
    # POWER DATA
    calories = data[data["datatype"] == "calories"][
        ["writetime", "value", "value2"]
    ].reset_index(drop=True)

    # Rename column that contains elapsed time:
    calories.rename(columns={"value2": "elapsedtime"}, inplace=True)

    if not calories.empty:
        calories = pad_data(calories)

    calories["writediff"] = calories.writetime.diff(1)

    calories.name = "calories"

    return calories


def structure_heartrate_data(data):
    # HEART RATE DATA
    heartrate = data[data["datatype"] == "heartrate"][
        ["writetime", "value"]
    ].reset_index(drop=True)

    if not heartrate.empty:
        heartrate = pad_data(heartrate)
        heartrate["writediff"] = heartrate.writetime.diff(1)

    heartrate.name = "heartrate"
    # explore_timestamps(heartrate)

    rr = data[data["datatype"] == "rr"][["writetime", "value"]].reset_index(drop=True)

    if not rr.empty:
        rr["cumulative"] = rr.value.cumsum()
        rr.cumulative += rr.writetime[0]
        rr["dRR"] = rr.value.diff(1)
        rr.loc[0, "dRR"] = 0
        rr["time"] = rr.writetime / 1000
        rr["writediff"] = rr.writetime.diff(1)

    rr.name = "rr"

    return heartrate


def restructure(filepaths, show=False, output=None):
    """Restructure raw data into dataframe.

    Args:
        filepaths (list of str): List of filepaths.

    """

    # If filepaths is a string (e.g. only one filepath), wrap this in a list
    if isinstance(filepaths, str):
        filepaths = [filepaths]

    DATA_RESTRUCTURED_PATH.mkdir(parents=True, exist_ok=True)

    for filepath in filepaths:

        print("Processing {}".format(filepath))
        # READ RAW DATA
        data = pd.read_csv(
            filepath,
            names=["writetime", "datatype", "value", "value2", "na"],
            header=None,
            index_col=False,
        )
        print(data)

        # change timestamp from unix time to relative start time:
        t0 = data["writetime"][0]
        data["writetime"] -= t0 - 700

        ribcage = structure_breathing_data(data, "ribcage", t0)
        abdomen = structure_breathing_data(data, "abdomen", t0)
        heartrate = structure_heartrate_data(data)
        power = structure_power_data(data)
        calories = structure_calories_data(data)

        dfs = [power, ribcage, abdomen, heartrate, calories]

        # Loop to make sure that the first dataframe in dfs actually contains
        # values, otherwise the merging of all dataframes will fail.
        for i in range(len(dfs)):
            try:
                _ = dfs[0][["time", "value"]]
                break
            except:
                dfs = dfs[1:] + dfs[:1]
                print(dfs[0].name)

        # Merging dataframes into one dataframe.
        for i in range(len(dfs)):
            df_name = dfs[i].name
            if i == 0:
                merged_dfs = dfs[0][["time", "value"]]
                merged_dfs = merged_dfs.rename(columns={"value": df_name})
                continue

            if dfs[i].empty:
                merged_dfs[df_name] = np.nan
                print(f"Dataframe number {i+1} not found; inserted NaN instead.")
            else:
                df = dfs[i][["value", "time"]]
                merged_dfs = merged_dfs.merge(df, on="time", how="outer", sort=True)
                merged_dfs.rename(columns={"value": df_name}, inplace=True)

        if "rest" in filepath:
            # If the file is recorded under rest, set all power values to zero
            # Must first delete all rows were we do not have all other data
            merged_dfs = merged_dfs[merged_dfs["heartrate"].notna()]
            merged_dfs = merged_dfs[merged_dfs["ribcage"].notna()]
            merged_dfs = merged_dfs[merged_dfs["abdomen"].notna()]
            merged_dfs["power"] = 0
        else:
            # Drop all rows where we do not have power data
            merged_dfs = merged_dfs[merged_dfs["power"].notna()]

        # Plot resulting dataframe.
        if show:
            plt.plot(merged_dfs.time, merged_dfs.ribcage, label="rib")
            plt.plot(merged_dfs.time, merged_dfs.abdomen, label="ab")
            plt.plot(merged_dfs.time, merged_dfs.power, label="power")
            plt.plot(merged_dfs.time, merged_dfs.calories, label="cal")
            plt.plot(merged_dfs.time, merged_dfs.heartrate, label="hr")
            plt.legend()
            # plt.savefig(filename + "-dataframe.png")
            plt.show()

        merged_dfs.to_csv(
            DATA_RESTRUCTURED_PATH
            / (os.path.basename(filepath).replace(".", "-restructured."))
        )


if __name__ == "__main__":

    restructure(sys.argv[1:], show=False)

    # if len(sys.argv) > 1:
    #     restructure(sys.argv[1:], show=False)
    # else:
    #     params = yaml.safe_load(open("params.yaml"))["restructure"]
    #     restructure(params["files"], show=False)
