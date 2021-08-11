#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test Erdre ML pipeline for classification.

Author:
    Erik Johannes Husom

Created:
    2021-08-11

"""
import os
from pathlib import Path
import shutil
import subprocess
import unittest

import json
import numpy as np
import pandas as pd
import yaml

class TestRegression(unittest.TestCase):

    def test_cnc_milling(self):
        """Test pipeline on CNC milling data with regression."""

        dataset_name = "cnc_milling"
        dataset_path = Path("../assets/data/raw/" + dataset_name)
        cnc_milling_url = "https://raw.githubusercontent.com/ejhusom/cnc_milling_tool_wear/master/data/experiment_{:02d}.csv"
        n_files = 10

        dataset_present = False
        files_missing = []

        for i in range(1, n_files + 1):
            if os.path.exists(dataset_path / "{:02d}.csv".format(i)):
                files_missing.append(False)
            else:
                files_missing.append(True)

        if any(files_missing):
            print("Data set not present, downloading files...")
            dataset_path.mkdir(parents=True, exist_ok=True)

            for i in range(1, n_files + 1):
                df = pd.read_csv(cnc_milling_url.format(i))
                df.to_csv(dataset_path / "{:02d}.csv".format(i))
        
        create_params_file()

        run_experiment = subprocess.Popen(["dvc", "repro"], cwd="../")
        run_experiment.wait()

        restore_params_file()

        with open("../assets/metrics/metrics.json", "r") as infile:
            metrics = json.load(infile)

        print(metrics)
        r2_score = metrics["r2"]

        assert r2 > 0.8


def create_params_file():

    params_string = """
profile:
    dataset: cnc_milling

clean:
    target: X1_ActualPosition
    classification: False
    onehot_encode_target: False
    combine_files: True
    percentage_zeros_threshold: 1.0
    correlation_metric: pearson
    input_max_correlation_threshold: 1.0

featurize:
    features:
    add_rolling_features: False
    rolling_window_size: 100
    remove_features:
    target_min_correlation_threshold: 0.0
    train_split: 0.6
    shuffle_files: False
    calibrate_split: 0.0

split:
    train_split: 0.6
    shuffle_files: False
    calibrate_split: 0.0

scale:
    input: standard
    output: minmax

sequentialize:
    window_size: 20
    target_size: 1
    shuffle_samples: False

train:
    learning_method: cnn
    n_epochs: 10
    batch_size: 512
    kernel_size: 5
    early_stopping: False
    patience: 10

evaluate:
    """

    params = yaml.safe_load(params_string)
    shutil.move("../params.yaml", "../params.yaml.bak")



    with open("../params.yaml", "w") as outfile:
        yaml.dump(params, outfile)

def restore_params_file():

    shutil.move("../params.yaml.bak", "../params.yaml")


if __name__ == '__main__':

    unittest.main()


