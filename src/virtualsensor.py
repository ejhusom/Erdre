#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Performing inference using virtual sensors created with Erdre.

Author:
    Erik Johannes Husom

Created:
    2021-10-29 Friday 09:13:14 

"""
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np

from config import *
from profiling import *
from clean import *
from featurize import *
from split import *
from scale import *
from sequentialize import *
from combine import *
from train import *
from evaluate import *
from preprocess_utils import split_X_sequences

class VirtualSensor:

    def __init__(
        self,
        params_file=PARAMS_FILE_PATH,
        profile_file=PROFILE_JSON_PATH,
        input_scaler_file=INPUT_SCALER_PATH,
        output_scaler_file=OUTPUT_SCALER_PATH,
        model_file=MODELS_FILE_PATH,
        verbose=True
    ):
        """Initialize virtual sensor with the needed assets.

        To run a virtual sensor, certain assets are needed:

        - params.yaml: Parameters used to create virtual sensor.
        - profile.json: Profiling of training data set.
        - input_scaler.scl: Scaler used to transform input data.
        - input_scaler.scl: Scaler used to transform output data.
        - model.h5: Model.

        """

        self.params_file = params_file
        self.profile_file = profile_file
        self.input_scaler_file = input_scaler_file
        self.output_scaler_file = output_scaler_file
        self.model_file = model_file

        self.verbose = verbose

        self.assets_files = [
            self.params_file,
            self.profile_file,
            self.input_scaler_file,
            self.output_scaler_file,
            self.model_file
        ]

        self._check_assets_existence()

    def _check_assets_existence(self):
        """Check if the needed assets exists."""

        check_ok = True

        for path in self.assets_files:
            if not os.path.exists(path):
                print(f"File {path} not found.")
                check_ok = False

        if check_ok:
            print("All assets found.")
        else:
            raise AssertionError("Assets missing.")

    def run_virtual_sensor(self, input_df):
        """Run virtual sensor.

        Args:
            TODO: What input format here? Currently it takes a CSV-file.
            input_data (): 

        """

        params = yaml.safe_load(open("params.yaml"))
        classification = params["clean"]["classification"]
        onehot_encode_target = params["clean"]["onehot_encode_target"]
        learning_method = params["train"]["learning_method"]
        input_method = params["scale"]["input"]
        output_method = params["scale"]["output"]
        window_size = params["sequentialize"]["window_size"]
        overlap = params["sequentialize"]["overlap"]
        
        df = clean(input_df=input_df)
        df = featurize(inference=True, input_df=df)

        X = np.array(df)


        input_scaler = joblib.load(INPUT_SCALER_PATH)
        output_scaler = joblib.load(OUTPUT_SCALER_PATH)

        if input_method is not None:
            X = input_scaler.transform(X)

        X = split_X_sequences(X, window_size, overlap=overlap)

        if learning_method in NON_DL_METHODS:
            model = load(MODELS_FILE_PATH)
        else:
            model = models.load_model(MODELS_FILE_PATH)

        y_pred = model.predict(X)

        if onehot_encode_target:
            y_pred = np.argmax(y_pred, axis=-1)
        elif classification:
            y_pred = np.array((y_pred > 0.5), dtype=np.int)

        print(y_pred)
        plt.figure()
        plt.plot(y_pred)
        plt.show()

            
if __name__ == '__main__': 

    df = pd.read_csv("./assets/data/raw/cnc_without_target/02.csv")

    vs = VirtualSensor()
    vs.run_virtual_sensor(input_df=df)
