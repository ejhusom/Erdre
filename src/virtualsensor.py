#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Performing inference using virtual sensors created with Erdre.

Author:
    Erik Johannes Husom

Created:
    2021-10-29 Friday 09:13:14 

"""
import os

from config import *

class VirtualSensor:

    def __init__(
        self,
        params_file=PARAMS_PATH,
        profile_file=PROFILE_JSON_PATH,
        input_scaler_file=INPUT_SCALER_PATH,
        output_scaler_file=OUTPUT_SCALER_PATH,
        model_file=MODEL_PATH,
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

        self.assets = [
            self.params_file,
            self.profile_file,
            self.input_scaler_file,
            self.output_scaler_file,
            self.model_file
        ]

        self.check_assets_existence()

    def check_assets_existence(self):
        """Check if the needed assets exists.  """

        # try:
        #     os
        pass

    def 
