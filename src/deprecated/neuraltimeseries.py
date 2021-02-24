#!/usr/bin/env python3
# ============================================================================
# File:     neuraltimeseries.py
# Author:   Erik Johannes Husom
# Created:  2020-08-24
# ----------------------------------------------------------------------------
# Description:
# Defining a model for predicting power.
# ============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Model, load_model


def CNNKeras(input_x, input_y, n_steps_out=1):
    """Define a CNN model architecture using Keras.
    Parameters
    ----------
    input_x : int
        Number of time steps to include in each sample, i.e. how much history
        is matched with a given target.
    input_y : int
        Number of features for each time step in the input data.
    n_steps_out : int
        Number of output steps.

    Returns
    -------
    model : Keras model
        Model to be trained.

    """

    kernel_size = 2

    model = models.Sequential()
    model.add(
        layers.Conv1D(
            filters=64,
            kernel_size=kernel_size,
            activation="relu",
            input_shape=(input_x, input_y),
        )
    )
    model.add(layers.Conv1D(filters=32, kernel_size=kernel_size, activation="relu"))
    # model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(n_steps_out, activation="linear"))
    model.compile(optimizer="adam", loss="mse", metrics=["mae", "mape"])

    return model


class NeuralTimeSeries:
    """Run training and prediction using Keras."""

    def __init__(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        n_epochs=100,
        net="cnn",
        time_id=time.strftime("%Y%m%d-%H%M%S"),
    ):

        tf.random.set_seed(2020)

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        self.n_epochs = n_epochs
        self.net = net.lower()
        self.model_loaded = False
        self.time_id = time_id

        self.n_features = self.X_train.shape[-1]

        try:
            self.input_x = self.X_train.shape[-2]
            self.input_y = self.X_train.shape[-1]
        except:
            self.input_x = self.X_test.shape[-2]
            self.input_y = self.X_test.shape[-1]

        if self.net == "lstm":
            # self.model = LSTMKeras(
            #     self.input_x, self.input_y, self.n_steps_out
            # )
            raise NotImplementedError
        else:
            self.model = CNNKeras(self.input_x, self.input_y)

        if self.verbose:
            print(self.model.summary())

    def fit(self, verbose=None):
        """Train the model."""

        if not hasattr(self, "model"):
            self.build_model()

        if verbose == None:
            verbose = 1 if self.verbose else 0

        self.history = self.model.fit(
            self.X_train,
            self.y_train,
            epochs=self.n_epochs,
            verbose=verbose,
            batch_size=128,
            # validation_split=0.2
        )

        if self.verbose:
            scores = self.model.evaluate(self.X_test, self.y_test, verbose=1)
            print(scores)

        self.model.save(self.result_dir + self.time_id + "-model.h5")

        with open(self.result_dir + self.time_id + "-config.json", "w") as f:
            f.write(self.model.to_json())

    def set_model(self, model_file):
        """Setting the model to be used from a model saved with Keras."""

        self.model = models.load_model(model_file)
        self.model_loaded = True
        print("Model loaded: {}".format(model_file))


if __name__ == "__main__":
    # Setting seed for consistent results during testing.
    np.random.seed(2020)

    analysis = NeuralTimeSeries()
