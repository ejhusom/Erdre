#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train deep learning model to estimate power from breathing data.


Author:   
    Erik Johannes Husom

Created:  
    2020-09-16  

"""
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import yaml

from config import MODELS_PATH, MODELS_FILE_PATH, TRAININGLOSS_PLOT_PATH
from config import PLOTS_PATH
from model import cnn, dnn, lstm, cnndnn

def train(filepath):
    """Train model to estimate power.

    Args:
        filepath (str): Path to training set.

    """
    
    MODELS_PATH.mkdir(parents=True, exist_ok=True)

    # Load parameters
    params = yaml.safe_load(open("params.yaml"))["train"]
    net = params["net"]
    use_early_stopping = params["early_stopping"]
    patience = params["patience"]

    # Load training set
    train = np.load(filepath)

    X_train = train["X"]
    y_train = train["y"]

    n_features = X_train.shape[-1]

    hist_size = X_train.shape[-2]
    target_size = y_train.shape[-1]

    # Build model
    if net == "cnn":
        hist_size = X_train.shape[-2]
        model = cnn(hist_size, n_features, output_length=target_size,
                kernel_size=params["kernel_size"]
        )
    elif net == "dnn":
        model = dnn(n_features, output_length=target_size)
    # elif net == "lstm":
    #     pass
    # elif net == "cnndnn":
    #     pass
    else:
        raise NotImplementedError("Only 'cnn' is implemented.")

    print(model.summary())

    # Save a plot of the model. Will not work if Graphviz is not installed, and
    # is therefore skipped if an error is thrown.
    try:
        PLOTS_PATH.mkdir(parents=True, exist_ok=True)
        plot_model(
            model,
            to_file=PLOTS_PATH / 'model.png',
            show_shapes=False,
            show_layer_names=True,
            rankdir='TB',
            expand_nested=True,
            dpi=96
        )
    except:
        print("Failed saving plot of the network architecture.")

    early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=patience,
            verbose=4
    )

    model_checkpoint = ModelCheckpoint(
            MODELS_FILE_PATH, 
            monitor="val_loss",
            save_best_only=True
    )
    
    if use_early_stopping:
        # Train model for 10 epochs before adding early stopping
        history = model.fit(
            X_train, y_train, 
            epochs=10,
            batch_size=params["batch_size"],
            validation_split=0.25,
        )

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        history = model.fit(
            X_train, y_train, 
            epochs=params["n_epochs"],
            batch_size=params["batch_size"],
            validation_split=0.25,
            callbacks=[early_stopping, model_checkpoint]
        )

        loss += history.history['loss']
        val_loss += history.history['val_loss']

    else:
        history = model.fit(
            X_train, y_train, 
            epochs=params["n_epochs"],
            batch_size=params["batch_size"],
            validation_split=0.25,
        )

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        model.save(MODELS_FILE_PATH)

    TRAININGLOSS_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Best model in epoch: {np.argmax(np.array(val_loss))}")

    n_epochs = range(len(loss))

    plt.figure()
    plt.plot(n_epochs, loss, label="Training loss")
    plt.plot(n_epochs, val_loss, label="Validation loss")
    plt.legend()
    plt.savefig(TRAININGLOSS_PLOT_PATH)



if __name__ == "__main__":

    np.random.seed(2020)

    train(sys.argv[1])
