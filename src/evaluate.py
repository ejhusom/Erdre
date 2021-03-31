#!/usr/bin/env python3
"""Evaluate deep learning model.

Author:   
    Erik Johannes Husom

Created:  
    2020-09-17

"""
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras import models
import yaml

from config import METRICS_FILE_PATH, PLOTS_PATH, PREDICTION_PLOT_PATH, DATA_PATH


def evaluate(model_filepath, test_filepath):
    """Evaluate model to estimate power.

    Args:
        model_filepath (str): Path to model.
        test_filepath (str): Path to test set.

    """

    METRICS_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Load parameters
    params = yaml.safe_load(open("params.yaml"))["evaluate"]
    params_train = yaml.safe_load(open("params.yaml"))["train"]

    test = np.load(test_filepath)

    X_test = test["X"]
    y_test = test["y"]

    model = models.load_model(model_filepath)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("MSE: {}".format(mse))
    print("R2: {}".format(r2))

    plot_prediction(y_test, y_pred, inputs=X_test, info="(MSE: {})".format(mse))
    plot_individual_predictions(y_test, y_pred)

    with open(METRICS_FILE_PATH, "w") as f:
        json.dump(dict(mse=mse), f)


def plot_prediction(y_true, y_pred, inputs=None, info=""):
    """Plot the prediction compared to the true targets.

    Args:
        y_true (array): True targets.
        y_pred (array): Predicted targets.
        include_input (bool): Whether to include inputs in plot. Default=True.
        inputs (array): Inputs corresponding to the targets passed. If
            provided, the inputs will be plotted together with the targets.
        info (str): Information to include in the title string.

    """

    PREDICTION_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)

    x = np.linspace(0, y_true.shape[0]-1, y_true.shape[0])
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    config = dict({'scrollZoom': True})

    fig.add_trace(
            go.Scatter(x=x, y=y_true[:,-1].reshape(-1), name="true"),
            secondary_y=False,
    )

    fig.add_trace(
            go.Scatter(x=x, y=y_pred[:,-1].reshape(-1), name="pred"),
            secondary_y=False,
    )

    if inputs is not None:
        input_columns = pd.read_csv(DATA_PATH / "input_columns.csv")
        
        if len(inputs.shape) == 3:
            n_features = inputs.shape[-1]
        elif len(inputs.shape) == 2:
            n_features = len(input_columns) - 1

        for i in range(n_features):

            if len(inputs.shape) == 3:
                fig.add_trace(
                        go.Scatter(
                            x=x, y=inputs[:, -1, i],
                            name=input_columns.iloc[i+1, 1]
                        ),
                        secondary_y=True,
                )
            elif len(inputs.shape) == 2:
                fig.add_trace(
                        go.Scatter(
                            x=x, y=inputs[:, i-n_features],
                            name=input_columns.iloc[i+1, 1]
                        ),
                        secondary_y=True,
                )

    fig.update_layout(title_text="True vs pred " + info)
    fig.update_xaxes(title_text="time step")
    fig.update_yaxes(title_text="target variable", secondary_y=False)
    fig.update_yaxes(title_text="scaled units", secondary_y=True)

    fig.write_html(str(PLOTS_PATH / "prediction.html"))
    # fig.show(config=config)

def plot_individual_predictions(y_true, y_pred):
    """
    Plot the prediction compared to the true targets.
    
    """

    target_size = y_true.shape[-1]
    pred_curve_step = target_size

    pred_curve_idcs = np.arange(0, y_true.shape[0], pred_curve_step)
    # y_indeces = np.arange(0, y_true.shape[0]-1, 1)
    y_indeces = np.linspace(0, y_true.shape[0]-1, y_true.shape[0])

    n_pred_curves = len(pred_curve_idcs)

    # plt.figure()
    # plt.gca().set_prop_cycle(plt.cycler(
    #     'color', plt.cm.jet(np.linspace(0, 1, n_pred_curves))
    # ))
    fig = go.Figure()

    y_true_df = pd.DataFrame(y_true[:,0])

    # plt.plot(y_true_df, label="true", linewidth = 0.5)
    fig.add_trace(
            go.Scatter(x=y_indeces, y=y_true[:,0].reshape(-1), name="true")
    )

    predictions = []

    for i in pred_curve_idcs:
        indeces = y_indeces[i:i + target_size]
        
        if len(indeces) < target_size:
            break

        y_pred_df = pd.DataFrame(y_pred[i,:], index=indeces)
        
        predictions.append(y_pred_df)

        # plt.plot(y_pred_df, alpha=0.6, linewidth = 0.4)
        fig.add_trace(
                go.Scatter(x=indeces, y=y_pred[i,:].reshape(-1),
                    showlegend=False,
                    # line=dict(color='green'),
                    mode="lines")
        )

    PREDICTION_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # plt.title("Predictions", wrap=True)
    # plt.savefig(str(PLOTS_PATH / "prediction_individuals.png"))
    plt.show()
    fig.write_html(str(PLOTS_PATH / "prediction_individuals.html"))

if __name__ == "__main__":

    np.random.seed(2020)

    if len(sys.argv) < 3:
        try:
            evaluate("assets/models/model.h5", "assets/data/combined/test.npz")
        except:
            print("Could not find model and test set.")
            sys.exit(1)
    else:
        evaluate(sys.argv[1], sys.argv[2])
