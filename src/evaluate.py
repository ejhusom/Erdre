#!/usr/bin/env python3
"""Evaluate deep learning model.

Author:
    Erik Johannes Husom

Created:
    2020-09-17


"""
import json
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sn
import shap
import yaml
from joblib import load
from nonconformist.base import RegressorAdapter
from nonconformist.cp import IcpRegressor
from nonconformist.nc import AbsErrorErrFunc, NcFactory, RegressorNc
from plotly.subplots import make_subplots
from sklearn.base import RegressorMixin
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_squared_error,
    r2_score,
)
from sklearn.neighbors import KNeighborsRegressor
from tensorflow.keras import metrics, models

import neural_networks as nn
from config import (
    DATA_PATH,
    INTERVALS_PLOT_PATH,
    METRICS_FILE_PATH,
    NON_DL_METHODS,
    PLOTS_PATH,
    PREDICTION_PLOT_PATH,
    PREDICTIONS_FILE_PATH
)


# class ConformalPredictionModel(RegressorMixin):
class ConformalPredictionModel(RegressorAdapter):
    """Implement custom sklearn model to use with the nonconformist library.

    Args:
        model_filepath (str): Path to ALREADY TRAINED model.

    """

    # def __init__(self, model_filepath):
    def __init__(self, model):
        # Load already trained model from h5 file.
        # self.model = models.load_model(model_filepath)
        # self.model_filepath = model_filepath

        # super(ConformalPredictionModel, self).__init__(models.load_model(model_filepath), fit_params=None)
        super(ConformalPredictionModel, self).__init__(model)

    def fit(self, X=None, y=None):
        # We don't do anything here because we are loading an already trained model in __init__().
        # Still, we need to implement this method so the conformal normalizer
        # is initialized by nonconformist.
        pass

    def predict(self, X=None):
        predictions = self.model.predict(X)
        predictions = predictions.reshape((predictions.shape[0],))

        return predictions


def evaluate(model_filepath, train_filepath, test_filepath, calibrate_filepath):
    """Evaluate model to estimate power.

    Args:
        model_filepath (str): Path to model.
        train_filepath (str): Path to train set.
        test_filepath (str): Path to test set.
        calibrate_filepath (str): Path to calibrate set.

    """

    METRICS_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Load parameters
    params = yaml.safe_load(open("params.yaml"))["evaluate"]
    params_train = yaml.safe_load(open("params.yaml"))["train"]
    params_split = yaml.safe_load(open("params.yaml"))["split"]
    classification = yaml.safe_load(open("params.yaml"))["clean"]["classification"]
    onehot_encode_target = yaml.safe_load(open("params.yaml"))["clean"][
        "onehot_encode_target"
    ]
    learning_method = params_train["learning_method"]

    test = np.load(test_filepath)
    X_test = test["X"]
    y_test = test["y"]

    # pandas data frame to store predictions and ground truth.
    df_predictions = None

    y_pred = None

    if params_split["calibrate_split"] > 0 and not classification:
        trained_model = models.load_model(model_filepath)
        # conformal_pred_model = ConformalPredictionModel(model_filepath)
        conformal_pred_model = ConformalPredictionModel(trained_model)

        m = nn.cnn(
            X_test.shape[-2],
            X_test.shape[-1],
            output_length=1,
            kernel_size=params_train["kernel_size"],
        )

        nc = RegressorNc(
            conformal_pred_model,
            err_func=AbsErrorErrFunc(),  # non-conformity function
            # normalizer_model=KNeighborsRegressor(n_neighbors=15)  # normalizer
            # normalizer=m
        )

        # nc = NcFactory.create_nc(conformal_pred_model,
        #     err_func=AbsErrorErrFunc(),  # non-conformity function
        #     # normalizer_model=KNeighborsRegressor(n_neighbors=15)  # normalizer
        #     normalizer_model=m
        # )

        model = IcpRegressor(nc)

        # Fit the normalizer.
        train = np.load(train_filepath)
        X_train = train["X"]
        y_train = train["y"]

        y_train = y_train.reshape((y_train.shape[0],))

        model.fit(X_train, y_train)

        # Calibrate model.
        calibrate = np.load(calibrate_filepath)
        X_calibrate = calibrate["X"]
        y_calibrate = calibrate["y"]
        y_calibrate = y_calibrate.reshape((y_calibrate.shape[0],))
        model.calibrate(X_calibrate, y_calibrate)

        print(f"Calibration: {X_calibrate.shape}")

        # Set conformal prediction error. This should be a parameter specified by the user.
        error = 0.05

        # Predictions will contain the intervals. We need to compute the middle
        # points to get the actual predictions y.
        predictions = model.predict(X_test, significance=error)

        # Compute middle points.
        y_pred = predictions[:, 0] + (predictions[:, 1] - predictions[:, 0]) / 2

        # Reshape to put it in the same format as without calibration set.
        y_pred = y_pred.reshape((y_pred.shape[0], 1))

        # Build data frame with predictions.
        my_results = list(
            zip(
                np.reshape(y_test, (y_test.shape[0],)),
                np.reshape(y_pred, (y_pred.shape[0],)),
                predictions[:, 0],
                predictions[:, 1],
            )
        )

        df_predictions = pd.DataFrame(
            my_results,
            columns=["ground_truth", "predicted", "lower_bound", "upper_bound"],
        )

        save_predictions(df_predictions)

        plot_confidence_intervals(df_predictions)

    else:
        if learning_method in NON_DL_METHODS:
            model = load(model_filepath)
        else:
            model = models.load_model(model_filepath)

        y_pred = model.predict(X_test)

        if onehot_encode_target:
            y_pred = np.argmax(y_pred, axis=-1)
        elif classification:
            y_pred = np.array((y_pred > 0.5), dtype=np.int)

    if classification:

        if onehot_encode_target:
            y_test = np.argmax(y_test, axis=-1)

        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy}")

        plot_prediction(y_test, y_pred, info="Accuracy: {})".format(accuracy))

        plot_confusion(y_test, y_pred)

        with open(METRICS_FILE_PATH, "w") as f:
            json.dump(dict(accuracy=accuracy), f)

        # ==========================================
        # TODO: Fix SHAP code
        # explainer = shap.TreeExplainer(model, X_test[:10])
        # shap_values = explainer.shap_values(X_test[:10])
        # plt.figure()
        # shap.summary_plot(shap_values[0][:,0,:], X_test[:10][:,0,:])
        # shap.image_plot([shap_values[i][0] for i in range(len(shap_values))], X_test[:10])
        # input_columns = pd.read_csv(DATA_PATH / "input_columns.csv").iloc[:,-1]
        # print(input_columns)
        # shap.force_plot(explainer.expected_value[0], shap_values[0][0])

        # plt.savefig("test.png")

        # feature_importances = model.feature_importances_
        # imp = list()
        # for i, f in enumerate(feature_importances):
        #     imp.append((f,i))

        # sorted_feature_importances = sorted(imp)

        # print("Feature importances")
        # print(sorted_feature_importances)
        # ==========================================

    # Regression:
    else:
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print("MSE: {}".format(mse))
        print("R2: {}".format(r2))

        plot_prediction(y_test, y_pred, inputs=None, info="(R2: {})".format(r2))

        # Only plot predicted sequences if the output samples are sequences.
        if len(y_test.shape) > 1 and y_test.shape[1] > 1:
            plot_sequence_predictions(y_test, y_pred)

        with open(METRICS_FILE_PATH, "w") as f:
            json.dump(dict(mse=mse, r2=r2), f)


def plot_confusion(y_test, y_pred):
    """Plotting confusion matrix of a classification model."""

    output_columns = np.array(
        pd.read_csv(DATA_PATH / "output_columns.csv", index_col=0)
    ).reshape(-1)

    n_output_cols = len(output_columns)
    indeces = np.arange(0, n_output_cols, 1)

    confusion = confusion_matrix(y_test, y_pred, normalize="true")
    # labels=indeces)

    print(confusion)

    df_confusion = pd.DataFrame(confusion)

    df_confusion.index.name = "True"
    df_confusion.columns.name = "Pred"
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_confusion, cmap="Blues", annot=True, annot_kws={"size": 16})
    plt.savefig(PLOTS_PATH / "confusion_matrix.png")


def save_predictions(df_predictions):
    """Save the predictions along with the ground truth as a csv file.

    Args:
        df_predictions_true (pandas dataframe): pandas data frame with the predictions and ground truth values.

    """

    PREDICTIONS_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)

    df_predictions.to_csv(PREDICTIONS_FILE_PATH, index=False)


def plot_confidence_intervals(df):
    """Plot the confidence intervals generated with conformal prediction.

    Args:
        df (pandas dataframe): pandas data frame.

    """

    INTERVALS_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)

    x = [x for x in range(1, df.shape[0] + 1, 1)]

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=x, y=df["predicted"], name="predictions"))

    fig.add_trace(
        go.Scatter(
            name="Upper Bound",
            x=x,
            y=df["upper_bound"],
            marker=dict(color="#444"),
            line=dict(width=0),
            mode="lines",
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            name="Lower Bound",
            x=x,
            y=df["lower_bound"],
            marker=dict(color="#444"),
            line=dict(width=0),
            mode="lines",
            fillcolor="rgba(68, 68, 68, 0.3)",
            fill="tonexty",
            showlegend=False,
        )
    )

    fig.write_html(str(PLOTS_PATH / "intervals.html"))


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

    x = np.linspace(0, y_true.shape[0] - 1, y_true.shape[0])
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    if len(y_true.shape) > 1:
        y_true = y_true[:, -1].reshape(-1)
        y_pred = y_pred[:, -1].reshape(-1)
    else:
        y = y_true
        y = y_pred

    fig.add_trace(
        go.Scatter(x=x, y=y_true, name="true"),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(x=x, y=y_pred, name="pred"),
        secondary_y=False,
    )

    if inputs is not None:
        input_columns = pd.read_csv(DATA_PATH / "input_columns.csv")

        if len(inputs.shape) == 3:
            n_features = inputs.shape[-1]
        elif len(inputs.shape) == 2:
            n_features = len(input_columns)

        for i in range(n_features):

            if len(inputs.shape) == 3:
                fig.add_trace(
                    go.Scatter(x=x, y=inputs[:, -1, i], name=input_columns.iloc[i, 1]),
                    secondary_y=True,
                )
            elif len(inputs.shape) == 2:
                fig.add_trace(
                    go.Scatter(
                        x=x, y=inputs[:, i - n_features], name=input_columns.iloc[i, 1]
                    ),
                    secondary_y=True,
                )

    fig.update_layout(title_text="True vs pred " + info)
    fig.update_xaxes(title_text="time step")
    fig.update_yaxes(title_text="target variable", secondary_y=False)
    fig.update_yaxes(title_text="scaled units", secondary_y=True)

    fig.write_html(str(PLOTS_PATH / "prediction.html"))


def plot_sequence_predictions(y_true, y_pred):
    """
    Plot the prediction compared to the true targets.

    """

    target_size = y_true.shape[-1]
    pred_curve_step = target_size

    pred_curve_idcs = np.arange(0, y_true.shape[0], pred_curve_step)
    # y_indeces = np.arange(0, y_true.shape[0]-1, 1)
    y_indeces = np.linspace(0, y_true.shape[0] - 1, y_true.shape[0])

    n_pred_curves = len(pred_curve_idcs)

    fig = go.Figure()

    y_true_df = pd.DataFrame(y_true[:, 0])

    fig.add_trace(go.Scatter(x=y_indeces, y=y_true[:, 0].reshape(-1), name="true"))

    predictions = []

    for i in pred_curve_idcs:
        indeces = y_indeces[i: i + target_size]

        if len(indeces) < target_size:
            break

        y_pred_df = pd.DataFrame(y_pred[i, :], index=indeces)

        predictions.append(y_pred_df)

        fig.add_trace(
            go.Scatter(
                x=indeces, y=y_pred[i, :].reshape(-1), showlegend=False, mode="lines"
            )
        )

    PREDICTION_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fig.write_html(str(PLOTS_PATH / "prediction_sequences.html"))


if __name__ == "__main__":

    if len(sys.argv) < 3:
        try:
            evaluate(
                "assets/models/model.h5",
                "assets/data/combined/train.npz",
                "assets/data/combined/test.npz",
                "assets/data/combined/calibrate.npz",
            )
        except:
            print("Could not find model and test set.")
            sys.exit(1)
    else:
        evaluate(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
