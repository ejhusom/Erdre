#!/usr/bin/env python3
# ============================================================================
# File:     deeppower.py
# Author:   Erik Johannes Husom
# Created:  2020-08-25
# ----------------------------------------------------------------------------
# Description:
# Estimate power during workout using deep learning.
# ============================================================================
import json
import os
import pickle

try:
    import gnuplotlib as gp
except:
    pass

from preprocess import *
from neuraltimeseries import *
from utils import *


class DeepPower(Dataset, NeuralTimeSeries):
    """Estimate power from breathing and heart rate, using deep learning."""

    def __init__(
        self,
        data_file="../data/20200812-1809-merged.csv",
        hist_size=100,
        train_split=0.6,
        scale=True,
        reverse_train_split=False,
        verbose=False,
        net="cnn",
        n_epochs=100,
        time_id=time.strftime("%Y%m%d-%H%M%S"),
    ):

        self.net = net
        self.n_epochs = n_epochs

        Dataset.__init__(
            self,
            data_file=data_file,
            hist_size=hist_size,
            train_split=train_split,
            scale=scale,
            reverse_train_split=reverse_train_split,
            verbose=verbose,
            time_id=time_id,
        )

        self.plot_title = """File: {}, hist_size: {}, net: {}, n_epochs: {}, 
            added feats.: {}""".format(
            self.data_file, self.hist_size, self.net, self.n_epochs, self.added_features
        )

    def build_model(self):
        """Build the model."""

        try:
            NeuralTimeSeries.__init__(
                self,
                self.X_train,
                self.y_train,
                self.X_test,
                self.y_test,
                self.n_epochs,
                self.net,
                self.time_id,
            )
        except:
            raise AttributeError("Data is not preprocessed.")

    def predict(self, X_test=None, y_test=None):
        """Perform prediction using the trained model."""

        if X_test != None and y_test != None:
            self.X_test = X_test
            self.y_test = y_test

        self.y_pred = self.model.predict(self.X_test)

        # if self.y_scaler != None:
        #     self.y_test = self.y_scaler.inverse_transform(self.y_test)
        #     self.y_pred = self.y_scaler.inverse_transform(self.y_pred)
        #     print("Targets inverse transformed.")

    def plot_prediction(self, include_input=True):
        """
        Plot the prediction compared to the true targets.
        """

        # error_plot_average(self.y_test, self.y_pred, 168, self.time_id)

        plt.figure()

        plt.plot(self.y_test, label="true")
        plt.plot(self.y_pred, label="pred")

        if include_input:
            for i in range(self.X_test_pre_seq.shape[1]):
                # plt.plot(self.df.iloc[:,i], label=self.input_columns[i])
                plt.plot(
                    self.X_test_pre_seq[:, i] * 250, label=self.input_columns[i + 1]
                )

        plt.legend()
        plt.title(self.plot_title, wrap=True)
        plt.autoscale()
        plt.savefig(self.result_dir + self.time_id + "-pred.png")
        plt.show()

    def plot_prediction_plotly(self, include_input=True):
        """
        Plot the prediction compared to the true targets, using plotly.
        """

        x_len = len(self.y_test.flatten())
        x = np.linspace(0, x_len - 1, x_len)

        fig = go.Figure()
        config = dict({"scrollZoom": True})

        fig.add_trace(go.Scatter(x=x, y=self.y_test.flatten(), name="true"))
        fig.add_trace(go.Scatter(x=x, y=self.y_pred.flatten(), name="pred"))

        if include_input:
            for i in range(self.X_test_pre_seq.shape[1]):
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=self.X_test_pre_seq[:, i] * 250,
                        name=self.input_columns[i + 1],
                    )
                )

        fig.show(config=config)

    def plot_prediction_gp(self):
        """
        Plot the prediction compared to the true targets, using gnuplotlib.
        """

        termsize[0], termsize[1] = get_terminal_size()

        y_true = self.y_test.flatten()
        y_pred = self.y_pred.flatten()

        x_len = len(y_true)
        x = np.linspace(0, x_len - 1, x_len)

        gp.plot(
            (x, y_true, dict(legend="true")),
            (x, y_pred, dict(legend="pred")),
            unset="grid",
            terminal="dumb {} {}".format(termsize[0], termsize[1]),
        )


if __name__ == "__main__":
    np.random.seed(2020)

    args = parse_arguments()

    # Open config parameters from file if arguments is given
    if args.config != None:
        with open(args.config, "rt") as f:
            t_args = argparse.Namespace()
            t_args.__dict__.update(json.load(f))
            args = parser.parse_args(namespace=t_args)

    power_estimation = DeepPower(
        data_file=args.data_file,
        hist_size=int(args.hist_size),
        train_split=float(args.train_split),
        reverse_train_split=to_bool(args.reverse_train_split),
        net=args.net,
        n_epochs=int(args.n_epochs),
        verbose=to_bool(args.verbose),
        time_id=time.strftime("%Y%m%d-%H%M%S"),
    )

    power_estimation.preprocess(list(args.features), list(args.remove))
    power_estimation.build_model()

    # Load model and scaler if argument is given
    if args.model != None and args.model != "None":
        if args.model.endswith(".h5"):
            power_estimation.set_model(args.model)
        else:
            raise ValueError("Model does not have correct extension.")

        if args.scaler != None and args.scaler != "None":
            power_estimation.set_scaler(args.scaler)
        else:
            raise Exception("To load pretrained model, scaler must be given.")
            sys.exit(1)

    # Train if argument is given
    if to_bool(args.train):
        power_estimation.fit()

    # Predict and plot if argument is given
    if to_bool(args.predict):
        power_estimation.predict()
        power_estimation.plot_prediction()
        if to_bool(args.plotly):
            power_estimation.plot_prediction_plotly()
        elif to_bool(args.gnuplotlib):
            power_estimation.plot_prediction_gp()
