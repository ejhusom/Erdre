#!/usr/bin/env python3
# ============================================================================
# File:     Dataset.py
# Created:  2020-08-24
# Author:   Erik Johannes Husom
# ----------------------------------------------------------------------------
# Description: Preprocessing workout data.
# ============================================================================
import matplotlib.pyplot as plt

# plt.rcParams['figure.figsize'] = [5.0, 3.0]
# plt.rcParams['figure.dpi'] = 300
import plotly.express as px
import plotly.graph_objects as go

import datetime
import numpy as np
import os
import pandas as pd
import pickle
import string
import sys
import time
from scipy.fftpack import fft, ifft

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from raw_to_dataframe import *
from preprocess_utils import *
from utils import *


class Dataset:
    def __init__(
        self,
        filename="../data/20200812-1809-merged.csv",
        hist_size=1000,
        train_split=0.6,
        scale=True,
        reverse_train_split=False,
        verbose=False,
        target_name="power",
        time_id=time.strftime("%Y%m%d-%H%M%S"),
    ):
        """
        Load and preprocess data set.

        Parameters
        ----------
        date : string
            The last day of the desired data set.
        hist_size : int, default=1000
            How many past time steps should be used for prediction.
        train_split : float, (0,1)
            How much data to use for training (the rest will be used for testing.)
        scale : bool, default=True
            Whether to scale the data set or not.
        filename : string, default="DfEtna.csv"
            What file to get the data from.
        reverse_train_split : boolean, default=False
            Whether to use to first part of the dataset as test and the second
            part for training (if set to True). If set to False, it uses the
            first part for training and the second part for testing.
        verbose : boolean, default=False
            Printing what the program does.

        """

        self.filename = filename
        self.train_split = train_split
        self.hist_size = hist_size
        self.scale = scale
        self.reverse_train_split = reverse_train_split
        self.time_id = time_id
        self.verbose = verbose

        self.preprocessed = False
        self.scaler_loaded = False
        self.added_features = []
        self.target_name = target_name
        self.result_dir = "../results/" + time_id + "/"
        os.makedirs(self.result_dir)

    def preprocess(self, features=[], remove_features=[]):

        self.df, self.index = read_csv(
            self.filename,
            delete_columns=["time", "calories"] + remove_features,
            verbose=self.verbose,
        )
        print(self.df)

        # Move target column to the beginning of dataframe
        self.df = move_column(self.df, self.target_name, 0)

        if self.verbose:
            print_horizontal_line()
            print("DATAFRAME BEFORE FEATURE ENGINEERING")
            print(self.df)

        self.add_features(features)

        # Save the names of the input columns
        self.input_columns = self.df.columns
        input_columns_df = pd.DataFrame(self.input_columns)
        input_columns_df.to_csv(self.result_dir + self.time_id + "-input_columns.csv")

        self.data = self.df.to_numpy()

        self._split_train_test(test_equals_train=False)
        self._scale_data()

        # Save data for inspection of preprocessed data set
        self.df.to_csv("tmp_df.csv")
        np.savetxt("tmp_X_train.csv", self.X_train, delimiter=",")

        self._split_sequences()

        self.n_features = self.X_train.shape[-1]

        if self.verbose:
            print("Following features are used in data set:")
            print(self.input_columns)

        # Save test targets for inspection
        np.savetxt("tmp_y_test.csv", self.y_test, delimiter=",")

        self.preprocessed = True

    def add_features(self, features):
        """
        This function adds features to the input data, based on the arguments
        given in the features-list.

        Parameters
        ----------
        features : list
            A list containing keywords specifying which features to add.

        """
        pass

    def add_feature(self, name, feature_col, add_to_hist_matrix=False):
        """
        Adding a feature to the data set. The name is 'registered' into one of
        two lists, in order to keep track of what features that are added to
        the raw data.

        The differences between the two lists are as follows: When using the
        CNNDense submethod, the list self.added_features consists of features
        that will be sent to the Dense-part of the network. If the parameter
        'add_to_hist_matrix' is set to True, the feature will be registered in
        the other list, self.added_hist_features, which will be sent to the
        CNN-part of the network when using CNNDense, together with the
        observation history of the data set. The separation of the two lists
        only matter when using the CNNDense submethod.

        The point of this function is to make a consistent treatment of added
        features and reducing the number of lines required to add a feature.

        Parameters
        ----------
        name : string
            What to call the new feature.
        feature_col : array-like
            The actual data to add to the input matrix.
        add_to_hist_matrix : boolean, default=False
            Whether to use the feature in as historical data, meaning that data
            points from previous time steps also will be included in the input
            matrix. If set to True, only the current data point will be used as
            input.

        """

        self.df[name] = feature_col

        if add_to_hist_matrix:
            self.added_hist_features.append(name)
        else:
            self.added_features.append(name)

        print("Feature added: {}".format(name))

    def _split_sequences(self):
        """Wrapper function for splitting the input data into sequences. The
        point of this function is to accomodate for the possibility of
        splitting the data set into seasons, and fitting a model for each
        season.

        """

        self.X_train_pre_seq = self.X_train.copy()
        self.X_test_pre_seq = self.X_test.copy()
        # Combine data
        self.train_data = np.hstack((self.y_train, self.X_train))
        self.test_data = np.hstack((self.y_test, self.X_test))

        self.X_train, self.y_train = split_sequences(self.train_data, self.hist_size)
        self.X_test, self.y_test = split_sequences(self.test_data, self.hist_size)

    def _split_train_test(self, test_equals_train=False):
        """
        Splitting the data set into training and test set, based on the
        train_split ratio.

        Parameters
        ----------
        test_equals_train : boolean, default=False
            Setting test set to be the same as training set, for debugging
            purposes.

        """

        self.train_hours = int(self.data.shape[0] * self.train_split)

        self.train_data = self.data[: self.train_hours, :]
        self.train_indeces = self.index[: self.train_hours]
        self.test_data = self.data[self.train_hours :, :]
        self.test_indeces = self.index[self.train_hours :]

        if test_equals_train:
            print("WARNING: TRAINING SET IS USED AS TEST SET")
            self.test_data = self.train_data
            self.test_indeces = self.train_indeces

        # Split data in inputs and targets
        self.X_train = self.train_data[:, 1:]
        self.X_test = self.test_data[:, 1:]
        self.y_train = self.train_data[:, 0].reshape(-1, 1)
        self.y_test = self.test_data[:, 0].reshape(-1, 1)

    def _scale_data(self, scaler_type="minmax"):
        """
        Scaling the input data.

        Default behaviour is to create a scaler based on the training data, and
        then scale the test set with this scaler. If a scaler object has been
        loaded, by using the function set_scaler(), then the dataset will be
        scaled using the loaded scaler.

        Parameters
        ----------
        scaler_type : string, default='minmax'
            What type of scaling to perform, Not applicable if a scaler object
            is loaded.

        """

        if self.scale:

            if self.scaler_loaded:

                try:
                    self.X_train = self.X_scaler.transform(self.X_train)
                except:
                    pass

                self.X_test = self.X_scaler.transform(self.X_test)

                if self.verbose:
                    print("Loaded scaler used to scale data.")

            else:
                if scaler_type == "standard":
                    self.X_scaler = StandardScaler()
                elif scaler_type == "robust":
                    self.X_scaler = RobustScaler()
                else:
                    self.X_scaler = MinMaxScaler()

                self.X_train = self.X_scaler.fit_transform(self.X_train)
                self.X_test = self.X_scaler.transform(self.X_test)

                # Save the scaler in order to reuse it on other test sets
                pickle.dump(
                    self.X_scaler,
                    open(self.result_dir + self.time_id + "-scaler.pkl", "wb"),
                )

                if self.verbose:
                    print("Data scaled and scaler saved.")

    def set_scaler(self, scaler_file):
        """
        Loading a saved scaler object from a previous data preprocessing.
        Useful when testing a model on new data.
        """

        self.X_scaler = pickle.load(open(scaler_file, "rb"))
        self.scaler_loaded = True

        if self.verbose:
            print("Scaler loaded : {}".format(scaler_file))

    def augment_data(self, thresh=20):
        """
        Augment data from certain periods.

        Parameters
        ----------
        thresh : int
            If a target vector contains a value above this threshold, the data
            point will be duplicated.

        """

        print("Augmenting data...")
        original_num_points = self.y_train.shape[0]

        for i in range(len(self.X_train[0])):
            if np.max(self.y_train[i, :]) > thresh:
                self.y_train = np.vstack((self.y_train, self.y_train[i, :]))
                self.X_train[0] = np.concatenate(
                    (self.X_train[0], np.array([self.X_train[0][i]]))
                )
                self.X_train[1] = np.concatenate(
                    (self.X_train[1], np.array([self.X_train[1][i]]))
                )
                self.X_train[2] = np.concatenate(
                    (self.X_train[2], np.array([self.X_train[2][i]]))
                )

        new_num_points = self.y_train.shape[0]

        print("Original number of points: {}.".format(original_num_points))
        print("New number of points: {}.".format(new_num_points))
        print(
            "Data augmented with {} points.".format(
                new_num_points - original_num_points
            )
        )


if __name__ == "__main__":

    data = Dataset()
    data.preprocess()
