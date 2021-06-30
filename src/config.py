#!/usr/bin/env python3
"""Global parameters for project.

Example:

    >>> from config import *
    >>> file = DATA_PATH / "filename.txt"

Author:   Erik Johannes Husom
Created:  2020-09-16

"""

from pathlib import Path


ASSETS_PATH = Path("./assets")
"""Path to all assets of project."""

PROFILE_PATH = ASSETS_PATH / "profile"
"""Path to profiling report."""

DATA_PATH = ASSETS_PATH / "data"
"""Path to data."""

DATA_PATH_RAW = DATA_PATH / "raw"
"""Path to raw data."""

DATA_FEATURIZED_PATH = DATA_PATH / "featurized"
"""Path to data that is has added features."""

DATA_CLEANED_PATH = DATA_PATH / "cleaned"
"""Path to data that is cleaned."""

DATA_SEQUENTIALIZED_PATH = DATA_PATH / "sequentialized"
"""Path to data that is split into sequences."""

DATA_SPLIT_PATH = DATA_PATH / "split"
"""Path to data that is split into train and test set."""

DATA_SCALED_PATH = DATA_PATH / "scaled"
"""Path to scaled data."""

DATA_COMBINED_PATH = DATA_PATH / "combined"
"""Path to combined data, ready for training."""

MODELS_PATH = ASSETS_PATH / "models"
"""Path to models."""

MODELS_FILE_PATH = MODELS_PATH / "model.h5"
"""Path to model file."""

METRICS_PATH = ASSETS_PATH / "metrics"
"""Path to folder containing metrics file."""

METRICS_FILE_PATH = METRICS_PATH / "metrics.json"
"""Path to file containing metrics."""

PREDICTIONS_PATH = ASSETS_PATH / "predictions"
"""Path to folder containing predictions file."""

PREDICTIONS_FILE_PATH = PREDICTIONS_PATH / "predictions.csv"
"""Path to file containing predictions."""

PLOTS_PATH = ASSETS_PATH / "plots"
"""Path to folder plots."""

PREDICTION_PLOT_PATH = PLOTS_PATH / "prediction.png"
"""Path to file containing prediction plot."""

INTERVALS_PLOT_PATH = PLOTS_PATH / "intervals.png"
"""Path to file containing intervals plot."""

TRAININGLOSS_PLOT_PATH = PLOTS_PATH / "trainingloss.png"
"""Path to file containing training loss plot."""
