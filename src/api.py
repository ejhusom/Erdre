#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""REST API for Erdre.

Author:
    Erik Johannes Husom

Created:
    2021-11-10 Wednesday 11:22:39

"""
import os
import json
import time
import subprocess
import urllib.request
from pathlib import Path

import flask
import numpy as np
import pandas as pd
import plotly
import plotly.graph_objs as go
import requests
import yaml
from flask_restful import Api, Resource, reqparse
from plotly.subplots import make_subplots

from config import METRICS_FILE_PATH
from evaluate import plot_prediction
from virtualsensor import VirtualSensor

app = flask.Flask(__name__)
api = Api(app)
app.config["DEBUG"] = True

@app.route("/")
def home():
    return flask.render_template("index.html")

@app.route("/virtual_sensors")
def virtual_sensors():

    virtual_sensors = get_virtual_sensors()

    return flask.render_template(
            "virtual_sensors.html",
            length=len(virtual_sensors),
            virtual_sensors=virtual_sensors
    )

@app.route("/inference")
def inference():

    virtual_sensors = get_virtual_sensors()

    return flask.render_template(
            "inference.html",
            virtual_sensors=virtual_sensors
    )

@app.route("/result")
def result(plot_div):
    # plot_div = session["plot_div"]
    return flask.render_template("result.html",
            plot=flask.Markup(plot_div))

@app.route("/prediction")
def prediction():
    return flask.render_template("prediction.html")

def get_virtual_sensors():

    try:
        virtual_sensors = json.load(open("virtual_sensors.json"))
    except:
        virtual_sensors = {}

    return virtual_sensors


class CreateVirtualSensor(Resource):
    def get(self):

        try:
            virtual_sensors = json.load(open("virtual_sensors.json"))
            return virtual_sensors, 200
        except:
            return {"message": "No virtual sensors exist."}, 401

    def post(self):


        try:
            # Read params file
            params_file = flask.request.files["file"]
            params = yaml.safe_load(params_file)
        except:
            params = yaml.safe_load(open("params_default.yaml"))
            params["profile"]["dataset"] = flask.request.form["dataset"]
            params["clean"]["target"]= flask.request.form["target"]
            params["train"]["learning_method"]= flask.request.form["learning_method"]
            params["split"]["train_split"] = float(flask.request.form["train_split"]) / 10
            print(params)

        # Create dict containing all metadata about virtual sensor
        virtual_sensor_metadata = {}
        # The ID of the virtual sensor is set to the current Unix time for
        # uniqueness.
        virtual_sensor_id = int(time.time())
        virtual_sensor_metadata["id"] = virtual_sensor_id
        virtual_sensor_metadata["params"] = params

        # Save params to be used by DVC when creating virtual sensor.
        yaml.dump(params, open("params.yaml", "w"), allow_unicode=True)

        # Run DVC to create virtual sensor.
        subprocess.run(["dvc", "repro"])

        metrics = json.load(open(METRICS_FILE_PATH))
        virtual_sensor_metadata["metrics"] = metrics

        try:
            virtual_sensors = json.load(open("virtual_sensors.json"))
        except:
            virtual_sensors = {}

        virtual_sensors[virtual_sensor_id] = virtual_sensor_metadata
        print(virtual_sensors)

        json.dump(virtual_sensors, open("virtual_sensors.json", "w+"))

        return flask.redirect("virtual_sensors")


class Infer(Resource):
    def get(self):
        return 200

    def post(self):

        virtual_sensor_id = flask.request.form["id"]
        csv_file = flask.request.files["file"]
        inference_df = pd.read_csv(csv_file)

        virtual_sensors = get_virtual_sensors()
        virtual_sensor = virtual_sensors[virtual_sensor_id]
        params = virtual_sensor["params"]

        vs = VirtualSensor(params_file=params)

        # Run DVC to fetch correct assets.
        subprocess.run(["dvc", "repro"])

        y_pred = vs.run_virtual_sensor(inference_df=inference_df)
        window_size = params["sequentialize"]["window_size"]
        original_target_values = inference_df[params["clean"]["target"]][::window_size]
        print(y_pred.shape)
        print(original_target_values.shape)

        x = np.linspace(0, y_pred.shape[0] - 1, y_pred.shape[0])
        x_orig = np.linspace(0, original_target_values.shape[0] - 1,
                original_target_values.shape[0])
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        if len(original_target_values.shape) > 1:
            original_target_values = original_target_values[:, -1].reshape(-1)
        if len(y_pred.shape) > 1:
            y_pred = y_pred[:, -1].reshape(-1)

        fig.add_trace(
            go.Scatter(x=x_orig, y=original_target_values, name="original"),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(x=x, y=y_pred, name="pred"),
            secondary_y=False,
        )
        fig.update_layout(title_text="Original vs predictions")
        fig.update_xaxes(title_text="time step")
        fig.update_yaxes(title_text="target variable", secondary_y=False)
        fig.write_html("src/templates/prediction.html")

        # plot_div = plotly.offline.plot([
        #     go.Scatter(
        #         x=np.linspace(0, len(y_pred)),
        #         y=y_pred
        #     )],
        #     output_type="div",
        #     include_plotlyjs=True
        # )
        # print(plot_div)

        # flask.session["plot_div"] = plot_div

        # return flask.redirect(flask.url_for("result", plot_div=plot_div))
        return flask.redirect("prediction")
        # return flask.render_template("inference.html",
        #         virtual_sensors=virtual_sensors, plot_div=plot_div)


if __name__ == "__main__":

    api.add_resource(CreateVirtualSensor, "/create_virtual_sensor")
    api.add_resource(Infer, "/infer")
    app.run()
