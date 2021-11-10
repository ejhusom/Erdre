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
import pandas as pd
import requests
import yaml
from flask_restful import Api, Resource, reqparse

from virtualsensor import VirtualSensor

app = flask.Flask(__name__)
api = Api(app)
app.config["DEBUG"] = True

@app.route("/")
def home():
    return flask.render_template("index.html")

@app.route("/virtual_sensors")
def virtual_sensors():

    try:
        virtual_sensors = json.load(open("virtual_sensors.json"))
    except:
        virtual_sensors = {}


    return flask.render_template(
            "virtual_sensors.html",
            length=len(virtual_sensors),
            virtual_sensors=virtual_sensors
    )

class CreateVirtualSensor(Resource):
    def get(self):

        try:
            virtual_sensors = json.load(open("virtual_sensors.json"))
            return virtual_sensors, 200
        except:
            return {"message": "No virtual sensors exist."}, 401

    def post(self):

        # Read params file
        params_file = flask.request.files["file"]
        params = yaml.safe_load(params_file)

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

        try:
            virtual_sensors = json.load(open("virtual_sensors.json"))
        except:
            virtual_sensors = {}

        virtual_sensors[virtual_sensor_id] = virtual_sensor_metadata
        print(virtual_sensors)

        json.dump(virtual_sensors, open("virtual_sensors.json", "w+"))

        return 200

        # parser = reqparse.RequestParser()

        # parser.add_argument("datasetName", required=True)
        # parser.add_argument("targetVariable", required=True)

        # args = parser.parse_args()

        # new_data = pd.DataFrame(
        #     [
        #         {
        #             "datasetName": args["datasetName"],
        #             "targetVariable": args["targetVariable"],
        #         }
        #     ]
        # )

        # try:
        #     data = pd.read_csv("virtual_sensors.csv")
        #     data = data.append(new_data, ignore_index=True)
        # except:
        #     # If the file does not exists already, the new data is the only
        #     # data.
        #     data = new_data

        # data.to_csv("virtual_sensors.csv", index=False)

        # return {"data": data.to_dict()}, 200


class Infer(Resource):
    def get(self):

        return 200

    def post(self):

        csv_file = flask.request.files["file"]
        data = pd.read_csv(csv_file)

        vs = VirtualSensor()
        vs.run_virtual_sensor(inference_df=data)

        return 200


if __name__ == "__main__":

    api.add_resource(CreateVirtualSensor, "/create_virtual_sensor")
    api.add_resource(Infer, "/infer")
    app.run()
