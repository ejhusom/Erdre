#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""REST API for Erdre.

Author:
    Erik Johannes Husom

Created:
    2021-11-10 Wednesday 11:22:39

"""
import urllib.request
from pathlib import Path

import flask
import pandas as pd
import requests
from flask_restful import Api, Resource, reqparse

from virtualsensor import VirtualSensor

app = flask.Flask(__name__)
api = Api(app)
app.config["DEBUG"] = True


class CreateVirtualSensor(Resource):
    def get(self):

        try:
            data = pd.read_csv("virtual_sensors.csv")
            data = data.to_dict()
            return {"data": data}, 200
        except:
            return {"message": "No virtual sensors exist."}, 401

    def post(self):
        parser = reqparse.RequestParser()

        parser.add_argument("datasetName", required=True)
        parser.add_argument("targetVariable", required=True)

        args = parser.parse_args()

        new_data = pd.DataFrame(
            [
                {
                    "datasetName": args["datasetName"],
                    "targetVariable": args["targetVariable"],
                }
            ]
        )

        try:
            data = pd.read_csv("virtual_sensors.csv")
            data = data.append(new_data, ignore_index=True)
        except:
            # If the file does not exists already, the new data is the only
            # data.
            data = new_data

        data.to_csv("virtual_sensors.csv", index=False)

        return {"data": data.to_dict()}, 200


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

    api.add_resource(CreateVirtualSensor, "/virtual_sensors")
    api.add_resource(Infer, "/infer")
    app.run()
