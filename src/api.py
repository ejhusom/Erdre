#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""REST API for Erdre.

Author:
    Erik Johannes Husom

Created:
    2021-11-10 Wednesday 11:22:39

"""
import flask
from flask_restful import Resource, Api, reqparse
import pandas as pd

app = flask.Flask(__name__)
api = Api(app)
app.config["DEBUG"] = True

class VirtualSensors(Resource):

    def get(self):

        try:
            data = pd.read_csv("virtual_sensors.csv")
            data = data.to_dict()
            return {"data": data}, 200
        except:
            return {
                    "message": "No virtual sensors exist."
            }, 401

    def post(self):
        parser = reqparse.RequestParser()

        parser.add_argument("datasetName", required=True)
        parser.add_argument("targetVariable", required=True)

        args = parser.parse_args()

        new_data = pd.DataFrame([{
            "datasetName": args["datasetName"],
            "targetVariable": args["targetVariable"]
        }])

        try:
            data = pd.read_csv("virtual_sensors.csv")

            if 
            data = data.append(new_data, ignore_index=True)
        except:
            # If the file does not exists already, the new data is the only
            # data.
            data = new_data

        data.to_csv("virtual_sensors.csv", index=False)

        return {"data": data.to_dict()}, 200


if __name__ == '__main__': 

    api.add_resource(VirtualSensors, "/virtual_sensors")
    app.run()


