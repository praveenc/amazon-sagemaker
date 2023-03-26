from io import StringIO
import os
import json
import numpy as np
import flask
import pandas as pd
import pickle
import xgboost as xgb
from flask import Flask
import csv

app = Flask(__name__)
model = None
MODEL_PATH = "/opt/ml/model"


def load_model():
    global model
    global MODEL_PATH

    xgb_model_path = os.path.join(MODEL_PATH, "xgboost-model")
    with open(xgb_model_path, "rb") as f:
        model = pickle.load(f)
    print(f"xgboost model loaded", flush=True)
    return model


def preprocess(input_data, content_type):
    if content_type == "text/csv":
        # Create a StringIO object from the CSV string
        csv_buffer = StringIO(input_data)

        # Read the CSV data into a list of lists
        csv_reader = csv.reader(csv_buffer)
        data = [list(map(float, row)) for row in csv_reader]

        # Convert the list of lists to a NumPy array
        input_data = np.array(data)
        print(type(input_data), flush=True)
        data = xgb.DMatrix(data=input_data)
        print(type(data), flush=True)
        return data


def predict(input_data):
    global model
    if model is None:
        model = load_model()
        print(type(model), flush=True)
    predictions = model.predict(input_data)
    return predictions


@app.route("/ping", methods=["GET"])
def ping():
    model = load_model()
    status = 200
    if model is None:
        status = 500
    return flask.Response(response="\n", status=status, mimetype="application/json")


@app.route("/invocations", methods=["POST"])
def invocations():
    if model is None:
        load_model()

    # Convert from JSON to dict
    if flask.request.content_type == "text/csv":
        input = flask.request.data.decode("utf-8")
        print(f"Input: {input}")
        transformed_data = preprocess(input, flask.request.content_type)
        print(f"Transformed data: {transformed_data}")
        predictions = predict(transformed_data)
        if isinstance(predictions, np.ndarray):
            predictions = predictions.tolist()
        return json.dumps({"result": predictions})
    else:
        return flask.Response(
            response="This predictor only supports CSV data",
            status=415,
            mimetype="text/plain",
        )
