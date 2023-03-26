import numpy as np
import pandas as pd
import io
from io import StringIO
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import flask
from flask import Flask
import os
import joblib
import sklearn
import csv

app = Flask(__name__)
# np.set_printoptions(threshold=sys.maxsize)
# np.set_printoptions(precision=6)
featurizer = None
MODEL_PATH = "/opt/ml/model"


def load_model():
    global featurizer
    global MODEL_PATH

    ft_model_path = os.path.join(MODEL_PATH, "preprocess.joblib")
    with open(ft_model_path, "rb") as f:
        featurizer = joblib.load(f)
        print(f"featurizer model loaded", flush=True)
    # print(f"Model loaded", flush=True)
    return featurizer


# sagemaker inference.py script
def transform_fn(request_body, request_content_type):
    global featurizer
    """
    A function that transforms a request body into a usable
    numpy array to be used in our model.
    """
    # Since we get a headerless CSV file, we specify the column names here.
    feature_columns_names = [
        "sex",
        "length",
        "diameter",
        "height",
        "whole_weight",
        "shucked_weight",
        "viscera_weight",
        "shell_weight",
    ]
    label_column = "rings"

    if request_content_type == "text/csv":
        if not isinstance(
            featurizer, sklearn.compose._column_transformer.ColumnTransformer
        ):
            # print(f"calling load model")
            featurizer = load_model()
            print(type(featurizer), flush=True)
        # print(f"Inside transform_fn", flush=True)
        df = pd.read_csv(StringIO(request_body), header=None)

        # print(f"Received DF", flush=True)
        # print(df, flush=True)
        if len(df.columns) == len(feature_columns_names) + 1:
            # This is a labelled example, includes the ring label
            # print(f"Labelled")
            df.columns = feature_columns_names + [label_column]
        elif len(df.columns) == len(feature_columns_names):
            # This is an unlabelled example.
            # print(f"Unlabelled")
            df.columns = feature_columns_names

        data = featurizer.transform(df)
        print(f"Data after transform: {np.squeeze(data)}", flush=True)
        print(f"Data type: {type(data)}", flush=True)
        print(f"Data shape: {data.shape}", flush=True)
        return data
    else:
        raise ValueError("Unsupported content type: {}".format(request_content_type))


@app.route("/ping", methods=["GET"])
def ping():
    model = load_model()
    if isinstance(model, sklearn.compose._column_transformer.ColumnTransformer):
        print(f"Model loaded", flush=True)
        status = 200
    else:
        status = 500
    return flask.Response(response="\n", status=status, mimetype="application/json")


@app.route("/invocations", methods=["POST"])
def invocations():
    global featurizer
    print(f"Inside invocations")
    # Convert from JSON to dict
    if flask.request.content_type == "text/csv":
        input = flask.request.data.decode("utf-8")
        print(f"Input: {input}", flush=True)
        transformed_data = transform_fn(input, flask.request.content_type)
        if isinstance(transformed_data, np.ndarray):
            # transformed_data = transformed_data.tolist()
            print(f"Received ndarray", flush=True)
        else:
            print(f"Transformed Data: {transformed_data}", flush=True)
            print(type(transformed_data), flush=True)

        csv_buffer = io.StringIO()
        csv_writer = csv.writer(csv_buffer)

        for row in transformed_data:
            csv_writer.writerow(row)

        csv_buffer.seek(0)
        # out = StringIO()
        # pd.DataFrame({"data": transformed_data.tolist()}).to_csv(
        #     out, header=False, index=False
        # )
        # pd.DataFrame({"data": transformed_data.tolist()}).to_csv(
        #     out, header=False, index=False
        # )
        result = csv_buffer.getvalue()
        print(f"Result: {result}", flush=True)
        return flask.Response(response=csv_buffer, status=200, mimetype="text/csv")
    else:
        return flask.Response(
            response="This predictor only supports CSV data",
            status=415,
            mimetype="text/plain",
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
