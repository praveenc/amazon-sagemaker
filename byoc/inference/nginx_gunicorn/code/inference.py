# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function
from __future__ import absolute_import

from io import StringIO
import joblib
import os
import pickle
import flask
import pandas as pd
import xgboost as xgb
import numpy as np

from sagemaker_inference import content_types, errors
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


prefix = "/opt/ml/"
model_path = os.path.join(prefix, "model")


# A singleton for holding the model. This simply loads the model and holds it.
class MyAbaloneInference(object):
    VALID_CONTENT_TYPES = (content_types.CSV, content_types.NPY)
    model = None  # Where we keep the model when it's loaded.

    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def get_model(cls):
        """Load the model. Called once when the service starts."""
        if cls.model == None:
            print(f"Model path: {model_path}")
            model_file = os.path.join(model_path, "xgboost-model")
            if os.path.exists(model_file):
                print(f"Loading model from {model_file}")
                model = pickle.load(open(model_file, "rb"))
                cls.model = model
                print(f"Model loaded")
                return model
            else:
                errors.GenericInferenceToolkitError(
                    f"Model file {model_file} not found."
                )

    @classmethod
    def transform_input(cls, input_data, content_type):
        """Preprocess the input data."""
        print(f"Content Type: {content_type}")

        # Since we get a headerless CSV file we specify the column names here.
        feature_columns_names = [
            "sex",  # M, F, and I (infant)
            "length",  # Longest shell measurement
            "diameter",  # perpendicular to length
            "height",  # with meat in shell
            "whole_weight",  # whole abalone
            "shucked_weight",  # weight of meat
            "viscera_weight",  # gut weight (after bleeding)
            "shell_weight",
        ]  # after being dried
        label_column = "rings"
        feature_columns_dtype = {
            "sex": str,
            "length": np.float64,
            "diameter": np.float64,
            "height": np.float64,
            "whole_weight": np.float64,
            "shucked_weight": np.float64,
            "viscera_weight": np.float64,
            "shell_weight": np.float64,
        }
        label_column_dtype = {"rings": np.float64}

        def merge_two_dicts(x, y):
            z = x.copy()
            z.update(y)
            return z

        if content_type == "text/csv":
            # preprocess = joblib.load(os.path.join(model_path, "preprocess.joblib"))
            # print(f"Loaded preprocess.joblib!")
            df = pd.read_csv(
                StringIO(input_data),
                header=None,
                dtype=merge_two_dicts(feature_columns_dtype, label_column_dtype),
            )
            if len(df.columns) == len(feature_columns_names) + 1:
                # This is a labelled example, includes the ring label
                df.columns = feature_columns_names + [label_column]
            elif len(df.columns) == len(feature_columns_names):
                # This is an unlabelled example.
                df.columns = feature_columns_names

            print("Invoked with {} records".format(df.shape[0]))

            print(f"Received DF:")
            print(df)

            numeric_features = [i for i, x in enumerate(df.dtypes) if x != object]
            categorical_features = [i for i, x in enumerate(df.dtypes) if x == object]

            numeric_transformer = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="mean")),
                    ("scaler", StandardScaler()),
                ]
            )

            # categorical_features = ["sex"]
            categorical_transformer = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="constant", fill_value="missing"),
                    ),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numeric_transformer, numeric_features),
                    ("cat", categorical_transformer, categorical_features),
                ]
            )

            X = preprocessor.fit_transform(df)
            # print(f"X: {X}")
            # X = preprocess.transform(df)
            data = xgb.DMatrix(data=X)
            return data
        else:
            errors.UnsupportedFormatError(
                f"Content type {content_type} is not supported by this script.\ncontent_type must be one of {cls.VALID_CONTENT_TYPES}"
            )

    @classmethod
    def predict(cls, input_data):
        """Preprocess input data, do a prediction, and postprocess the result."""
        print(f"Inside predict func. class method")
        print(input_data)
        print(cls)
        model = cls.get_model()
        if isinstance(model, xgb.core.Booster):
            print(f"model instance verified")
            predictions = model.predict(input_data)
        else:
            print(f"using cls.model directly")
            predictions = cls.model.predict(input_data)
        return predictions


# The flask app for serving predictions
app = flask.Flask(__name__)


@app.route("/ping", methods=["GET"])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = MyAbaloneInference.get_model() is not None

    status = 200 if health else 404
    return flask.Response(response="\n", status=status, mimetype="application/json")


@app.route("/invocations", methods=["POST"])
def predictions():
    """Do an inference on a single batch of data. In this sample server, we take data as JSON, convert
    it to a list of dictionaries for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    data = None
    # Convert from JSON to dict
    if flask.request.content_type == "text/csv":
        input = flask.request.data.decode("utf-8")
        print(f"Input: {input}")
        transformed_input = MyAbaloneInference.transform_input(
            input, flask.request.content_type
        )
        model = MyAbaloneInference.get_model()
        if isinstance(model, xgb.core.Booster):
            print(type(model))
            print(f"Model loaded for prediction")
            predictions = model.predict(transformed_input)
        else:
            print(f"Trying with class method")
            predictions = MyAbaloneInference.predict(transformed_input)

        print(f"Predictions obj")
        print(type(predictions))
        if isinstance(predictions, np.ndarray):
            predictions = predictions.tolist()
        out = StringIO()
        pd.DataFrame({"results": predictions}).to_csv(out, header=False, index=False)
        result = out.getvalue()
    else:
        return flask.Response(
            response="This predictor only supports CSV data",
            status=415,
            mimetype="text/plain",
        )

    return flask.Response(response=result, status=200, mimetype="text/csv")
