from io import StringIO
import numpy as np
import os
import pickle
import pandas as pd
import xgboost as xgb
import logging
import time
from sagemaker_inference import content_types, errors, decoder
import csv

prefix = "/opt/ml/"
model_path = os.path.join(prefix, "model")


class AbaloneHandler(object):
    def __init__(self):
        self.initialized = False
        self.error = None
        self._context = None
        self.xgb_model = None
        self.featurizer_model = None

        # Since we get a headerless CSV file we specify the column names here.
        self.feature_columns_names = [
            "sex",  # M, F, and I (infant)
            "length",  # Longest shell measurement
            "diameter",  # perpendicular to length
            "height",  # with meat in shell
            "whole_weight",  # whole abalone
            "shucked_weight",  # weight of meat
            "viscera_weight",  # gut weight (after bleeding)
            "shell_weight",
        ]  # after being dried

        self.label_column = "rings"

    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return: model
        """
        self.initialized = True
        self._context = context
        properties = context.system_properties
        print(f"Context Properties:")
        print(properties)

        # Contains the url parameter passed to the load request
        model_dir = properties.get("model_dir")
        gpu_id = properties.get("gpu_id")

        print(f"Model dir: {model_dir}")
        model_file_path = os.path.join(model_dir, "xgboost-model")
        # featurizer_model_path = os.path.join(model_dir, "preprocess.joblib")

        # Load XGBoost serialized model from disk
        try:
            print(f"Loading {model_file_path}")
            with open(os.path.join(model_file_path), "rb") as inp:
                self.xgb_model = pickle.load(inp)
        except Exception as e:
            logging.error(f"Error loading XGBModel {model_file_path}")
            pass

    def preprocess(self, request):
        """
        Transform raw input into model input data.
        :param request: list of raw requests
        :return: list of preprocessed model input data
        """
        # Take the input data and pre-process it make it inference ready

        print(f"INSIDE preprocess")
        for _, data in enumerate(request):
            request = data.get("body").decode("utf-8")
            print(f"model_input", flush=True)
            print(request, flush=True)
            csv_buffer = StringIO(request)
            csv_reader = csv.reader(csv_buffer)
            data = [list(map(float, row)) for row in csv_reader]
            array = np.array(data)
            data = xgb.DMatrix(data=array)
            print(type(data))
            print(data)
        return data

    def inference(self, model_input):
        """
        Internal inference methods
        :param model_input: transformed model input data list
        :return: list of inference output in NDArray
        """
        import csv

        # Do some inference call to engine here and return output
        print(f"Inside inference function")
        predictions = self.xgb_model.predict(model_input).tolist()
        print(f"Predictions: {predictions}")
        return predictions

    # def postprocess(self, inference_output, context):
    #     """
    #     Return predict result in batch.
    #     :param inference_output: list of inference output
    #     :return: list of predict results
    #     """
    #     # Take output from network and post-process to desired format
    #     print(f"Inside postprocess")
    #     out = StringIO()
    #     pd.DataFrame({"results": inference_output}).to_csv(out, header=False, index=False)
    #     result = out.getvalue()
    #     print(f"Result: {result}")
    #     request_processor = context.request_processor
    #     request_processor.report_status(200, "Prediction OK")
    #     return [{'result': result}]

    def handle(self, data, context):
        """
        Call preprocess, inference and post-process functions
        :param data: input data
        :param context: model server context
        """
        self.error = None
        print(f"Inside handle")
        request_processor = context.request_processor
        try:
            preprocess_start = time.time()
            model_input = self.preprocess(data)
            inference_start = time.time()
            model_out = self.inference(model_input)
            end_time = time.time()
            # postprocess_start = time.time()
            # output = self.postprocess(model_out, context)
            # Add cloudwatch metrics
            metrics = context.metrics
            metrics.add_time(
                "PreprocessTime", round((inference_start - preprocess_start) * 1000, 2)
            )
            metrics.add_time(
                "InferenceTime", round((end_time - inference_start) * 1000, 2)
            )
            # metrics.add_time("PostprocessTime", round((end_time - postprocess_start) * 1000, 2))
            # request_processor.report_status(200, "Prediction OK")
            return model_out
        except Exception as e:
            logging.error(e, exc_info=True)
            # request_processor.report_status(500, "Unknown inference error")
            return [str(e)]


_service = AbaloneHandler()


def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    return _service.handle(data, context)
