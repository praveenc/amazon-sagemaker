from sagemaker.estimator import Estimator

import os
import pathlib
from sagemaker.inputs import TrainingInput
from sagemaker.local import LocalSession
from sagemaker.xgboost import XGBoost, XGBoostModel
from datetime import datetime
from uuid import uuid4


local_session = LocalSession()
local_session.config = {"local": {"local_code": True}}

DATADIR = os.path.abspath("./data/abalone.csv")

dataset_uri = pathlib.Path(DATADIR).as_uri()

role = "arn:aws:iam::111111111111:role/service-role/AmazonSageMaker-ExecutionRole-20230101T000001"

print(f"Role: {role}")
print(f"Dataset URI: {dataset_uri}")

train_input = pathlib.Path(os.path.abspath("./data/train.csv")).as_uri()
val_input = pathlib.Path(os.path.abspath("./data/validation.csv")).as_uri()

model_output_uri = pathlib.Path(os.path.abspath("./models")).as_uri()

xgb_estimator = Estimator(
    image_uri="abalone-train",
    role=role,
    train_instance_count=1,
    train_instance_type="local",
    output_path=model_output_uri,
    sagemaker_session=local_session,
)

xgb_estimator.set_hyperparameters(
    objective="reg:squarederror",
    learning_rate=0.01,
    num_round=100,
    max_depth=5,
    eta=0.2,
    gamma=4,
    min_child_weight=6,
    subsample=0.7,
)

suffix = f"{str(uuid4())[:5]}-{datetime.now().strftime('%d%b%Y')}"
train_job_name = f"abalone-train-{suffix}"


xgb_estimator.fit(
    inputs={
        "train": TrainingInput(s3_data=train_input, content_type="text/csv"),
        "validation": TrainingInput(s3_data=val_input, content_type="text/csv"),
    },
    job_name=train_job_name,
)
