from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from datetime import datetime
from uuid import uuid4
# from rich import print
import boto3
from sagemaker import Session, get_execution_role
from sagemaker.s3 import S3Uploader, s3_path_join

session = Session()
region = session._region_name
account_id = boto3.client('sts').get_caller_identity().get('Account')
bucket = session.default_bucket()
role = get_execution_role()
prefix = "sagemaker/abalone"

base_s3uri = s3_path_join(f"s3://{bucket}", prefix)
model_base_s3uri = s3_path_join(base_s3uri, "models")
data_s3uri = s3_path_join(base_s3uri, "data")

train_input = s3_path_join(data_s3uri, "train/train.csv")
val_input = s3_path_join(data_s3uri, "validation/validation.csv")
model_output_s3uri = s3_path_join(base_s3uri, "temp")

train_image_uri = f"{account_id}.dkr.ecr.{region}.amazonaws.com/abalone-train:latest"

suffix = f"{str(uuid4())[:5]}-{datetime.now().strftime('%d%b%Y')}"
train_job_name = f"abalone-train-{suffix}"

print(
    f"Lauching train job with docker image: {train_image_uri} and name: {train_job_name}"
)

print(f"Model artifacts written to: {model_output_s3uri}")

xgb_estimator = Estimator(
    image_uri=train_image_uri,
    role=role,
    instance_count=1,
    instance_type="ml.m5.xlarge",
    output_path=model_output_s3uri,
    sagemaker_session=session,
)

xgb_estimator.set_hyperparameters(
    objective="reg:squarederror",
    learning_rate=0.01,
    num_round=150,
    max_depth=5,
    eta=0.2,
    gamma=4,
    min_child_weight=6,
    subsample=0.7,
)

print(f"Calling fit on generic Estimator ...")

xgb_estimator.fit(
    inputs={
        "train": TrainingInput(s3_data=train_input, content_type="text/csv"),
        "validation": TrainingInput(s3_data=val_input, content_type="text/csv"),
    },
    job_name=train_job_name,
)
