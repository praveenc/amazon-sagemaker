from __future__ import absolute_import

import os
import sys
import time

import argparse
import logging
import pickle as pkl

import numpy as np
import pandas as pd
import xgboost as xgb

from sagemaker_training import environment


from utils import (
    ExitSignalHandler,
    load_json_object,
    print_files_in_path,
    print_json_object,
    save_model_artifacts,
    write_failure_file,
)

hyperparameters_file_path = "/opt/ml/input/config/hyperparameters.json"
inputdataconfig_file_path = "/opt/ml/input/config/inputdataconfig.json"
resource_file_path = "/opt/ml/input/config/resourceconfig.json"
data_files_path = "/opt/ml/input/data/"
failure_file_path = "/opt/ml/output/failure"
model_artifacts_path = "/opt/ml/model/"

training_job_name_env = "TRAINING_JOB_NAME"
training_job_arn_env = "TRAINING_JOB_ARN"


logging.basicConfig(level=logging.INFO)

TRAIN_VALIDATION_FRACTION = 0.2
RANDOM_STATE_SAMPLING = 200

def prepare_data(train_dir, validation_dir):
    """Read data from train and validation channel, and return predicting features and target variables.

    Args:
        data_dir (str): directory which saves the training data.

    Returns:
        Tuple of training features, training target, validation features, validation target.
    """
    df_train = pd.read_csv(
        os.path.join(train_dir, "train.csv"),
        header=None,
    )
    df_train = df_train.iloc[np.random.permutation(len(df_train))]
    df_train.columns = ["target"] + [
        f"feature_{x}" for x in range(df_train.shape[1] - 1)
    ]

    try:
        df_validation = pd.read_csv(
            os.path.join(validation_dir, "validation.csv"),
            header=None,
        )
        df_validation.columns = ["target"] + [
            f"feature_{x}" for x in range(df_validation.shape[1] - 1)
        ]

    except FileNotFoundError:  # when validation data is not available in the directory
        logging.info(
            f"Validation data is not found. {TRAIN_VALIDATION_FRACTION * 100}% of training data is "
            f"randomly selected as validation data. The seed for random sampling is {RANDOM_STATE_SAMPLING}."
        )
        df_validation = df_train.sample(
            frac=TRAIN_VALIDATION_FRACTION,
            random_state=RANDOM_STATE_SAMPLING,
        )
        df_train.drop(df_validation.index, inplace=True)
        df_validation.reset_index(drop=True, inplace=True)
        df_train.reset_index(drop=True, inplace=True)

    X_train, y_train = df_train.iloc[:, 1:], df_train.iloc[:, :1]
    X_val, y_val = df_validation.iloc[:, 1:], df_validation.iloc[:, :1]

    return X_train.values, y_train.values, X_val.values, y_val.values



def train():
    try:
        print("\nRunning training...")

        if os.path.exists(hyperparameters_file_path):
            hyperparameters = load_json_object(hyperparameters_file_path)
            print("\nHyperparameters configuration:")
            print_json_object(hyperparameters)

        if os.path.exists(inputdataconfig_file_path):
            input_data_config = load_json_object(inputdataconfig_file_path)
            print("\nInput data configuration:")
            print_json_object(input_data_config)

            for key in input_data_config:
                print("\nList of files in {0} channel: ".format(key))
                channel_path = data_files_path + key + "/"
                print_files_in_path(channel_path)

        if os.path.exists(resource_file_path):
            resource_config = load_json_object(resource_file_path)
            print("\nResource configuration:")
            print_json_object(resource_config)

        if training_job_name_env in os.environ:
            print("\nTraining job name: ")
            print(os.environ[training_job_name_env])

        if training_job_arn_env in os.environ:
            print("\nTraining job ARN: ")
            print(os.environ[training_job_arn_env])

        # This object is used to handle SIGTERM and SIGKILL signals.
        signal_handler = ExitSignalHandler()

        env = environment.Environment()
        training_dir = env.channel_input_dirs["train"]

        # get a the hyperparameter "training_data_file" from `hyperparameters.json` file
        # file_name = env.hyperparameters["training_data_file"]

        # get the folder where the model should be saved
        model_dir = env.model_dir
        
        print(f"Training DIR: {training_dir}")
        print(f"Model DIR: {model_dir}")
        
        """Run training."""
        parser = argparse.ArgumentParser()

        parser.add_argument(
            "--max_depth",
            type=int,
            default=5
        )
        parser.add_argument("--eta", type=float)
        parser.add_argument("--gamma", type=int)
        parser.add_argument("--min_child_weight", type=int, default=6)
        parser.add_argument("--subsample", type=float, default=0.7)
        parser.add_argument("--verbosity", type=int)
        parser.add_argument("--objective", type=str, default="reg:squarederror")
        parser.add_argument("--num_round", type=int,default=hyperparameters['num_round'])
        parser.add_argument("--tree_method", type=str, default="auto")
        parser.add_argument("--predictor", type=str, default="auto")
        parser.add_argument("--learning_rate", type=float, default="0.01")
        parser.add_argument(
            "--output_data_dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR")
        )
        parser.add_argument("--model_dir", type=str, default=env.model_dir)
        parser.add_argument("--train", type=str, default=env.channel_input_dirs["train"])
        parser.add_argument(
            "--validation", type=str, default=env.channel_input_dirs["validation"]
        )
        parser.add_argument("--sm_hosts", type=str, default=os.environ.get("SM_HOSTS"))
        parser.add_argument(
            "--sm_current_host", type=str, default=os.environ.get("SM_CURRENT_HOST")
        )

        args, _ = parser.parse_known_args()

        print(f"xgboost version: {xgb.__version__}")
        print(f"numpy version: {np.__version__}")
        print(f"train dir: {args.train}")
        print(f"val dir: {args.validation}")

        X_train, y_train, X_val, y_val = prepare_data(args.train, args.validation)

        # create dataset for lightgbm
        dtrain = xgb.DMatrix(data=X_train, label=y_train)
        dval = xgb.DMatrix(data=X_val, label=y_val)
        watchlist = [(dtrain, "train"), (dval, "validation")]

        # specify your configurations as a dict
        # params = {
        #     "booster": "gbtree",
        #     "objective": args.objective,
        #     "learning_rate": args.learning_rate,
        #     "gamma": args.gamma,
        #     "min_child_weight": args.min_child_weight,
        #     "max_depth": args.max_depth,
        #     "subsample": args.subsample,
        #     "colsample_bytree": 1,
        #     "reg_lambda": 1,
        #     "reg_alpha": 0,
        #     "eval_metric": "rmse",
        # }

        bst = xgb.train(
            params=hyperparameters,
            dtrain=dtrain,
            num_boost_round=args.num_round,
            evals=watchlist,
            xgb_model=None,
        )

        model_location = os.path.join(args.model_dir, "xgboost-model")
        # model_location = args.model_dir + '/xgboost-model'args.model_dir + '/xgboost-model'
        bst.save_model(model_location)
        pkl.dump(bst, open(model_location, "wb"))
        logging.info("Stored trained model at {}".format(model_location))

        print("\nTraining completed!")
    except Exception as e:
        write_failure_file(failure_file_path, str(e))
        print(e, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    if sys.argv[1] == "train":
        train()
    else:
        print("Missing required argument 'train'.", file=sys.stderr)
        sys.exit(1)