#!/usr/bin/env python
"""
Performs training on Random Forest Classifier model with GridSearch
Author: Kay Sun
Date: September 2023
"""

import os
import pickle
import argparse
import logging
import wandb
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from src.ml.data import load_data, process_data
from src.ml.model import train_model, inference, compute_model_metrics, compute_model_metrics_on_slices


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

filepath = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../"))
with open(os.path.join(filepath, "config.yaml"), "r") as fp:
    config = yaml.safe_load(fp)


def go(args):

    run = wandb.init(job_type="train_model")
    run.config.update(args)

    # Add code to load in the data.
    logger.info("Load data")
    data = load_data(args.input_artifact)

    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    logger.info("Split train and test data")
    train, test = train_test_split(data, test_size=0.20, random_state=123, stratify=data['salary'])

    logger.info("Process data")
    cat_features = config['data']['cat_features']

    X_train, y_train, encoder, lb = process_data(
        train,
        categorical_features=cat_features,
        label="salary",
        training=True
    )

    # Train and save a model.
    logger.info("Train and save model")
    model = train_model(X_train, y_train)

    output_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../model"))

    model_filename = os.path.join(output_path, "trained_model.pkl")
    with open(model_filename, 'wb') as fp:
        pickle.dump(model, fp)

    # Save encoder
    encoder_filename = os.path.join(output_path, "trained_encoder.pkl")
    with open(encoder_filename, 'wb') as fp:
        pickle.dump(encoder, fp)

    # Save labelbinarizer
    lb_filename = os.path.join(output_path, "trained_lb.pkl")
    with open(lb_filename, 'wb') as fp:
        pickle.dump(lb, fp)

    # Load saved mode, encoder, lb
    logger.info("Inference on test data")
    with open(model_filename, 'rb') as fp:
        model = pickle.load(fp)

    with open(encoder_filename, 'rb') as fp:
        encoder = pickle.load(fp)

    with open(lb_filename, 'rb') as fp:
        lb = pickle.load(fp)

    # Proces the test data with the process_data function.
    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb
    )

    # Make inference on test set and compute metrics
    predictions = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, predictions)
    logger.info("Metrics of test data:\
                precision %f,\
                recall %f,\
                fbeta %f"\
                % (precision, recall, fbeta))

    run.summary['precision'] = precision
    run.summary['recall'] = recall
    run.summary['fbeta'] = fbeta

    # Compute metrics for slices of categorical feature
    df_feature_metrics = pd.DataFrame(columns=['Feature',
                                                'Value',
                                                'Precision',
                                                'Recall',
                                                'F-beta'])
    for feature in cat_features:
        df_feature_metrics = pd.concat([df_feature_metrics,
                                        compute_model_metrics_on_slices(test,
                                                                        y_test,
                                                                        predictions,
                                                                        feature)])

    df_feature_metrics.to_csv(os.path.join(output_path, "slice_output.txt"), index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="This step trains the model")


    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="Name of cleaned data",
        required=True
    )

    args = parser.parse_args()

    go(args)
