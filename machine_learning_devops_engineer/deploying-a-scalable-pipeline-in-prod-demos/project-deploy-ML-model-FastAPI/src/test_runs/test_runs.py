#!/usr/bin/env python
"""
Perform unit tests on pipeline
Author: Kay Sun
Date: September 20 2023
"""

import os
import argparse
import logging
import yaml
from src.ml.model import inference, compute_model_metrics, make_inference, make_inference_labels


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

filepath = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../"))
with open(os.path.join(filepath, "config.yaml"), "r") as fp:
    config = yaml.safe_load(fp)


def test_column_names(data):
    """
    Check if categorical features in process_data is in cleaned data
    """
    cat_features = config['data']['cat_features']

    column_names = data.columns
    check =  all(name in column_names for name in cat_features)
    assert check is True


def test_inference(trained_model, test_data):
    """
    Check inference of trained model
    """
    
    X_test, y_test = test_data

    try:
        predictions = inference(trained_model, X_test)
    except RuntimeError as err:
        logger.error("Inference failed, {err}")
        raise err


def test_compute_metrics(trained_model, test_data):
    """
    Check compute of metrics
    """

    X_test, y_test = test_data
    predictions = trained_model.predict(X_test)

    try:
        precise, recall, fbeta = compute_model_metrics(y_test, predictions)
    except Exception as err:
        logger.error("Performance metrics calculations failed, {err}")
        raise err


def test_make_inference(data, cat_features):
    """
    Check inference of trained model from raw data
    """
    data = data.drop(["salary"], axis=1)

    try:
        predictions = make_inference(data, cat_features)
    except RuntimeError as err:
        logger.error("Inference failed, {err}")
        raise err


def test_make_inference_sample_data(sample_data, cat_features):
    """
    Check inference of trained model from raw data
    """

    try:
        predictions = make_inference_labels(sample_data, cat_features)
    except RuntimeError as err:
        logger.error("Inference failed, {err}")
        raise err
