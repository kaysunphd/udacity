"""
Units test with conftest data
"""

import os
import pickle
import pytest
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from src.ml.data import process_data


config_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../"))
with open(os.path.join(config_path, "config.yaml"), "r") as fp:
    config = yaml.safe_load(fp)


@pytest.fixture(scope='session')
def data():
    """
    Get source data

    Return
    ------
    df: pd.DataFrame
        Loaded clean data
    """
    filepath = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../data"))
    df = pd.read_csv(os.path.join(filepath, "cleaned_census.csv"))

    return df


@pytest.fixture(scope='session')
def trained_model():
    """
    Get trained model

    Return
    ------
    model: sklearn model
        Loaded model
    """

    filepath = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../model"))
    filename = os.path.join(filepath, "trained_model.pkl")

    with open(filename, 'rb') as fp:
        model = pickle.load(fp)

    return model


@pytest.fixture(scope='session')
def trained_encoder():
    """
    Get encoder

    Return
    ------
    model: sklearn encoder
        Loaded encoder
    """

    filepath = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../model"))
    filename = os.path.join(filepath, "trained_encoder.pkl")

    with open(filename, 'rb') as fp:
        encoder = pickle.load(fp)

    return encoder


@pytest.fixture(scope='session')
def trained_lb():
    """
    Get labelbinzarier

    Return
    ------
    model: sklearn label binarizer
        Loaded binarizer
    """

    filepath = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../model"))
    filename = os.path.join(filepath, "trained_lb.pkl")

    with open(filename, 'rb') as fp:
        lb = pickle.load(fp)

    return lb


@pytest.fixture(scope='session')
def cat_features():
    """
    Get categorical features of dataset

    Return
    ------
    cat_features: list
                List of categorical feature names
    """

    cat_features = config['data']['cat_features']

    return cat_features


@pytest.fixture(scope='session')
def test_data(data, cat_features, trained_encoder, trained_lb):
    """
    Get trained model

    Input
    ------
    data: pd.DataFrame
        Source dataset
    cat_features: list of strings
        List of categorical features names

    Return
    ------
    model: sklearn model
        Loaded model
    """

    # train_test_split
    _, test = train_test_split(data, test_size=0.20, random_state=123, stratify=data['salary'])

    # process data to create test dataset
    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=trained_encoder,
        lb=trained_lb
    )

    return X_test, y_test


@pytest.fixture(scope='session')
def sample_data():
    """
    Sample data

    Return
    ------
    df: pd.DataFrame
        Loaded clean data
    """
    test_sample = {
            "age": 39,
            "workclass": "State-gov",
            "fnlgt": 77516,
            "education": "Bachelors",
            "marital_status": "Never-married",
            "occupation": "Adm-clerical",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Male",
            "hours_per_week": 40,
            "native_country": "United-States"
            }
            
    df_test_sample = pd.DataFrame([test_sample])
    
    return df_test_sample