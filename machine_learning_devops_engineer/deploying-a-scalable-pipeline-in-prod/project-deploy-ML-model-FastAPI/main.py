"""
RESTful API using FastAPI
Author: Kay Sun
Date: September 20 2023
"""

import os
import yaml
import pandas as pd
from fastapi import FastAPI
from src.ml.model import make_inference, make_inference_labels
from src.ml.data import InputData


filepath = os.path.normpath(os.path.join(os.path.dirname(__file__), "./"))
with open(os.path.join(filepath, "config.yaml"), "r") as fp:
    config = yaml.safe_load(fp)


app = FastAPI()

@app.get("/")
async def welcome():
    """
    GET with welcome message

    Return
    ------
    text string
    """
    return "Greetings! Welcome to our model deployment"


@app.post("/inference")
async def inference(input_data: InputData):
    """
    POST with inference

    Return
    ------
    JSON: predictions
    """
    df_input_data = pd.DataFrame([input_data.model_dump()])

    cat_features = config['data']['cat_features']

    predictions = make_inference(df_input_data, cat_features)

    return {"prediction": predictions}


@app.post("/inference_labels")
async def inference_labels(input_data: InputData):
    """
    POST with inference labels

    Return
    ------
    JSON: predictions
    """
    df_input_data = pd.DataFrame([input_data.model_dump()])

    cat_features = config['data']['cat_features']

    predict_labels = make_inference_labels(df_input_data, cat_features)

    return {"prediction": predict_labels}
