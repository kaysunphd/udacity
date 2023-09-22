"""
Query live API
Author: Kay Sun
Date: September 21 2023
"""

import os
import requests

url = "https://ml-deploy-csft.onrender.com/inference_labels"

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

response = requests.post(url, json=test_sample)

print("status code", response.status_code)
print("output body", response.json())
