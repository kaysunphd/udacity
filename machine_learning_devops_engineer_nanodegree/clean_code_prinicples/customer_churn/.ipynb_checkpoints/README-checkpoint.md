# Predict Customer Churn

#### Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description

#### A manager at the bank is disturbed with more and more customers leaving their credit card services. It is important for them to identify to identify credit card customers that are most likely to churn.

#### Machine learning can be used to predict who is gonna get churned so they can proactively go to the customer to provide them better services and turn customers' decisions in the opposite direction

## Files and data description
#### The bank data is from [Kaggle](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers). It consists of 10,000 customers mentioning their age, salary, marital_status, credit card limit, credit card category, etc. There are nearly 18 features.

##### **File Structure**
|-Guide.ipynb          # Getting started and troubleshooting tips \
|-churn_notebook.ipynb # Contains the refactored code \
|-churn_library.py     # Define the functions that follows coding PEP 8 style \
|-churn_script_logging_and_tests.py # Contains tests and logs \
|-README.md            # Provides project overview, and instructions to use the code \
|-data                 # Read this data \
|---|-- bank_data.csv \
|-images               
|---|-- eda            # Store EDA results \
|---|-- results        # Store training results \
|-logs                 # Store logs \
|-models               # Store models


## Running Files
#### - Run with **python churn_library.py**
#### This will read in the bank data files, run EDA, encoding, feature engineering and run training. The images of histograms and distributions created from EDA are in images/eda. The images of ROC curve, classification reports and features of importance created are in images/results. Training models are saved in models/.

#### - Run with **python churn_script_logging_and_tests.py**
#### This will run the functions in churn_library to import, do EDA, encoding, feature engineering, training, create classification reports, calculate features of importance.
#### Unit tests are performed and respective log on success and failure is created in logs/churn_library.log.