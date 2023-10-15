import pandas as pd
import numpy as np
import timeit
import os
from io import StringIO
import json
import pickle
import subprocess
import training

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])

##################Function to get model predictions
def model_predictions(df):
    """
    read the deployed model and a test dataset, calculate predictions
    input: dataset
    return: list of predictions
    """
    # load the model
    modelpath = os.path.join(prod_deployment_path, 'trainedmodel.pkl')
    with open(modelpath, 'rb') as fp:
        model = pickle.load(fp)
    
     # process data for inference
    target, features = training.process_data(df)

    # make inference
    y_pred = model.predict(features)

    return y_pred


##################Function to get summary statistics
def dataframe_summary():
    """
    calculate summary statistics here
    return: list of statistics for each numeric column
    """
    # read data and isolate numeric columns
    df_data = pd.read_csv(os.path.join(dataset_csv_path, "finaldata.csv"))
    df_data = df_data.drop(['exited'], axis=1)
    df_data = df_data.select_dtypes(np.number)

    # calculate statistics
    statistics = []
    for column in df_data.columns:
        mean = df_data[column].mean()
        median = df_data[column].median()
        stddev = df_data[column].std()
        statistics.append((column, mean, median, stddev))

    return statistics


##################Function to get missing data
def missing_data():
    """
    calculate missing data
    return: list of missing data as percentage
    """
    # read data
    df_data = pd.read_csv(os.path.join(dataset_csv_path, "finaldata.csv"))

    # calculate missing data as percentage
    df_missing = df_data.isna().sum(axis=0)
    df_missing /= len(df_data.isna().sum(axis=0)) *100
    
    return list(zip(df_missing.index, df_missing))


##################Function to get timings
def execution_time():
    """
    calculate timing of training.py and ingestion.py
    return: list of 2 timings for ingestion and training
    """
    timings = []

    # timing ingestion
    start_time = timeit.default_timer()
    os.system('python ingestion.py')
    duration = timeit.default_timer() - start_time
    timings.append(('ingestion', duration))

    # timing training
    start_time = timeit.default_timer()
    os.system('python training.py')
    duration = timeit.default_timer() - start_time
    timings.append(('training', duration))

    return timings


##################Function to check dependencies
def outdated_packages_list():
    """
    get a list of outdated dependencies
    return: list of dependencies
    """
    # from https://stackoverflow.com/questions/48946492/capturing-terminal-output-into-pandas-dataframe-without-creating-external-text-f

    output = subprocess.Popen(['pip', 'list', '--outdated'], stdout=subprocess.PIPE)
    output = StringIO(output.communicate()[0].decode('utf-8'))
    df = pd.read_csv(output, sep='\s+')
    df.drop(index=[0], axis=0, inplace=True)
    df.drop(['Type'], axis=1, inplace=True)
    
    return df.values.tolist()


if __name__ == '__main__':
    # model_predictions()
    # dataframe_summary()
    # execution_time()
    # outdated_packages_list()
    dataframe_summary()





    
