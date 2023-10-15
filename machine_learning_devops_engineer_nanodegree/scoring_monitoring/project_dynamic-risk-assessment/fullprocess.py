import json
import os
import pandas as pd

from ingestion import merge_multiple_dataframe
from diagnostics import model_predictions
from training import process_data, train_model
from scoring import score_model
from deployment import store_model_into_pickle
from reporting import report_score_model

from sklearn import metrics


###################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']
prod_deployment_path = config['prod_deployment_path']
model_path = config['output_model_path']


def calculate_old_score():
    """
    Retrieve latest F1 score from latestscore.txt
    Return: latest F1 score
    """
    latest_score = pd.read_csv(os.path.join(prod_deployment_path, 'latestscore.txt'), header=None, names=['date', 'F1'])
    latest_score.drop_duplicates(inplace=True)
    latest_score['F1'] = latest_score['F1'].str.replace("F1 Score:", "")
    # print(latest_score)

    return float(latest_score['F1'].iloc[-1])


def calculate_new_score():
    """
    Calculate F1 score using updated finaldata.csv
    Return: latest F1 score
    """
    df_newdata = pd.read_csv(os.path.join(output_folder_path,'finaldata.csv'))
    target, features = process_data(df_newdata)
    y_pred = model_predictions(features)
    score = metrics.f1_score(target, y_pred)
    # print(score)

    return score


def fullprocess():
    to_proceed = False

    ##################Check and read new data
    #first, read ingestedfiles.txt
    ingested_files = pd.read_csv(os.path.join(prod_deployment_path,'ingestedfiles.txt'), header=None, names=['date', 'file'])
    ingested_files.drop_duplicates(inplace=True)
    ingested_files = ingested_files['file'].tolist()
    # print(ingested_files)

    #second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
    source_files = []
    for root, subFolders, files in os.walk(input_folder_path):
        for file in files:
            extension = os.path.splitext(file)[1]
            if extension == '.csv' and file not in ingested_files:
                source_files.append(file)

    # print(source_files)

    ##################Deciding whether to proceed, part 1
    #if you found new data, you should proceed. otherwise, do end the process here
    if len(source_files):
        merge_multiple_dataframe()
        to_proceed = True
    else:
        to_proceed = False

    ##################Checking for model drift
    #check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
    if to_proceed:
        old_score = calculate_old_score()
        new_score = calculate_new_score()

        if new_score > old_score:
            to_proceed = True
        else:
            to_proceed = False

    ##################Deciding whether to proceed, part 2
    #if you found model drift, you should proceed. otherwise, do end the process here
    if to_proceed:
        train_model()
        score_model()

        ##################Re-deployment
        #if you found evidence for model drift, re-run the deployment.py script
        store_model_into_pickle()
    
        ##################Diagnostics and reporting
        #run diagnostics.py and reporting.py for the re-deployed model
        report_score_model()
        os.system('python apicalls.py')


if __name__ == '__main__':
    fullprocess()



