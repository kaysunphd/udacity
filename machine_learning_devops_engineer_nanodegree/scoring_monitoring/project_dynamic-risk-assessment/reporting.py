import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import diagnostics
import training

###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
model_path = os.path.join(config['output_model_path'])
test_data_path = os.path.join(config['test_data_path'])

##############Function for reporting
def report_score_model():
    #calculate a confusion matrix using the test data and the deployed model
    #write the confusion matrix to the workspace

    # load test data and process it for inference
    df_test = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))
    target, features = training.process_data(df_test)

    y_pred = diagnostics.model_predictions(features)
    cm = confusion_matrix(target, y_pred, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['no exit','exited'])
    disp.plot()
    disp.ax_.set_title("Normalized Confusion Matrix")
    disp.figure_.savefig(os.path.join(model_path,'confusionmatrix.png'))


if __name__ == '__main__':
    report_score_model()
