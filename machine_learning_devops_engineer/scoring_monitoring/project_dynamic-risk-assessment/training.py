import pandas as pd
import pickle
import os
import pathlib
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
from datetime import datetime


###################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
model_path = os.path.join(config['output_model_path']) 
output_folder_path = config['output_folder_path']

# create output folder
pathlib.Path(model_path).mkdir(parents=True, exist_ok=True)

# save file with training details
training_filename = "trainingfiles.txt"
training_files = []

# today's date
today = datetime.today().strftime('%Y%m%d')


def process_data(df):
    """
    process dataset for training or inference
    input: original dataset
    output: processed dataset
    """
    Y = None
    if 'exited' in df.columns:
        Y = df['exited']
    X = df.drop(['exited', 'corporation'], axis=1, errors='ignore')

    return Y, X


#################Function for training the model
def train_model():
    
    #use this logistic regression for training
    model_LR = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                    intercept_scaling=1, l1_ratio=None, max_iter=100,
                    multi_class='auto', n_jobs=None, penalty='l2',
                    random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                    warm_start=False)
    
    # read data and separate target from features
    df_data = pd.read_csv(os.path.join(dataset_csv_path, "finaldata.csv"))
    target, features = process_data(df_data)

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=123)

    #fit the logistic regression to your data
    model_LR.fit(X_train, y_train)

    # evaluate model
    y_pred = model_LR.predict(X_test)
    
    # write to training log
    with open(os.path.join(output_folder_path, training_filename), 'a') as fp:
        fp.write('{}\nClassification Report\n{}\nConfusion Matrix\n{}\n\n'.format(today, classification_report(y_test, y_pred), confusion_matrix(y_test, y_pred)))

    #write the trained model to your workspace in a file called trainedmodel.pkl
    with open(os.path.join(model_path, "trainedmodel.pkl"), 'wb') as fp:
        pickle.dump(model_LR, fp)


if __name__ == '__main__':
    train_model()