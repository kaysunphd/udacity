import pandas as pd
import pickle
import os
from sklearn import metrics
import json
from datetime import datetime
import training


#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path'])
model_path = os.path.join(config['output_model_path']) 

# today's date
today = datetime.today().strftime('%Y%m%d')


def predict_model(df):
    """
    function to make prediction from model.
    input: data
    output: predictions
    """
    # load the model
    modelpath = os.path.join(model_path, 'trainedmodel.pkl')
    with open(modelpath, 'rb') as fp:
        model = pickle.load(fp)

    # make inference
    y_pred = model.predict(df)

    return y_pred
    

#################Function for model scoring
def score_model():
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file

    # load test data and process it for inference
    df_test = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))
    target, features = training.process_data(df_test)

    # make inference
    y_pred = predict_model(features)
    f1 = metrics.f1_score(target, y_pred)

    # write to scores log
    with open(os.path.join(model_path, "latestscore.txt"), 'a') as fp:
        fp.write('{}, F1 Score:{}\n'.format(today, f1))

    return f1


if __name__ == '__main__':
    score_model()