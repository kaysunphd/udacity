from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import json
import os
from scoring import score_model
from training import process_data
from diagnostics import model_predictions, dataframe_summary, missing_data, execution_time, outdated_packages_list


######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 

prediction_model = None


@app.route("/")
def index():        
    #welcoming message
    return 'This part working'

#######################Prediction Endpoint
@app.route("/prediction", methods=['POST', 'OPTIONS'])
def predict():        
    #call the prediction function you created in Step 3
    filepath = request.args.get('filename')

    df = pd.read_csv(filepath)
    _, features = process_data(df)

    y_pred = model_predictions(features)
    return jsonify(y_pred.tolist())


#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def scoring():        
    #check the score of the deployed model
    return {'F1 score': score_model()}


#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def summarystats():        
    #check means, medians, and modes for each column
    return jsonify(dataframe_summary())


#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnostics():        
    #check timing and percent NA values
    missing = missing_data()
    time = execution_time()
    outdated = outdated_packages_list()

    diagnose = {
        'missing_data': missing,
        'execution_time': time,
        'outdated_packages_list': outdated
    }

    return jsonify(diagnose)


if __name__ == "__main__":    
    app.run(host='127.0.0.1', port=8000, debug=True, threaded=True)
