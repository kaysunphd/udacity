import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score
from src.ml.data import process_data


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    cv: cross-validation object
        Optional cross-validation
        
    Returns
    -------
    model
        Trained machine learning model.
    """

    rfc = RandomForestClassifier(random_state=123)
    search_grid = {
                'n_estimators': [10, 20, 30],
                'max_depth': [5, 10, 20],
                'min_samples_split': [20, 50, 100],
                }

    rfc_gridsearch = GridSearchCV(estimator=rfc,
                                  param_grid=search_grid,
                                  scoring='f1_weighted',
                                  cv=5,
                                  verbose=2,
                                  n_jobs=-1)

    rfc_gridsearch.fit(X_train, y_train)
    best_grid = rfc_gridsearch.best_estimator_

    # train_predictions = best_grid.predict(X_train)
    # precision, recall, fbeta = compute_model_metrics(y_train, train_predictions)

    return best_grid


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """

    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ 
    Run model inferences and return the predictions.

    Inputs
    ------
    model : Hyperparameter tunned Random Forest Classifier
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """

    predictions = model.predict(X)

    return predictions


def compute_model_metrics_on_slices(test_X, test_Y, test_pred, feature):
    """
    Validates the trained machine learning model using precision, recall, and F1
    on specified feature.

    Inputs
    ------
    test_X: np.array
        Data with features
    test_Y: np.array
        Known labels, binarized.
    test_pred : np.array
        Predicted labels, binarized.
    feature: string
        Name of feature in column

    Returns
    -------
    df_selected_metrics: pd.DataFrame
        Contains feature: string,
                 feature value: string,
                 precision: float,
                 recall : float,
                 fbeta : float
    """

    feature_values = test_X[feature].unique().tolist()

    selected_metrics = []
    for feature_value in feature_values:
        selected = test_X[feature] == feature_value
        selected_Y = test_Y[selected]
        selected_pred = test_pred[selected]
        precision, recall, fbeta = compute_model_metrics(selected_Y, selected_pred)
        selected_metrics.append([feature, feature_value, precision, recall, fbeta])

    df_selected_metrics = pd.DataFrame(selected_metrics,
                                    columns = ['Feature',
                                                'Value',
                                                'Precision',
                                                'Recall',
                                                'F-beta'])
    return df_selected_metrics


def make_inference(X, cat_features):
    """ 
    Run model inferences pipeline, make inference and return the predictions.

    Inputs
    ------
    X : np.array
        Data used for prediction.
    cat_features: list of strings
        List of categorical features.

    Returns
    -------
    predictions : list
        Predictions from the model.
    """

    # Load saved mode, encoder, lb
    filepath = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../model"))
    model_filename = os.path.join(filepath, "trained_model.pkl")
    encoder_filename = os.path.join(filepath, "trained_encoder.pkl")

    with open(model_filename, 'rb') as fp:
        model = pickle.load(fp)

    with open(encoder_filename, 'rb') as fp:
        encoder = pickle.load(fp)

    # Proces the query data with the process_data function.
    X_test, _, _, _ = process_data(
        X,
        categorical_features=cat_features,
        training=False,
        encoder=encoder,
    )

    predictions_list = model.predict(X_test).tolist()

    return predictions_list


def make_inference_labels(X, cat_features):
    """ 
    Run model inferences pipeline, make inference and return the label predictions.

    Inputs
    ------
    X : np.array
        Data used for prediction.
    cat_features: list of strings
        List of categorical features.

    Returns
    -------
    label_predictions : list
        Lables of predictions from the model.
    """
    filepath = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../model"))
    lb_filename = os.path.join(filepath, "trained_lb.pkl")

    with open(lb_filename, 'rb') as fp:
        lb = pickle.load(fp)

    predictions_list = make_inference(X, cat_features)

    label_predictions = lb.inverse_transform(np.array(predictions_list)).tolist()

    return label_predictions