import sys
import pandas as pd
from sqlalchemy import create_engine

import nltk
nltk.download(['punkt', 'wordnet', 'stopwords', 'averaged_perceptron_tagger'])

import re
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier

import pickle


def load_data(database_filepath):
    """
    Loads data stored in database.
    
    Args:
    database_filepath: File path to database.
    
    Return:
    X: Dataframe of predictors ("messages" column).
    Y: Dataframe of target (message classification columns).
    categories_names: List of classification categories names.
    """
    
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('messages', engine)

    # optional reduction of data size to speed up training.
#     drop_indices = np.random.choice(df.index, int(df.shape[0]*0.5), replace=False)
#     df = df.drop(drop_indices)

    X = df['message']
    Y = df.iloc[:, -36:]
    categories_names = df.iloc[:, -36:].columns

    return X, Y, categories_names


def tokenize(text):
    """
    Tokenize the text from messages.
    
    Args:
    text: Text string from message.
    
    Return:
    clean_tokens: List of cleaned tokenized text.
    """
    
    # lower case text, regex to remove special characters, tokenize
    tokens = word_tokenize(re.sub(r"[^a-zA-Z0-9]", " ", text.lower()))
    
    # initialize lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # list of stop words in english
    stop_words = stopwords.words("english")
    
    # lemmatize, lower case, strip and append if not stop word.
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        if clean_tok not in stop_words:
            clean_tokens.append(clean_tok)
    
    return clean_tokens


def build_model(verbose=3, n_jobs=1):
    """
    Build machine learning model using assigned parameters ranges for GridSearchCV.
    
    Optional Args:
    verbose: Set how much output to display during testing, ranging from 0 to 3.
    n_jobs: Set how many cpus to use, starts from -1 (use all).
    
    Return:
    cv: GridSearchCV ML model.
    """
    
    # define pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # define parameters for grid search
    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.5, 1.0),
        'vect__max_features': (None, 5000)
    }

    # initialize GridSearchCV
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=verbose, n_jobs=n_jobs)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate machine learning model with accuracy, precision, recall and f1.
    
    Args:
    model: Model for evaluation.
    X_test: Predictors for testing.
    Y_test: Targets for testing.
    category_names: Target categories names.
    """
    
    # run prediction using X_test
    y_pred = pd.DataFrame(model.predict(X_test), columns=category_names)

    # calculate each evaluation score (accuracy, precision, recall, f1) for each category.
    evaluate = {}
    for i, column in enumerate(y_pred.columns):
        evaluate[column] = []
        evaluate[column].append(accuracy_score(Y_test[column], y_pred[column]))
        evaluate[column].append(precision_score(Y_test[column], y_pred[column]))
        evaluate[column].append(recall_score(Y_test[column], y_pred[column]))
        evaluate[column].append(f1_score(Y_test[column], y_pred[column]))

    evaluate_df = pd.DataFrame(evaluate).T
    evaluate_df.columns = ['accuracy', 'precision', 'recall', 'f1']

    # print out for visualization
    print(evaluate_df)


def save_model(model, model_filepath):
    """
    Save machine learning model.
    
    Args:
    model: Model for evaluation.
    model_filepath: File path to save model to.
    """
    
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    """
    Load data, split data into train and test datasets, and build, train, evaluate and save model.
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model(verbose=3, n_jobs=-1)
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()