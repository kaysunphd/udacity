# library doc string
'''
Library to predict customer churn from bank data using machining learning.
Specifically random forest and logistic regression.
This will read in the bank data files, run EDA, encoding, feature engineering and run training. 
The images of histograms and distributions created from EDA are in images/eda. 
The images of ROC curve, classification reports and features of importance created are in images/results. 
Training models are saved in models/.

Author: Kay Sun
Creation Date: September 5th 2023 
'''

# import libraries
import os

from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import RocCurveDisplay, classification_report

import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
os.environ['QT_QPA_PLATFORM']='offscreen'


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    df = pd.read_csv(pth)
    return df


def perform_eda_statistcs(df):
    '''
    perform eda on df and display basic statistcs
    input:
            df: pandas dataframe

    output:
            display basic statistcs
    '''
    # print summary of df
    # top 5 rows
    print("\nTop 5 rows")
    print(df.head())

    # size of df
    print("\nSize of data ", df.shape)

    # list number of nulls
    print("\nNumber of nulls ", df.isnull().sum())

    # statistics of data
    print("\nDescription of data")
    print(df.describe())


def perform_eda_histgrams(df, features_list):
    '''
    calculate histogram and save figures to images folder
    input:
            df: pandas dataframe
            features_list: list of features

    output:
            images saved in images/eda
    '''
    for feature in features_list:
        save_filename = 'images/eda/' + feature + '-histogram.jpg'
        df[feature].hist()
        plt.xlabel(feature)
        plt.ylabel('count')
        plt.title('Histogram of ' + feature)
        plt.tight_layout()
        plt.savefig(save_filename)
        # plt.show()


def perform_eda_value_counts(df, features_list):
    '''
    calculate normalized value_counts and save figure to images folder
    input:
            df: pandas dataframe
            features_list: list of features

    output:
            images saved in images/eda
    '''
    for feature in features_list:
        save_filename = 'images/eda/' + feature + '-value_counts.jpg'
        df[feature].value_counts('normalize').plot(kind='bar')
        plt.xlabel(feature)
        plt.ylabel('count')
        plt.title('Value counts of ' + feature)
        plt.tight_layout()
        plt.savefig(save_filename)


def perform_eda_kde(df, features_list):
    '''
    calculate kde and save figure to images folder
    input:
            df: pandas dataframe
            features_list: list of features

    output:
            images saved in images/eda
    '''
    for feature in features_list:
        save_filename = 'images/eda/' + feature + '-kde.jpg'
        sns.histplot(df[feature], stat='density', kde=True)
        plt.xlabel(feature)
        plt.ylabel('count')
        plt.title('Kernel density estimation of ' + feature)
        plt.tight_layout()
        plt.savefig(save_filename)


def perform_eda_heatmap(df, save_filename):
    '''
    calculate kde and save figure to images folder
    input:
            df: pandas dataframe

    output:
            images saved in images/eda
    '''
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths = 2)
    plt.title('Heatmap')
    plt.tight_layout()
    plt.savefig('images/eda/' + save_filename + '.jpg')


def encoder_helper(df, category_lst, response=None):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used 
                      for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    for feature in category_lst:
        feature_groups = df[[feature, 'Churn']].groupby(feature).mean()
        df[feature + '_Churn'] = df[feature].apply(lambda val: feature_groups.loc[val])

    return df


def perform_feature_engineering(df, feature_list, target, size_of_test, random_seed):
    '''
    input:
              df: pandas dataframe
              feature_list: list of features
              target: target of analysis
              size_of_test: train/test split ratio
              random_seed: integer for random state

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    # features
    X = pd.DataFrame()
    X[feature_list] = df[feature_list]

    # specify target
    y = df[target]

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size= size_of_test,
                                                        random_state=random_seed)

    return X_train, X_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf,
                                X_test):
    '''
    produces classification report for training and testing results and stores report 
    as images in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest
            X_test: test training dataset

    output:
             None
    '''
    # classification report random forest
    plt.figure()
    #plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
    plt.text(0.01, 1.25, str('Random Forest Train'), {'fontsize': 10},
             fontproperties = 'monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)),
             {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {'fontsize': 10},
             fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.axis('off')
    save_filename = './images/results/rfc_classification_report.jpg'
    plt.savefig(save_filename)

    # classification report logistic regression
    plt.figure()
    plt.text(0.01, 1.25, str('Logistic Regression Train'), {'fontsize': 10},
             fontproperties = 'monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {'fontsize': 10},
             fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {'fontsize': 10},
             fontproperties = 'monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {'fontsize': 10},
             fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.axis('off')
    save_filename = './images/results/logistic_classification_report.jpg'
    plt.savefig(save_filename)

    # load saved models
    rfc_model = joblib.load('./models/rfc_model.pkl')
    lr_model = joblib.load('./models/logistic_model.pkl')

    # plot roc_auc
    fig = plt.figure()
    ax = plt.gca()
    RocCurveDisplay.from_estimator(rfc_model, X_test, y_test, ax=ax, alpha=0.8)
    RocCurveDisplay.from_estimator(lr_model, X_test, y_test, ax=ax, alpha=0.8)
    fig.savefig('./images/results/roc_curve.jpg')


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances from model
    importances = model.feature_importances_

    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure()

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)

    plt.savefig(output_pth)

    # # shap explanation of feature importances
    # explainer = shap.TreeExplainer(model)
    # shap_values = explainer.shap_values(X_data)
    # shap.summary_plot(shap_values, X_data, plot_type="bar")


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    # grid search
    rfc = RandomForestClassifier(random_state=42)

    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference: https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth' : [4,5,100],
        'criterion' :['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)

    # fit into models
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)

    # make predictions
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # save best model
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    rfc_model = joblib.load('./models/rfc_model.pkl')

    # create classification report
    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf,
                                X_test)

    # find features of importance
    save_filename = './images/results/feature_importance_plot.jpg'
    feature_importance_plot(rfc_model, X_train, save_filename)


if __name__ == "__main__":
    # define path of input file
    csv_path = "./data/bank_data.csv"

    # read csv into dataframe
    df_data = import_data(csv_path)

    # perform basic statistics on data
    perform_eda_statistcs(df_data)

    # list of catagorical features
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'                
    ]

    # list of quantitative features
    quant_columns = [
        'Customer_Age',
        'Dependent_count', 
        'Months_on_book',
        'Total_Relationship_Count', 
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon', 
        'Credit_Limit', 
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy', 
        'Total_Amt_Chng_Q4_Q1', 
        'Total_Trans_Amt',
        'Total_Trans_Ct', 
        'Total_Ct_Chng_Q4_Q1', 
        'Avg_Utilization_Ratio'
    ]

    # eda
    # kde for Total_Trans_Ct
    perform_eda_kde(df_data, ['Total_Trans_Ct'])

    # value_counts for cat_columns
    perform_eda_value_counts(df_data, cat_columns)

    # convert Attrition to Churn by converting string to int
    df_data['Churn'] = df_data['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    # histgram for Churn and Customer_Age
    perform_eda_histgrams(df_data, ['Churn', 'Customer_Age'])

    # heatmap of quantitative features
    perform_eda_heatmap(df_data[quant_columns], 'heatmap.jpg')

    # calculate proporation based on churn
    df_data = encoder_helper(df_data,
                             ['Gender',
                             'Education_Level',
                             'Marital_Status',
                             'Income_Category',
                             'Card_Category'])

    # select features
    keep_cols = ['Customer_Age',
                 'Dependent_count',
                 'Months_on_book',
                 'Total_Relationship_Count',
                 'Months_Inactive_12_mon',
                 'Contacts_Count_12_mon',
                 'Credit_Limit',
                 'Total_Revolving_Bal',
                 'Avg_Open_To_Buy',
                 'Total_Amt_Chng_Q4_Q1',
                 'Total_Trans_Amt',
                 'Total_Trans_Ct',
                 'Total_Ct_Chng_Q4_Q1',
                 'Avg_Utilization_Ratio',
                 'Gender_Churn',
                 'Education_Level_Churn',
                 'Marital_Status_Churn', 
                 'Income_Category_Churn',
                 'Card_Category_Churn']

    # select features, split to train/test
    XX_train, XX_test, yy_train, yy_test = perform_feature_engineering(df_data,
                                                                   keep_cols,
                                                                   'Churn',
                                                                   0.3,
                                                                   42)

    # train models and evaluate models
    train_models(XX_train, XX_test, yy_train, yy_test)
