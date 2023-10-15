'''
Tests and logging to library to predict customer churn from bank data using machining learning.
This will run the functions in churn_library to import, do EDA, encoding, feature engineering, \
training, create classification reports, calculate features of importance.
Unit tests are performed and respective log on success and failure is created in logs/churn_library.log.

Author: Kay Sun
Creation Date: September 5th 2023 
'''

import os
import logging
import churn_library as cls

logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error("Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda_statistics(perform_eda_statistcs, import_data):
    '''
    test perform eda statistics function
    '''
    try:
        df = import_data("./data/bank_data.csv")
        perform_eda_statistcs(df)
        logging.info("Testing perform_eda_statistcs: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing perform_eda_statistcs: The input file for dataframe wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error("Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda_histograms(perform_eda_histgrams, import_data):
    '''
    test perform eda histograms function
    '''
    try:
        df = import_data("./data/bank_data.csv")
        perform_eda_histgrams(df, ['Customer_Age'])
        logging.info("Testing perform_eda_histgrams: SUCCESS")
        assert os.path.isfile("./images/eda/Customer_Age-histogram.jpg")
    except FileNotFoundError as err:
        logging.error("Testing perform_eda_histgrams: The input file for dataframe wasn't found")
        raise err
    except AssertionError as err:
        logging.error("Testing perform_eda_histgrams: Could not save histogram image")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
        assert 'Customer_Age' in df.columns
    except AssertionError as err:
        logging.error("Testing perform_eda_histgrams: The dataframe doesn't appear to have \
                        rows and columns or contain 'Customer_Age'")
        raise err


def test_eda_value_counts(perform_eda_value_counts, import_data):
    '''
    test perform eda value_counts function
    '''
    try:
        df = import_data("./data/bank_data.csv")
        perform_eda_value_counts(df, ['Marital_Status'])
        logging.info("Testing perform_eda_value_counts: SUCCESS")
        assert os.path.isfile("./images/eda/Marital_Status-value_counts.jpg")
    except FileNotFoundError as err:
        logging.error("Testing perform_eda_value_counts: The input file for dataframe wasn't found")
        raise err
    except AssertionError as err:
        logging.error("Testing perform_eda_value_counts: Could not save value_counts image")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
        assert 'Marital_Status' in df.columns
    except AssertionError as err:
        logging.error("Testing perform_eda_value_counts: The dataframe doesn't appear to have \
                        rows and columns or contain 'Marital_Status'")
        raise err


def test_eda_kde(perform_eda_kde, import_data):
    '''
    test perform eda kde function
    '''
    try:
        df = import_data("./data/bank_data.csv")
        perform_eda_kde(df, ['Total_Trans_Ct'])
        logging.info("Testing perform_eda_kde: SUCCESS")
        assert os.path.isfile("./images/eda/Total_Trans_Ct-kde.jpg")
    except FileNotFoundError as err:
        logging.error("Testing perform_eda_kde: The input file for dataframe wasn't found")
        raise err
    except AssertionError as err:
        logging.error("Testing perform_eda_kde: Could not save kde image")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
        assert 'Total_Trans_Ct' in df.columns
    except AssertionError as err:
        logging.error("Testing perform_eda_kde: The dataframe doesn't appear to have \
                        rows and columns or contain 'Total_Trans_Ct'")
        raise err


def test_eda_heatmap(perform_eda_heatmap, import_data):
    '''
    test perform eda heatmap function
    '''
    try:
        df = import_data("./data/bank_data.csv")
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
        perform_eda_heatmap(df[quant_columns], 'heatmap')
        logging.info("Testing perform_eda_heatmap: SUCCESS")
        assert os.path.isfile("./images/eda/heatmap.jpg")
    except FileNotFoundError as err:
        logging.error("Testing perform_eda_heatmap: The input file for dataframe wasn't found")
        raise err
    except AssertionError as err:
        logging.error("Testing perform_eda_heatmap: Could not save heatmap image")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
        assert 'Total_Trans_Ct' in df.columns
    except AssertionError as err:
        logging.error("Testing perform_eda_heatmap: The dataframe doesn't appear to have \
                        rows and columns or contain 'Total_Trans_Ct'")
        raise err


def test_encoder_helper(encoder_helper, import_data):
    '''
    test encoder helper
    '''
    try:
        df = import_data("./data/bank_data.csv")
        df['Churn'] = df['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)
        df = encoder_helper(df, ['Gender'])
        logging.info("Testing encoder_helper: SUCCESS")
        assert 'Gender_Churn' in df.columns
    except FileNotFoundError as err:
        logging.error("Testing encoder_helper: The input file for dataframe wasn't found")
        raise err
    except AssertionError as err:
        logging.error("Testing encoder_helper: Could not create Gender_Churn column")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
        assert 'Gender' in df.columns
    except AssertionError as err:
        logging.error("Testing encoder_helper: The dataframe doesn't appear to have \
                        rows and columns or contain 'Gender'")
        raise err


def test_perform_feature_engineering(perform_feature_engineering, encoder_helper, import_data):
    '''
    test perform_feature_engineering
    '''
    try:
        df = import_data("./data/bank_data.csv")

        # convert Attrition to Churn by converting string to int
        df['Churn'] = df['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)

        # calculate proporation based on churn
        df = encoder_helper(df,
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
        XX_train, XX_test, yy_train, yy_test = perform_feature_engineering(df,
                                                                       keep_cols,
                                                                       'Churn',
                                                                       0.3,
                                                                       42)
        logging.info("Testing perform_feature_engineering: SUCCESS")
        assert XX_train.shape[0] > 0
        assert XX_train.shape[1] > 0
        assert XX_test.shape[0] > 0
        assert XX_test.shape[1] > 0
        assert yy_train.shape[0] > 0
        assert yy_test.shape[0] > 0
    except FileNotFoundError as err:
        logging.error("Testing perform_feature_engineering: The input file for \
                        dataframe wasn't found")
        raise err
    except AssertionError as err:
        logging.error("Testing perform_feature_engineering: The train and test split \
                        doesn't appear to have rows and columns")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
        assert 'Customer_Age' in df.columns
    except AssertionError as err:
        logging.error("Testing perform_feature_engineering: The dataframe doesn't \
                        appear to have rows and columns or contain 'Customer_Age'")
        raise err


def test_train_models(train_models, perform_feature_engineering, encoder_helper, import_data):
    '''
    test train_models
    '''
    try:
        df = import_data("./data/bank_data.csv")

        # convert Attrition to Churn by converting string to int
        df['Churn'] = df['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)

        # calculate proporation based on churn
        df = encoder_helper(df,
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
        XX_train, XX_test, yy_train, yy_test = perform_feature_engineering(df,
                                                                       keep_cols,
                                                                       'Churn',
                                                                       0.3,
                                                                       42)

        # train models and evaluate models
        train_models(XX_train, XX_test, yy_train, yy_test)

        logging.info("Testing test_train_models: SUCCESS")
        assert XX_train.shape[0] > 0
        assert XX_train.shape[1] > 0
        assert XX_test.shape[0] > 0
        assert XX_test.shape[1] > 0
        assert yy_train.shape[0] > 0
        assert yy_test.shape[0] > 0
        assert os.path.isfile("./models/rfc_model.pkl")
        assert os.path.isfile("./models/logistic_model.pkl")
        assert os.path.isfile("./images/results/rfc_classification_report.jpg")
        assert os.path.isfile("./images/results/logistic_classification_report.jpg")
        assert os.path.isfile("./images/results/feature_importance_plot.jpg")
    except FileNotFoundError as err:
        logging.error("Testing test_train_models: The input file for \
                        dataframe wasn't found")
        raise err
    except AssertionError as err:
        logging.error("Testing test_train_models: The train and test \
                        split doesn't appear \
                        to have rows and columns \
                        or could not save models as pickle \
                        or could not save classification reports \
                        or could not save the feature importance plot")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
        assert 'Customer_Age' in df.columns
    except AssertionError as err:
        logging.error("Testing test_train_models: The dataframe doesn't \
                      appear to have rows and columns \
                      or contain 'Customer_Age'")
        raise err


if __name__ == "__main__":
    test_import(cls.import_data)
    test_eda_histograms(cls.perform_eda_histgrams,
                        cls.import_data)
    test_eda_value_counts(cls.perform_eda_value_counts,
                          cls.import_data)
    test_eda_heatmap(cls.perform_eda_heatmap,
                     cls.import_data)
    test_encoder_helper(cls.encoder_helper,
                        cls.import_data)
    test_perform_feature_engineering(cls.perform_feature_engineering,
                                     cls.encoder_helper,
                                     cls.import_data)
    test_train_models(cls.train_models,
                      cls.perform_feature_engineering,
                      cls.encoder_helper,
                      cls.import_data)
