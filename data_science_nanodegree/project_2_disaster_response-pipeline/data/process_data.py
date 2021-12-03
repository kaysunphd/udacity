import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Loads messages and categories csv from filepaths, and merge them based on their "id".
    
    Args:
    messages_filepath: File path to messages.
    categories_filepath: File path to categories.
    
    Return:
    df: Merged dataframe of messages and categories.
    """
    
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    
    # load categories dataset
    categories = pd.read_csv(categories_filepath)

    # merge datasets
    df = pd.merge(left=messages, right=categories, left_on="id", right_on="id", how="left")
    
    return df


def clean_data(df):
    """
    Clean the dataframe by
    1) splitting "categories" column into separate category columns.
    2) converting category values to numbers 0 or 1.
    3) replacing "categories" column with new catgeory columns.
    4) removing duplicates.
    5) confirm only 0 or 1 for categories.
    
    Args:
    df: Dataframe for cleaning.
    
    Return:
    df: Cleaned dataframe.
    """
    # 1) splitting "categories" column into separate category columns.
    # create a dataframe of the 36 individual category columns
    categories = df["categories"].str.split(";", expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = categories.iloc[0].apply(lambda x: x.split("-")[0]).tolist()

    # rename the columns of `categories`
    categories.columns = category_colnames
    
    # 2) convert category values to just numbers 0 or 1.
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]
    
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        

    # 3) replacing "categories" column with new catgeory columns.
    # drop the original categories column from `df`
    df.drop(['categories'], axis=1, inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # 4) drop duplicates
    df.drop_duplicates(inplace=True)
    
    # confirm only 0 or 1 for categories. outliers/mislabels dropped.
    for column in categories:
        df.drop(df[df[column] > 1].index, inplace=True)
    
    
    return df


def save_data(df, database_filename):
    """
    Save dataframe to sqlite database.
    
    Args:
    df: Dataframe for saving.
    database_filename: Filename to database.
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('messages', engine, index=False, if_exists="replace")  


def main():
    """
    Load, clean and save data using file paths from system arguments.
    """
    
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()