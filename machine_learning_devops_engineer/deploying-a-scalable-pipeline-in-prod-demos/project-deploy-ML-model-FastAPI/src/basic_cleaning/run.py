#!/usr/bin/env python
"""
Performs basic cleaning on raw data
Author: Kay Sun
Date: September 20 2023
"""

import os
import argparse
import logging
import wandb
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # load raw dataset
    logger.info("Loading local census data")
    input_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../data"))
    df = pd.read_csv(os.path.join(input_path, "census.csv"))

    # remove prevailing whitespace in column names
    logger.info("Removing whitespaces from column names and values")
    df.columns = df.columns.str.strip()

    # remove prevailing whitespace in column values which are string
    for column in df.columns:
        if pd.api.types.is_string_dtype(df[column].dtype):
            df[column] = df[column].str.strip()

    # remove rows with "?" values
    logger.info("Removing rows with '?' values")
    columns = ['workclass', 'native-country']
    for column in columns:
        remove = df[column] == '?'
        df = df[~remove]

    # drop columns with mostly constants (capital-gain/loss) and already represented
    # by other column (education-num)
    df.drop(["education-num", "capital-gain", "capital-loss"], axis=1, inplace=True)

    # replace dashes to underscore in column names for FastAPI
    df.columns = df.columns.str.replace("-", "_")

    # save cleaned dataset
    logger.info("Writing to local data folder")
    filename = args.output_artifact
    df.to_csv(os.path.join(input_path, filename), index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="This step cleans the data")


    parser.add_argument(
        "--output_artifact", 
        type=str,
        help="Name of output artifact cleaned",
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=str,
        help="Type of output artifact",
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type=str,
        help="Description of output artifact",
        required=True
    )


    args = parser.parse_args()

    go(args)
