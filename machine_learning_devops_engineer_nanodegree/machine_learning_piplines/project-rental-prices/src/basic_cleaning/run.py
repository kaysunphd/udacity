#!/usr/bin/env python
"""
Performs basic cleaning on data and save results in Weights & Biases
"""
import argparse
import logging
import wandb
import pandas as pd
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    # artifact_local_path = run.use_artifact(args.input_artifact).file()

    logger.info("Downloading artifact")
    artifact_path = run.use_artifact(args.input_artifact).file()

    df = pd.read_csv(artifact_path)
    
    # Drop outliers
    logger.info("Dropping outliers")
    idx = df['price'].between(args.min_price, args.max_price)
    df = df[idx].copy()

    # Convert last_review to datetime
    logger.info("Converting last_review from string to datetime")
    df['last_review'] = pd.to_datetime(df['last_review'])

    # Remove any unamed columns
    logger.info("Remove any unamed columns")
    df = df.drop(['Unnamed: 0'], axis=1, errors='ignore')

    # Impose boundary
    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()

    # Save cleaned dataset
    logger.info("Writing to artifact")
    filename = args.output_artifact
    df.to_csv(filename, index=False)

    artifact = wandb.Artifact(
        name=args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file(filename)

    # Logging artifact
    logger.info("Logging artifact")
    run.log_artifact(artifact)

    # Remove saved cleaned file
    os.remove(filename)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="This step cleans the data")

    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="Name of input artifact to be cleaned",
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type=float,
        help="Minimum price to eliminate outliers",
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type=float,
        help="Maximum price to eliminate outliers",
        required=True
    )

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
