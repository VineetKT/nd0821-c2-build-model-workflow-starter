#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging

import pandas as pd
import wandb

logging.basicConfig(filename='/Users/vineetkumar/Documents/udacity_ml_devops/project 2/nd0821-c2-build-model-workflow-starter/logs/basic_clean.log',
                    level=logging.INFO,
                    format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    artifact_local_path = run.use_artifact(args.input_artifact).file()
    logger.info('Input artifact received')

    ######################
    # YOUR CODE HERE     #
    ######################
    df = pd.read_csv(artifact_local_path)

    # filter outliers in 'price' column
    idx = df['price'].between(args.min_price, args.max_price)
    df = df[idx].copy()

    # filter outliers in 'longitude' column
    idx = df['longitude'].between(-74.25, -73.50) & \
        df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()

    # convert last_review column type from str to datetime
    df['last_review'] = pd.to_datetime(df['last_review'])
    logger.info('Dataframe cleaning steps done')

    # save resultant dataframe to a local file
    df.to_csv('clean_sample.csv', index=False)
    logger.info('Dataframe saved to csv')

    output_artifact = wandb.Artifact(
        name=args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    output_artifact.add_file('clean_sample.csv')

    run.log_artifact(output_artifact)
    logger.info('Cleaned data artifact logged to W&B')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help='Name of input artifact raw data',
        required=True
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help='Name of output artifact cleaned data',
        required=True
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help='Type of output artifact',
        required=True
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help='Cleaned data after EDA steps',
        required=True
    )

    parser.add_argument(
        "--min_price",
        type=float,
        help='minimum price to filter price column value',
        required=True
    )

    parser.add_argument(
        "--max_price",
        type=float,
        help='maximum price to filter price column value',
        required=True
    )

    args = parser.parse_args()

    go(args)
