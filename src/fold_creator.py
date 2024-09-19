#!/usr/bin/env python
"""
fold_creator.py

CLI tool for creating K-Folds for classification, regression, and general purposes.

Usage Examples:

1. General K-Folds:

   python fold_creator.py kfolds --file_path ./data/train.csv --n_splits 5 --shuffle --random_state 42 --save_path ./data/train_folds.csv

2. Classification K-Folds:

   python fold_creator.py classification_kfolds --file_path ./data/train.csv --target target_column --n_splits 5 --random_state 42 --save_path ./data/train_folds.csv

3. Regression K-Folds with Sturges' Rule:

   python fold_creator.py regression_kfolds --file_path ./data/train.csv --target target_column --n_splits 5 --binning_method sturges --random_state 42 --save_path ./data/train_folds.csv

For more information, use the --help flag with any command.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.cluster import KMeans
import click

@click.group()
def cli():
    """CLI for creating K-Folds for classification, regression, and general purposes."""
    pass

@cli.command()
@click.option("--file_path", default="./input/train.csv", type=str, help="Path to the input CSV file containing the dataset. Default is './input/train.csv'.")
@click.option("--n_splits", default=5, type=int, help="Number of folds. Default is 5.")
@click.option("--shuffle/--no-shuffle", default=True, help="Whether to shuffle the data. Default is True.")
@click.option("--random_state", default=42, type=int, help="Seed for the random number generator. Default is 42.")
@click.option("--save_path", default=None, type=str, help="Optional path to save the CSV file. If None, the file is not saved.")
def kfolds(file_path, n_splits, shuffle, random_state, save_path):
    """
    Creates K-Fold indices for a dataset loaded from a CSV file.
    """
    data = pd.read_csv(file_path)
    data["kfold"] = -1
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    for fold, (train_idx, val_idx) in enumerate(kf.split(data)):
        data.loc[val_idx, "kfold"] = fold

    if save_path:
        data.to_csv(save_path, index=False)

    click.echo(f"K-Folds created with {n_splits} splits and saved to {save_path if save_path else 'not saved.'}")

@cli.command()
@click.option("--file_path", default="./input/train.csv", type=str, help="Path to the input CSV file containing the dataset. Default is './input/train.csv'.")
@click.option("--n_splits", default=5, type=int, help="Number of folds. Default is 5.")
@click.option("--target", required=True, type=str, help="The name of the target column.")
@click.option("--random_state", default=42, type=int, help="Seed for the random number generator. Default is 42.")
@click.option("--save_path", default="./input/train_folds.csv", type=str, help="Optional path to save the CSV file. Default is './input/train_folds.csv'.")
def classification_kfolds(file_path, n_splits, target, random_state, save_path):
    """
    Creates stratified K-Fold indices for classification tasks from a CSV file.
    """
    data = pd.read_csv(file_path)
    data["kfold"] = -1
    y = data[target].values

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for fold, (train_idx, val_idx) in enumerate(skf.split(data, y)):
        data.loc[val_idx, "kfold"] = fold

    # Pending: Should i remove the condition for saving the file?
    if save_path:
        data.to_csv(save_path, index=False)

    click.echo(f"Classification K-Folds created and saved to {save_path if save_path else 'not saved.'}")

@cli.command()
@click.option("--file_path", default="./input/train.csv", type=str, help="Path to the input CSV file containing the dataset. Default is './input/train.csv'.")
@click.option("--target", required=True, type=str, help="The name of the target column.")
@click.option("--n_splits", default=5, type=int, help="Number of folds. Default is 5.")
@click.option("--binning_method", default="sturges", type=click.Choice(['sturges', 'quantile', 'kmeans', 'custom']), help="Method for binning the target variable.")
@click.option("--custom_bins", default=None, type=str, help="Comma-separated list of custom bin edges for 'custom' binning method.")
@click.option("--random_state", default=42, type=int, help="Seed for the random number generator. Default is 42.")
@click.option("--save_path", default=None, type=str, help="Optional path to save the CSV file. If None, the file is not saved.")
def regression_kfolds(file_path, target, n_splits, binning_method, custom_bins, random_state, save_path):
    """
    Creates stratified K-Fold indices for regression tasks with various binning methods from a CSV file.
    """
    data = pd.read_csv(file_path)
    data["kfold"] = -1

    if binning_method == "sturges":
        num_bins = int(np.floor(1 + np.log2(len(data))))
        data["bins"] = pd.cut(data[target], bins=num_bins, labels=False)
    elif binning_method == "quantile":
        num_bins = int(np.floor(1 + np.log2(len(data))))
        data["bins"] = pd.qcut(data[target], q=num_bins, labels=False)
    elif binning_method == "kmeans":
        num_bins = int(np.floor(1 + np.log2(len(data))))
        kmeans = KMeans(n_clusters=num_bins, random_state=random_state) # TODO: Validate if it produces the correct number of bins
        data["bins"] = kmeans.fit_predict(data[[target]])
    elif binning_method == "custom":
        if not custom_bins:
            # raise ValueError("Custom bins must be provided when using custom binning.")
            raise click.UsageError("Custom bins must be provided when using custom binning.")
        custom_bins = list(map(float, custom_bins.split(',')))
        data["bins"] = pd.cut(data[target], bins=custom_bins, labels=False, include_lowest=True)

    y = data["bins"].values
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for fold, (train_idx, val_idx) in enumerate(skf.split(data, y)):
        data.loc[val_idx, "kfold"] = fold

    data = data.drop("bins", axis=1)

    if save_path:
        data.to_csv(save_path, index=False)

    click.echo(f"Regression K-Folds created and saved to {save_path if save_path else 'not saved.'}")

if __name__ == "__main__":
    cli()
