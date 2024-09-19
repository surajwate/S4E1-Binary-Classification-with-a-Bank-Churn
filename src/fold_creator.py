import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.cluster import KMeans
import click


def create_kfolds(file_path, n_splits=5, shuffle=True, random_state=42, save_path=None):
    """
    Creates K-Fold indices for a dataset loaded from a CSV file.

    Parameters:
    file_path (str): Path to the input CSV file containing the dataset.
    n_splits (int): Number of folds. Default is 5.
    shuffle (bool): Whether to shuffle the data. Default is True.
    random_state (int): Seed for the random number generator. Default is 42.
    save_path (str): Optional path to save the CSV file. If None, the file is not saved. Default is None.

    Returns:
    pd.DataFrame: DataFrame with an additional 'kfold' column.
    """
    data = pd.read_csv(file_path)
    data["kfold"] = -1
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    for fold, (train_idx, val_idx) in enumerate(kf.split(data)):
        data.loc[val_idx, "kfold"] = fold

    if save_path:
        data.to_csv(save_path, index=False)

    return data


@click.command()
@click.option("--file_path", default="./input/train.csv", type=str, help="Path to the input CSV file containing the dataset. Default is './input/train.csv'.")
@click.option("--n_splits", default=5, type=int, help="Number of folds. Default is 5.")
@click.option("--target", required=True, type=str, help="The name of the target column.")
@click.option("--random_state", default=42, type=int, help="Seed for the random number generator. Default is 42.")
@click.option("--save_path", default="./input/train_folds.csv", type=str, help="Optional path to save the CSV file. Default is './input/train_folds.csv'.")
def create_classification_kfolds(file_path, n_splits, target, random_state, save_path):
    """
    Creates stratified K-Fold indices for classification tasks from a CSV file.

    Parameters:
    file_path (str): Path to the input CSV file containing the dataset.
    target (str): The name of the target column.
    n_splits (int): Number of folds. Default is 5.
    random_state (int): Seed for the random number generator. Default is 42.
    save_path (str): Optional path to save the CSV file. Default is './input/train_folds.csv'.

    Returns:
    pd.DataFrame: DataFrame with an additional 'kfold' column.
    """
    # Read the input data
    data = pd.read_csv(file_path)
    data["kfold"] = -1

    # Extract the target variable values
    y = data[target].values

    # Initialize Stratified K-Fold with specified splits and random state
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Split data and assign fold numbers
    for fold, (train_idx, val_idx) in enumerate(skf.split(data, y)):
        data.loc[val_idx, "kfold"] = fold

    # Save the DataFrame with fold indices if save_path is provided
    if save_path:
        data.to_csv(save_path, index=False)

    return data


def create_regression_kfolds(
    file_path,
    target_column,
    n_splits=5,
    binning_method="sturges",
    custom_bins=None,
    random_state=42,
    save_path=None,
):
    """
    Creates stratified K-Fold indices for regression tasks with various binning methods from a CSV file.

    Parameters:
    file_path (str): Path to the input CSV file containing the dataset.
    target_column (str): The name of the target column.
    n_splits (int): Number of folds. Default is 5.
    binning_method (str): Method for binning the target variable. Options are 'sturges', 'quantile', 'kmeans', 'custom'. Default is 'sturges'.
    custom_bins (list): List of bin edges for custom binning. Required if binning_method is 'custom'.
    random_state (int): Seed for the random number generator. Default is 42.
    save_path (str): Optional path to save the CSV file. If None, the file is not saved. Default is None.

    Returns:
    pd.DataFrame: DataFrame with an additional 'kfold' column.
    """
    data = pd.read_csv(file_path)
    data["kfold"] = -1

    if binning_method == "sturges":
        num_bins = int(np.floor(1 + np.log2(len(data))))
        data["bins"] = pd.cut(data[target_column], bins=num_bins, labels=False)
    elif binning_method == "quantile":
        num_bins = int(np.floor(1 + np.log2(len(data))))
        data["bins"] = pd.qcut(data[target_column], q=num_bins, labels=False)
    elif binning_method == "kmeans":
        num_bins = int(np.floor(1 + np.log2(len(data))))
        kmeans = KMeans(n_clusters=num_bins, random_state=random_state)
        data["bins"] = kmeans.fit_predict(data[[target_column]])
    elif binning_method == "custom":
        if custom_bins is None:
            raise ValueError("Custom bins must be provided when using custom binning.")
        data["bins"] = pd.cut(
            data[target_column], bins=custom_bins, labels=False, include_lowest=True
        )
    else:
        raise ValueError(
            f"Invalid binning method: {binning_method}. Choose 'sturges', 'quantile', 'kmeans', or 'custom'."
        )

    y = data["bins"].values
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for fold, (train_idx, val_idx) in enumerate(skf.split(data, y)):
        data.loc[val_idx, "kfold"] = fold

    data = data.drop("bins", axis=1)

    if save_path:
        data.to_csv(save_path, index=False)

    return data


"""
from fold_creator import create_kfolds, create_classification_kfolds, create_regression_kfolds

# Example usage for K-Fold
kfold_data = create_kfolds(file_path="train.csv", n_splits=5, shuffle=True, random_state=42)
# Save manually if needed
kfold_data.to_csv("train_folds_kfold.csv", index=False)

# Example usage for Stratified K-Fold Classification with saving
stratified_classification_data = create_classification_kfolds(
    file_path="train.csv", target_column="target", n_splits=5, random_state=42, save_path="train_folds_classification.csv"
)

# Example usage for Stratified K-Fold Regression with Sturges' binning without saving
stratified_regression_data = create_regression_kfolds(
    file_path="train.csv", target_column="target", n_splits=5, binning_method='sturges', random_state=42
)
# Further processing or save manually
stratified_regression_data.to_csv("train_folds_regression_sturges.csv", index=False)

# Example usage for Stratified K-Fold Regression with custom binning and saving
custom_bins = [0, 1, 2, 3, 4.841, 5.1]
stratified_regression_custom_data = create_regression_kfolds(
    file_path="train.csv", target_column="target", n_splits=5, binning_method='custom', custom_bins=custom_bins, random_state=42, save_path="train_folds_regression_custom.csv"
)

"""

if __name__ == "__main__":
    # create_kfolds()
    create_classification_kfolds()
    # create_regression_kfolds()