import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython import get_ipython
import os

sns.set_style("whitegrid")
# sns.set_palette("coolwarm")

def is_jupyter_notebook():
    """
    Check if the code is running in a Jupyter notebook environment.

    Returns:
        bool: True if running in a Jupyter notebook or JupyterLab, False otherwise.
    """
    try:
        shell = get_ipython().__class__.__name__
        if "ZMQInteractiveShell" in shell:
            return True  # Jupyter notebook or JupyterLab
        elif "TerminalInteractiveShell" in shell:
            return False  # Terminal running IPython
        else:
            return False  # Other type (likely terminal)
    except NameError:
        return False  # Probably standard Python interpreter


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def categorical_feature(df, feature, target, show_plot=True, save_plot=False, plot_dir="./plots"):
    """
    Calculates and visualizes the distribution of a categorical feature with respect to a target variable.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        feature (str): The name of the categorical feature.
        target (str): The name of the target variable.
        show_plot (bool): Whether to display the plot. Default is True.
        save_plot (bool): Whether to save the plot to a file. Default is False.
        plot_dir (str): Directory to save the plot if save_plot is True. Default is './plots'.

    Returns:
        pd.DataFrame: A DataFrame containing the distribution of the feature, including the total count, total percentage,
                      percentages for each target class relative to the total, and percentages of each target class within
                      the feature category.

    Raises:
        ValueError: If the feature or target column is not found in the DataFrame.
    """
    # Validate input columns
    if feature not in df.columns:
        raise ValueError(f"Feature '{feature}' not found in DataFrame.")
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found in DataFrame.")

    # Calculate total counts and percentages
    total_counts = df[feature].value_counts()
    total_percentages = df[feature].value_counts(normalize=True) * 100

    # Calculate counts and percentages for each target class
    counts_per_class = df.groupby([feature, target]).size().unstack(fill_value=0)
    percentages_per_class_of_total = counts_per_class.div(len(df)) * 100  # Percentages relative to total
    percentages_within_feature = counts_per_class.div(counts_per_class.sum(axis=1), axis=0) * 100  # Percentages within feature

    # Combine all into a single DataFrame
    category_distribution = pd.DataFrame({
        'Total Count': total_counts,
        'Total Percentage': total_percentages,
    })

    # Add percentages per class relative to total
    for class_value in counts_per_class.columns:
        category_distribution[f"{class_value} of Total (%)"] = percentages_per_class_of_total[class_value]

    # Add percentages per class within feature category
    for class_value in counts_per_class.columns:
        category_distribution[f"{class_value} within {feature} (%)"] = percentages_within_feature[class_value]

    # Sort the DataFrame by total count
    category_distribution = category_distribution.sort_values('Total Count', ascending=False)

    # Plotting
    plt.figure(figsize=(12, 6))
    sns.countplot(
        data=df,
        x=feature,
        hue=target,
        order=category_distribution.index,
        palette="coolwarm"
    )
    plt.title(f"Distribution of '{feature}' by '{target}'")
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.grid(axis="y", linestyle="--", linewidth=0.5)
    plt.xticks(rotation=45)

    # Show or save the plot based on parameters
    if show_plot:
        plt.show()

    if save_plot:
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        plot_filename = f"{feature}-{target}-distribution.png"
        plt.savefig(os.path.join(plot_dir, plot_filename), bbox_inches='tight')
        print(f"Plot saved to {os.path.join(plot_dir, plot_filename)}")

    plt.close()  # Close the figure to free memory

    return category_distribution



def numerical_feature(df, feature, target=None, figsize=(15, 6), bins="sturges"):
    """
    Analyzes a numerical feature in a dataframe.
    Parameters:
    - df (pandas.DataFrame): The dataframe containing the data.
    - feature (str): The name of the numerical feature to analyze.
    - target (str, optional): The name of the target column for grouping the analysis. Default is None.
    - figsize (tuple, optional): The size of the figure. Default is (15, 6).
    - bins (int, str, optional): The number of bins for the histogram or the method to calculate it. Default is 'sturges'.
    Returns:
    - outliers_df (pandas.DataFrame): A dataframe containing the percentage of outliers in the data.
    - summary_df (pandas.DataFrame): A dataframe containing the overall statistics, lower outliers statistics, and upper outliers statistics.
    """

    if feature not in df.columns:
        raise ValueError(f"Column '{feature}' not found in the dataframe.")

    if target and target not in df.columns:
        raise ValueError(f"Column '{target}' not found in the dataframe.")

    # Calculate the number of bins if a method is provided
    if isinstance(bins, str):
        if bins == "sturges":
            bins = int(np.ceil(np.log2(len(df[feature])) + 1))
        elif bins == "rice":
            bins = int(np.ceil(2 * len(df[feature]) ** (1 / 3)))
        elif bins == "scott":
            bin_width = 3.5 * df[feature].std() * len(df[feature]) ** (-1 / 3)
            bins = int(np.ceil((df[feature].max() - df[feature].min()) / bin_width))
        elif bins == "fd":  # Freedman-Diaconis
            bin_width = (
                2
                * (df[feature].quantile(0.75) - df[feature].quantile(0.25))
                * len(df[feature]) ** (-1 / 3)
            )
            bins = int(np.ceil((df[feature].max() - df[feature].min()) / bin_width))
        else:
            raise ValueError(f"Unknown binning method: '{bins}'")

    # Create the figure and subplots
    fig, ax = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # First plot: Histogram with KDE
    sns.histplot(df, x=feature, kde=True, ax=ax[0], bins=bins, palette="coolwarm", hue=target)
    ax[0].set_title(f"Distribution of {feature} with KDE")
    ax[0].set_ylabel("Frequency")
    ax[0].grid(True, which="both", linestyle="--", linewidth=0.5)

    # Second plot: Boxplot of the feature by the target (if provided)
    if target:
        # Check seaborn version to decide whether to include the 'gap' parameter
        if sns.__version__ >= "0.13":
            sns.boxplot(df, x=feature, ax=ax[1], hue=target, gap=0.05, palette="coolwarm")
        else:
            sns.boxplot(df, x=feature, ax=ax[1], hue=target, palette="coolwarm")
        ax[1].set_title(f"Box Plot of {feature} by {target} Status")
    else:
        sns.boxplot(df, x=feature, ax=ax[1])
        ax[1].set_title(f"Box Plot of {feature}")

    ax[1].set_ylabel("")
    ax[1].grid(True, which="both", linestyle="--", linewidth=0.5)

    # Adjust layout for better spacing
    plt.tight_layout()

    # Display the plots
    if is_jupyter_notebook():
        plt.show()  # Show plot if in Jupyter notebook
    else:
        plt.savefig(f"./plots/{feature}-{target}-boxplot.png")

    # Calculate overall statistics
    overall_summary = df[feature].describe().to_frame().T
    overall_summary.index = [f"{feature}_Overall"]

    # Calculate the lower and upper bounds for outliers
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Identify lower and upper bound outliers
    lower_outliers = df[df[feature] < lower_bound]
    upper_outliers = df[df[feature] > upper_bound]

    # Get descriptive statistics for lower and upper outliers
    lower_outliers_summary = lower_outliers[feature].describe().to_frame().T
    lower_outliers_summary.index = [f"{feature}_Lower_Outliers"]

    upper_outliers_summary = upper_outliers[feature].describe().to_frame().T
    upper_outliers_summary.index = [f"{feature}_Upper_Outliers"]

    # Combine overall statistics with lower and upper outlier statistics
    summary_df = pd.concat(
        [overall_summary, lower_outliers_summary, upper_outliers_summary]
    )

    # Print the percentage of outliers in the data
    outlier_percentage = ((len(lower_outliers) + len(upper_outliers)) / len(df)) * 100
    lower_outliers_percentage = (len(lower_outliers) / len(df)) * 100
    upper_outliers_percentage = (len(upper_outliers) / len(df)) * 100
    outliers_df = pd.DataFrame(
        {
            "Outlier Percentage": [outlier_percentage],
            "Lower Outliers Percentage": [lower_outliers_percentage],
            "Upper Outliers Percentage": [upper_outliers_percentage],
        }
    )

    return outliers_df, summary_df


def missing_values(dataframe):
    """
    Generates a summary of missing values in the dataframe.

    Parameters:
    dataframe (pd.DataFrame): The input dataframe to analyze.

    Returns:
    pd.DataFrame: A dataframe containing the count and percentage of missing values,
                  along with the data type of each column that has missing values.
    """
    missing_values_summary = pd.DataFrame(
        {
            "Missing Count": dataframe.isnull().sum(),
            "Missing Percentage": (
                dataframe.isnull().sum() / len(dataframe) * 100
            ).round(2),
            "Data Type": dataframe.dtypes,
        }
    )

    # Filter out columns with no missing values
    missing_values_summary = missing_values_summary[
        missing_values_summary["Missing Count"] > 0
    ]

    return missing_values_summary