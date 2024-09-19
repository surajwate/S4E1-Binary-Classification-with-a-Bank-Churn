import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from feature_engineering import FeatureEngineer
from sklearn.preprocessing import PolynomialFeatures

import os
import argparse
import joblib

import config
import model_dispatcher

import time
import logging

# Set up logging
logging.basicConfig(
    filename=config.LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s"
)

def run(fold, model):
    # Import the data
    df = pd.read_csv(config.TRAINING_FILE)

    df['GeoGen'] = df['Geography'] + '_' + df['Gender']
    df['Salary/Age'] = df['EstimatedSalary'] / df['Age']

    # Split the data into training and testing
    train = df[df.kfold != fold].reset_index(drop=True)
    test = df[df.kfold == fold].reset_index(drop=True)

    # Split the data into features and target
    target_features = ['Exited']

    X_train = train.drop(['id', 'CustomerId', 'kfold'] + target_features, axis=1)
    X_test = test.drop(['id', 'CustomerId', 'kfold'] + target_features, axis=1)

    y_train = train[target_features].values
    y_test = test[target_features].values

    # Reshape the target arrays
    y_train = y_train.reshape(-1)
    y_test = y_test.reshape(-1)

    # Define features
    categorical_features = ['Geography', 'Gender', 'IsActiveMember', 'Surname', 'GeoGen']
    numerical_features = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary', 'NumOfProducts', 'HasCrCard', 'Tenure']

    # Create a column transformer for one-hot encoding and standard scaling
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    # Create a pipeline with the preprocessor and the model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model_dispatcher.models[model])
    ])


    try:
        start = time.time()

        # logging.info(f"Fold={fold}, Model={model}")

        # Fit the model
        pipeline.fit(X_train, y_train)

        # Make probability predictions
        preds = pipeline.predict_proba(X_test)[:, 1]

        end = time.time()
        time_taken = end - start

        # Calculate the AUC-ROC
        auc = roc_auc_score(y_test, preds)

        logging.info(f"Fold={fold}, AUC = {auc:.4f}, Time Taken={time_taken:.2f}sec")
        print(f"Fold={fold}, AUC = {auc:.4f}, Time Taken={time_taken:.2f}sec")

        # Save the model
        joblib.dump(pipeline, os.path.join(config.MODEL_OUTPUT, f"model_{fold}.bin"))
    except Exception as e:
        logging.exception(f"Error occurred for Fold={fold}, Model={model}: {str(e)}")
    

if __name__ == '__main__':
    # Initialize the ArgumentParser class of argparse
    parser = argparse.ArgumentParser()

    # Add the arguments to the parser
    parser.add_argument("--fold", type=int)
    parser.add_argument("--model", type=str)

    # Read the arguments from the command line
    args = parser.parse_args()

    # Run the fold specified by the command line arguments
    run(fold=args.fold, model=args.model)

