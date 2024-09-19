$models = @("logistic_regression", "random_forest", "xgboost", "lightgbm", "catboost", "mlp", "gradient_boosting", "knn", "svm")

foreach ($model in $models) {
    python main.py --model $model
}
