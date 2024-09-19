from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    BaggingClassifier,
    StackingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier


# Define base models

base_models = [
    ('xgboost', XGBClassifier()),
    ('lightgbm', LGBMClassifier()),
    ('catboost', CatBoostClassifier(verbose=0)),
    ('gradient_boosting', GradientBoostingClassifier())
]

# Meta model
meta_model = LogisticRegression(max_iter=5000)

# Define the stacking classifier
stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_model)

models = {
    "logistic_regression": LogisticRegression(max_iter=1000),
    "random_forest": RandomForestClassifier(),
    "decision_tree": DecisionTreeClassifier(),
    "svm": SVC(),
    "gradient_boosting": GradientBoostingClassifier(),
    "xgboost": XGBClassifier(),
    "knn": KNeighborsClassifier(),
    "extra_trees": ExtraTreesClassifier(),
    "ada_boost": AdaBoostClassifier(algorithm='SAMME'),
    "bagging": BaggingClassifier(),
    "lightgbm": LGBMClassifier(),
    "catboost": CatBoostClassifier(verbose=0),
    "stacking": stacking_model,
}
