import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

train = pd.read_csv("./input/train.csv")
test = pd.read_csv("./input/test.csv")

# Split the data into features and target
X_train = train.drop(['id', 'Target'], axis=1)
X_test = test.drop(['id'], axis=1)

y_train = train.Target.values

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit and transform the training data
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Label encode the target
le = LabelEncoder()
y_train = le.fit_transform(y_train)

# Initialize the model
model = XGBClassifier()

# Fit the model
model.fit(X_train, y_train)

# Make predictions
preds = model.predict(X_test)

# Convert predictions back to original labels
preds_labels = le.inverse_transform(preds)

# Calculate the accuracy on train
train_preds = model.predict(X_train)
accuracy = accuracy_score(y_train, train_preds)
print(f"Accuracy={accuracy}")

# Prepare the submission dataframe
submission = pd.DataFrame({
    'id': test['id'],
    'Target': preds_labels
})

# Save the submission file
submission.to_csv('./output/submission.csv', index=False)
print("Submission is successfully saved!")