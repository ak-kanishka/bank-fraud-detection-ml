import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import pickle

# Sample dataset
data = {
    "amount": [100, 2000, 150, 5000, 300, 7000, 120, 10000],
    "device_risk": [1, 8, 2, 9, 1, 10, 2, 9],
    "ip_risk": [1, 7, 2, 8, 1, 10, 2, 9],
    "fraud": [0, 1, 0, 1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

X = df[["amount", "device_risk", "ip_risk"]]
y = df["fraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    eval_metric="logloss"
)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy)

# Save model
with open("fraud_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved successfully!")