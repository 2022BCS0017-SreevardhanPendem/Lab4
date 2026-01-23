import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ------------------ PATHS ------------------
DATA_PATH = "dataset/winequality-red.csv"
OUTPUT_DIR = "outputs"
MODEL_PATH = os.path.join(OUTPUT_DIR, "model.joblib")
RESULTS_PATH = os.path.join(OUTPUT_DIR, "results.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------ LOAD DATA ------------------
df = pd.read_csv(DATA_PATH, sep=";")

X = df.drop("quality", axis=1)
y = df["quality"]

# ------------------ STRATIFICATION BINS ------------------
# Required because target is continuous
y_bins = pd.qcut(y, q=5, labels=False, duplicates="drop")

# ------------------ TRAIN-TEST SPLIT (75/25, STRATIFIED) ------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    stratify=y_bins,
    random_state=42
)

# ------------------ MODEL ------------------
model = RandomForestRegressor(
    n_estimators=150,
    max_depth=15,
    random_state=42
)
model.fit(X_train, y_train)



# ------------------ EVALUATION ------------------
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# ------------------ OUTPUT ------------------
print(f"Mean Squared Error (MSE): {mse}")
print(f"R2 Score: {r2}")
importances = model.feature_importances_
feature_names = X.columns

indices = np.argsort(importances)[::-1][:6]
selected_features = list(feature_names[indices])
joblib.dump(
    {
        "model": model,
        "selected_features": selected_features
    },
    "outputs/model.joblib"
)

results = {
    "experiment_id": "EXP-08",
    "model": "Random Forest",
    "hyperparameters": "n_estimators=150, max_depth=15",
    "preprocessing": "None",
    "feature_selection": "All",
    "split": "75/25 (Stratified)",
    "mse": mse,
    "r2_score": r2
}

with open(RESULTS_PATH, "w") as f:
    json.dump(results, f, indent=4)
