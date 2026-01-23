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

# ------------------ INITIAL TRAIN-TEST SPLIT (FOR FEATURE IMPORTANCE) ------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.40,
    stratify=y_bins,
    random_state=42
)

# ------------------ INITIAL MODEL (ALL FEATURES) ------------------
base_model = RandomForestRegressor(
    n_estimators=150,
    max_depth=15,
    random_state=42
)

base_model.fit(X_train, y_train)

# ------------------ FEATURE SELECTION (TOP 6) ------------------
importances = base_model.feature_importances_
feature_names = X.columns

indices = np.argsort(importances)[::-1][:6]
selected_features = list(feature_names[indices])

print("Selected Features:", selected_features)

# ------------------ REDUCE DATA TO SELECTED FEATURES ------------------
X_selected = X[selected_features]

X_train_sel, X_test_sel, y_train_sel, y_test_sel = train_test_split(
    X_selected,
    y,
    test_size=0.25,
    stratify=y_bins,
    random_state=42
)

# ------------------ FINAL MODEL (ONLY SELECTED FEATURES) ------------------
final_model = RandomForestRegressor(
    n_estimators=150,
    max_depth=15,
    random_state=42
)

final_model.fit(X_train_sel, y_train_sel)

# ------------------ EVALUATION ------------------
y_pred = final_model.predict(X_test_sel)

mse = mean_squared_error(y_test_sel, y_pred)
r2 = r2_score(y_test_sel, y_pred)

print(f"Final Mean Squared Error (MSE): {mse}")
print(f"Final R2 Score: {r2}")

# ------------------ SAVE ARTIFACT (DEPLOYMENT-READY) ------------------
joblib.dump(
    {
        "model": final_model,
        "selected_features": selected_features
    },
    MODEL_PATH
)

# ------------------ SAVE METRICS ------------------
results = {
    "experiment_id": "EXP-08",
    "model": "Random Forest (Top 6 Features)",
    "hyperparameters": "n_estimators=150, max_depth=15",
    "feature_selection": "Top 6 via feature_importances_",
    "split": "60/40 (Stratified)",
    "selected_features": selected_features,
    "mse": mse,
    "r2_score": r2
}

with open(RESULTS_PATH, "w") as f:
    json.dump(results, f, indent=4)
