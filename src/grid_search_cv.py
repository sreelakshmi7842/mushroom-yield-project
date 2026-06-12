from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import (
    GridSearchCV,
    TimeSeriesSplit
)

# ==================================================
# Create Output Directories
# ==================================================

Path("models").mkdir(parents=True, exist_ok=True)
Path("reports").mkdir(parents=True, exist_ok=True)

# ==================================================
# Load Train/Test Data
# ==================================================

X_train = np.load("data/processed/X_train.npy")
X_test = np.load("data/processed/X_test.npy")

y_train = np.load("data/processed/y_train.npy")
y_test = np.load("data/processed/y_test.npy")

print("===== DATA LOADED =====")
print("Train samples:", len(X_train))
print("Test samples :", len(X_test))

# ==================================================
# Parameter Grid
# ==================================================
#
# n_estimators:
#   Number of trees in the forest.
#
# max_depth:
#   Controls tree depth and model complexity.
#
# min_samples_leaf:
#   Minimum samples required in each leaf node.
#
# Grid kept intentionally small so search
# completes within minutes on a laptop.
# ==================================================

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 8, 16],
    "min_samples_leaf": [1, 3, 5],
}

# ==================================================
# Time Series Cross Validation
# ==================================================

tscv = TimeSeriesSplit(n_splits=3)

rf = RandomForestRegressor(
    random_state=42,
    n_jobs=-1
)

# ==================================================
# Grid Search
# ==================================================

print("\n===== RUNNING GRID SEARCH =====")

start_time = time.time()

search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=tscv,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
    refit=True,
    verbose=1
)

search.fit(X_train, y_train)

runtime = time.time() - start_time

# ==================================================
# Best Results
# ==================================================

best_params = search.best_params_
best_cv_mae = -search.best_score_

print("\n===== BEST PARAMETERS =====")
print(best_params)

print(f"\nBest CV MAE: {best_cv_mae:.3f}")
print(f"Runtime: {runtime:.2f} seconds")

# ==================================================
# Save CV Results (Transparency)
# ==================================================

cv_results = pd.DataFrame(search.cv_results_)

cv_results.head(20).to_csv(
    "reports/gridsearch_cv_results_head.csv",
    index=False
)

print(
    "\nSaved: reports/gridsearch_cv_results_head.csv"
)

# ==================================================
# Grid Edge Check (Fixed for None)
# ==================================================

print("\n===== GRID EDGE CHECK =====")

for param_name, values in param_grid.items():

    best_value = best_params[param_name]

    if None in values:

        numeric_values = [
            v for v in values
            if v is not None
        ]

        if best_value is None:
            print(
                f"{param_name}: selected None "
                "(unlimited depth)."
            )

        elif best_value == max(numeric_values):
            print(
                f"{param_name}: best value is at "
                f"upper edge ({best_value})."
            )

    else:

        if best_value == min(values):
            print(
                f"{param_name}: best value is at "
                f"lower edge ({best_value})."
            )

        elif best_value == max(values):
            print(
                f"{param_name}: best value is at "
                f"upper edge ({best_value})."
            )

# ==================================================
# Best Model
# ==================================================

best_model = search.best_estimator_

# ==================================================
# Held-Out Test Evaluation
# ==================================================

test_pred = best_model.predict(X_test)

test_mae = mean_absolute_error(
    y_test,
    test_pred
)

test_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        test_pred
    )
)

test_r2 = r2_score(
    y_test,
    test_pred
)

print("\n===== TEST SET RESULTS =====")

print(f"MAE  : {test_mae:.3f}")
print(f"RMSE : {test_rmse:.3f}")
print(f"R²   : {test_r2:.3f}")

# ==================================================
# Save Tuned Model
# ==================================================

model_path = "models/random_forest_tuned.joblib"

joblib.dump(
    best_model,
    model_path
)

print(f"\nSaved model: {model_path}")

# ==================================================
# Save Best Parameters
# ==================================================

params_path = "models/rf_best_params.json"

with open(
    params_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        best_params,
        f,
        indent=2
    )

print(f"Saved params: {params_path}")

# ==================================================
# Save Summary CSV
# ==================================================

summary = pd.DataFrame({
    "Metric": [
        "Best CV MAE",
        "Test MAE",
        "Test RMSE",
        "Test R2",
        "Runtime Seconds"
    ],
    "Value": [
        best_cv_mae,
        test_mae,
        test_rmse,
        test_r2,
        runtime
    ]
})

summary.to_csv(
    "reports/gridsearch_summary.csv",
    index=False
)

print(
    "Saved summary: reports/gridsearch_summary.csv"
)

# ==================================================
# Final Notes
# ==================================================

print("\n===== COMPLETED =====")
print("Grid search used TimeSeriesSplit only on training data.")
print("Test data was evaluated once after tuning.")
print("Model, parameters, and search results were saved.")