from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==================================================
# Create output directories
# ==================================================

Path("models").mkdir(parents=True, exist_ok=True)
Path("reports/figures").mkdir(parents=True, exist_ok=True)

# ==================================================
# Load train/test data
# ==================================================

X_train = np.load("data/processed/X_train.npy")
X_test = np.load("data/processed/X_test.npy")

y_train = np.load("data/processed/y_train.npy")
y_test = np.load("data/processed/y_test.npy")

print("===== DATA LOADED =====")
print("X_train shape:", X_train.shape)
print("X_test shape :", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape :", y_test.shape)

# ==================================================
# Train Linear Regression Baseline
# ==================================================

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

linear_pred = linear_model.predict(X_test)

linear_mae = mean_absolute_error(y_test, linear_pred)
linear_rmse = np.sqrt(mean_squared_error(y_test, linear_pred))
linear_r2 = r2_score(y_test, linear_pred)

print("\n===== LINEAR REGRESSION RESULTS =====")
print(f"MAE  : {linear_mae:.2f} kg")
print(f"RMSE : {linear_rmse:.2f} kg")
print(f"R²   : {linear_r2:.3f}")

# ==================================================
# Train Random Forest on TRAIN set only
# ==================================================

rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

# ==================================================
# Evaluate Random Forest
# ==================================================

rf_pred = rf.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2 = r2_score(y_test, rf_pred)

print("\n===== RANDOM FOREST RESULTS =====")
print(f"MAE  : {rf_mae:.2f} kg")
print(f"RMSE : {rf_rmse:.2f} kg")
print(f"R²   : {rf_r2:.3f}")

# ==================================================
# Model Comparison Table
# ==================================================

comparison = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest"],
    "MAE": [linear_mae, rf_mae],
    "RMSE": [linear_rmse, rf_rmse],
    "R²": [linear_r2, rf_r2]
})

print("\n===== MODEL COMPARISON =====")
print(comparison.to_string(index=False))

# Save comparison table

comparison.to_csv(
    "reports/model_comparison.csv",
    index=False
)

# ==================================================
# Feature Importance Analysis
# ==================================================

feature_names = [
    "temperature_c",
    "humidity_pct",
    "co2_ppm"
]

importances = rf.feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values("Importance", ascending=True)

print("\n===== FEATURE IMPORTANCES =====")
print(importance_df.to_string(index=False))

# ==================================================
# Plot Feature Importances
# ==================================================

plt.figure(figsize=(7, 4))
plt.barh(
    importance_df["Feature"],
    importance_df["Importance"]
)

plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.title("Random Forest Feature Importances")
plt.tight_layout()

plot_path = "reports/figures/rf_feature_importance.png"

plt.savefig(
    plot_path,
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print(f"\nFeature importance plot saved to: {plot_path}")

# ==================================================
# Interpretation
# ==================================================

most_important_feature = importance_df.iloc[-1]["Feature"]

print("\n===== INTERPRETATION =====")
print(
    f"The most influential predictor was "
    f"'{most_important_feature}', indicating that it "
    f"contributed the most to Random Forest predictions."
)

# ==================================================
# Save Random Forest Model
# ==================================================

model_path = "models/random_forest.joblib"

joblib.dump(rf, model_path)

print(f"\nModel saved to: {model_path}")

# ==================================================
# Complexity Justification
# ==================================================

print("\n===== CONCLUSION =====")

if rf_r2 > linear_r2:
    print(
        "Random Forest achieved a higher R² score than "
        "Linear Regression. This suggests that nonlinear "
        "relationships may exist in the data and the added "
        "model complexity is likely justified."
    )
else:
    print(
        "Random Forest did not meaningfully outperform "
        "Linear Regression. The additional complexity may "
        "not be justified for this dataset."
    )