import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# =====================================
# Load Day 8 Train/Test Split
# =====================================

train_df = pd.read_csv("data/processed/train.csv")
test_df = pd.read_csv("data/processed/test.csv")

print("Train Shape:", train_df.shape)
print("Test Shape :", test_df.shape)

# =====================================
# Features and Target
# =====================================

FEATURES = [
    "temperature_c",
    "humidity_pct",
    "co2_ppm"
]

TARGET = "yield_kg"

X_train = train_df[FEATURES]
y_train = train_df[TARGET]

X_test = test_df[FEATURES]
y_test = test_df[TARGET]

# =====================================
# Linear Regression Baseline
# =====================================

linear_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])

linear_pipeline.fit(X_train, y_train)

linear_preds = linear_pipeline.predict(X_test)

linear_mae = mean_absolute_error(y_test, linear_preds)

linear_rmse = np.sqrt(
    mean_squared_error(y_test, linear_preds)
)

linear_r2 = r2_score(y_test, linear_preds)

# =====================================
# Random Forest Regressor
# =====================================

rf_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ))
])

rf_pipeline.fit(X_train, y_train)

rf_preds = rf_pipeline.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_preds)

rf_rmse = np.sqrt(
    mean_squared_error(y_test, rf_preds)
)

rf_r2 = r2_score(y_test, rf_preds)

# =====================================
# Comparison Table
# =====================================

comparison = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Random Forest"
    ],
    "MAE": [
        round(linear_mae, 3),
        round(rf_mae, 3)
    ],
    "RMSE": [
        round(linear_rmse, 3),
        round(rf_rmse, 3)
    ],
    "R2": [
        round(linear_r2, 3),
        round(rf_r2, 3)
    ]
})

print("\n==============================")
print("MODEL COMPARISON")
print("==============================")
print(comparison)

# =====================================
# Feature Importance
# =====================================

rf_model = rf_pipeline.named_steps["model"]

importance_df = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": rf_model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=True
)

print("\n==============================")
print("FEATURE IMPORTANCE")
print("==============================")
print(importance_df)

# =====================================
# Save Feature Importance Plot
# =====================================

os.makedirs("plots", exist_ok=True)

plt.figure(figsize=(8, 5))

plt.barh(
    importance_df["Feature"],
    importance_df["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Random Forest Feature Importance")

plt.tight_layout()

plt.savefig(
    "plots/rf_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nFeature importance plot saved:")
print("plots/rf_feature_importance.png")

# =====================================
# Save Random Forest Model
# =====================================

os.makedirs("models", exist_ok=True)

joblib.dump(
    rf_pipeline,
    "models/random_forest.joblib"
)

print("\nRandom Forest model saved:")
print("models/random_forest.joblib")

# =====================================
# Verify Saved Model
# =====================================

loaded_model = joblib.load(
    "models/random_forest.joblib"
)

sample_predictions = loaded_model.predict(
    X_test.iloc[:5]
)

print("\nSample Predictions:")
print(sample_predictions)

# =====================================
# Interpretation
# =====================================

print("\n==============================")
print("INTERPRETATION")
print("==============================")

best_model = comparison.loc[
    comparison["R2"].idxmax(),
    "Model"
]

print(f"Best Model: {best_model}")

if rf_r2 > linear_r2:
    print(
        "\nRandom Forest performed better than "
        "Linear Regression."
    )
    print(
        "This suggests mushroom yield has "
        "nonlinear relationships with "
        "temperature, humidity and CO2."
    )
    print(
        "The added complexity of Random Forest "
        "is justified by the improved accuracy."
    )
else:
    print(
        "\nRandom Forest showed little or no "
        "improvement over Linear Regression."
    )
    print(
        "In this case, Linear Regression may be "
        "preferred because it is simpler and "
        "more interpretable."
    )