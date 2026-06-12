import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

# =====================================
# Load Data
# =====================================

train_df = pd.read_csv("data/processed/train.csv")
test_df = pd.read_csv("data/processed/test.csv")

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

print("Train Shape:", train_df.shape)
print("Test Shape :", test_df.shape)

# =====================================
# TimeSeriesSplit
# =====================================

tscv = TimeSeriesSplit(n_splits=3)

# =====================================
# Linear Regression Pipeline
# =====================================

linear_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])

# =====================================
# Random Forest Pipeline
# =====================================

rf_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ))
])

# =====================================
# Cross Validation MAE
# =====================================

linear_cv_scores = -cross_val_score(
    linear_pipeline,
    X_train,
    y_train,
    cv=tscv,
    scoring="neg_mean_absolute_error"
)

rf_cv_scores = -cross_val_score(
    rf_pipeline,
    X_train,
    y_train,
    cv=tscv,
    scoring="neg_mean_absolute_error"
)

print("\n==============================")
print("CROSS VALIDATION RESULTS")
print("==============================")

print("\nLinear Regression CV MAE:")
print(linear_cv_scores)

print("\nRandom Forest CV MAE:")
print(rf_cv_scores)

print("\nAverage Linear CV MAE:",
      round(linear_cv_scores.mean(), 3))

print("Average RF CV MAE:",
      round(rf_cv_scores.mean(), 3))

# =====================================
# Train Models on Full Train Set
# =====================================

linear_pipeline.fit(X_train, y_train)
rf_pipeline.fit(X_train, y_train)

# =====================================
# Train MAE
# =====================================

linear_train_preds = linear_pipeline.predict(X_train)
rf_train_preds = rf_pipeline.predict(X_train)

linear_train_mae = mean_absolute_error(
    y_train,
    linear_train_preds
)

rf_train_mae = mean_absolute_error(
    y_train,
    rf_train_preds
)

# =====================================
# Test MAE
# =====================================

linear_test_preds = linear_pipeline.predict(X_test)
rf_test_preds = rf_pipeline.predict(X_test)

linear_test_mae = mean_absolute_error(
    y_test,
    linear_test_preds
)

rf_test_mae = mean_absolute_error(
    y_test,
    rf_test_preds
)

# =====================================
# Overfitting Analysis
# =====================================

print("\n==============================")
print("OVERFITTING ANALYSIS")
print("==============================")

print("\nLinear Regression")
print("Train MAE:", round(linear_train_mae, 3))
print("Test MAE :", round(linear_test_mae, 3))

print("\nRandom Forest")
print("Train MAE:", round(rf_train_mae, 3))
print("Test MAE :", round(rf_test_mae, 3))

if rf_train_mae < (rf_test_mae * 0.5):
    rf_comment = (
        "Potential overfitting detected. "
        "Train MAE is much lower than Test MAE."
    )
else:
    rf_comment = (
        "No major overfitting detected."
    )

print("\nRF Interpretation:")
print(rf_comment)

# =====================================
# CV Results Table
# =====================================

cv_results = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Random Forest"
    ],
    "Mean CV MAE": [
        round(linear_cv_scores.mean(), 3),
        round(rf_cv_scores.mean(), 3)
    ],
    "CV Std": [
        round(linear_cv_scores.std(), 3),
        round(rf_cv_scores.std(), 3)
    ],
    "Test MAE": [
        round(linear_test_mae, 3),
        round(rf_test_mae, 3)
    ]
})

print("\n==============================")
print("CV RESULTS TABLE")
print("==============================")
print(cv_results)

# =====================================
# Save Chart
# =====================================

os.makedirs("reports/figures", exist_ok=True)

chart_path = "reports/figures/cv_mae_comparison.png"

plt.figure(figsize=(8, 5))

plt.bar(
    cv_results["Model"],
    cv_results["Mean CV MAE"]
)

plt.ylabel("Mean CV MAE")
plt.title("Cross Validation MAE Comparison")

plt.tight_layout()

plt.savefig(
    chart_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nChart saved:")
print(chart_path)

# =====================================
# Save Report
# =====================================

os.makedirs("reports", exist_ok=True)

report_path = "reports/cv_results.md"

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write("# Day 12 - Cross Validation Results\n\n")

    f.write("## TimeSeriesSplit\n\n")
    f.write("n_splits = 3\n\n")

    f.write("## Linear Regression CV Scores\n\n")
    f.write(f"{linear_cv_scores}\n\n")

    f.write("Average CV MAE: ")
    f.write(f"{linear_cv_scores.mean():.3f}\n\n")

    f.write("## Random Forest CV Scores\n\n")
    f.write(f"{rf_cv_scores}\n\n")

    f.write("Average CV MAE: ")
    f.write(f"{rf_cv_scores.mean():.3f}\n\n")

    f.write("## CV Results Summary\n\n")
    f.write(cv_results.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Overfitting Analysis\n\n")

    f.write(
        f"Linear Regression Train MAE: "
        f"{linear_train_mae:.3f}\n\n"
    )

    f.write(
        f"Linear Regression Test MAE: "
        f"{linear_test_mae:.3f}\n\n"
    )

    f.write(
        f"Random Forest Train MAE: "
        f"{rf_train_mae:.3f}\n\n"
    )

    f.write(
        f"Random Forest Test MAE: "
        f"{rf_test_mae:.3f}\n\n"
    )

    f.write("### Interpretation\n\n")
    f.write(f"{rf_comment}\n\n")

    f.write("## CV MAE Comparison Chart\n\n")
    f.write(
        "![CV MAE Comparison]"
        "(figures/cv_mae_comparison.png)\n"
    )

print("\nReport saved:")
print(report_path)

