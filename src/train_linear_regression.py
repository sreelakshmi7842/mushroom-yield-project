from pathlib import Path
import json
import joblib
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# --------------------------------------------------
# Create required folders
# --------------------------------------------------

MODELS = Path("models")
REPORTS = Path("reports")
PROCESSED = Path("data/processed")

MODELS.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)

# --------------------------------------------------
# Load train/test arrays from previous phase
# --------------------------------------------------

X_train = np.load(PROCESSED / "X_train.npy")
X_test = np.load(PROCESSED / "X_test.npy")

y_train = np.load(PROCESSED / "y_train.npy")
y_test = np.load(PROCESSED / "y_test.npy")

# --------------------------------------------------
# Train Linear Regression Model
# --------------------------------------------------

model = LinearRegression()

model.fit(X_train, y_train)

# --------------------------------------------------
# Predictions
# --------------------------------------------------

pred_train = model.predict(X_train)
pred_test = model.predict(X_test)

# --------------------------------------------------
# Evaluation Metrics
# --------------------------------------------------

mae = mean_absolute_error(y_test, pred_test)

rmse = np.sqrt(
    mean_squared_error(y_test, pred_test)
)

r2 = r2_score(y_test, pred_test)

# --------------------------------------------------
# Print Metrics
# --------------------------------------------------

print("\n===== MODEL PERFORMANCE =====")

print(f"Test MAE  : {mae:.3f} kg")
print(f"Test RMSE : {rmse:.3f} kg")
print(f"Test R²   : {r2:.3f}")

# --------------------------------------------------
# Inspect Coefficients
# --------------------------------------------------

feature_names = [
    "temperature_c",
    "humidity_pct",
    "co2_ppm",
]

print("\n===== COEFFICIENTS =====")

for name, coef in zip(feature_names, model.coef_):

    if coef > 0:
        relation = "increases"
    else:
        relation = "decreases"

    print(
        f"{name}: {coef:.4f} "
        f"(higher {name} tends to {relation} yield)"
    )

# --------------------------------------------------
# Save Model
# --------------------------------------------------

joblib.dump(
    model,
    MODELS / "linear_regression.joblib"
)

# --------------------------------------------------
# Save Metrics JSON
# --------------------------------------------------

metrics = {
    "mae": float(mae),
    "rmse": float(rmse),
    "r2": float(r2),
}

with open(
    REPORTS / "metrics_linear.json",
    "w"
) as f:
    json.dump(metrics, f, indent=4)

# --------------------------------------------------
# Save Coefficient Interpretation
# --------------------------------------------------

with open(
    REPORTS / "metrics_linear.md",
    "w",
    encoding="utf-8"
) as f:

    f.write("# Linear Regression Results\n\n")

    f.write("## Test Metrics\n\n")
    f.write(f"- MAE: {mae:.3f} kg\n")
    f.write(f"- RMSE: {rmse:.3f} kg\n")
    f.write(f"- R²: {r2:.3f}\n\n")

    f.write("## Coefficient Interpretation\n\n")

    for name, coef in zip(feature_names, model.coef_):

        if coef > 0:
            direction = "positive"
        else:
            direction = "negative"

        f.write(
            f"- **{name}**: coefficient = "
            f"{coef:.4f} ({direction} relationship with yield)\n"
        )

    f.write("\n## Baseline Assessment\n\n")

    if r2 >= 0.70:
        f.write(
            "R² indicates a strong baseline model.\n"
        )
    elif r2 >= 0.50:
        f.write(
            "R² indicates a reasonable baseline model.\n"
        )
    else:
        f.write(
            "R² is relatively low; additional feature engineering "
            "or more advanced models may improve performance.\n"
        )

# --------------------------------------------------
# Success Messages
# --------------------------------------------------

print("\n===== SAVED FILES =====")

print(
    f"Model   : {MODELS / 'linear_regression.joblib'}"
)

print(
    f"Metrics : {REPORTS / 'metrics_linear.json'}"
)

print(
    f"Report  : {REPORTS / 'metrics_linear.md'}"
)

print("\nTraining completed successfully.")