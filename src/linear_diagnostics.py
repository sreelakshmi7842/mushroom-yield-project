from pathlib import Path
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ----------------------------------
# Paths
# ----------------------------------

PROCESSED = Path("data/processed")
MODELS = Path("models")
FIGURES = Path("reports/figures")

FIGURES.mkdir(parents=True, exist_ok=True)

# ----------------------------------
# Load Data
# ----------------------------------

X_train = np.load(PROCESSED / "X_train.npy")
X_test = np.load(PROCESSED / "X_test.npy")

y_train = np.load(PROCESSED / "y_train.npy")
y_test = np.load(PROCESSED / "y_test.npy")

# ----------------------------------
# Load Model
# ----------------------------------

model = joblib.load(
    MODELS / "linear_regression.joblib"
)

# ----------------------------------
# Predictions
# ----------------------------------

pred_train = model.predict(X_train)
pred_test = model.predict(X_test)

# ----------------------------------
# Residuals
# actual - predicted
# ----------------------------------

train_residuals = y_train - pred_train
test_residuals = y_test - pred_test

print("Residuals calculated.")

# ----------------------------------
# Plot 1
# Residuals vs Predicted
# ----------------------------------

plt.figure(figsize=(8,6))

plt.scatter(
    pred_test,
    test_residuals,
    alpha=0.7
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Predicted Yield (kg)")
plt.ylabel("Residual")

plt.title(
    "Linear Regression Residuals vs Predicted Yield"
)

plt.tight_layout()

plt.savefig(
    FIGURES / "residuals_linear.png"
)

plt.close()

# ----------------------------------
# Plot 2
# Residuals vs Humidity
# ----------------------------------

humidity_column = 1

plt.figure(figsize=(8,6))

plt.scatter(
    X_test[:, humidity_column],
    test_residuals,
    alpha=0.7
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Humidity")
plt.ylabel("Residual")

plt.title(
    "Residuals vs Humidity"
)

plt.tight_layout()

plt.savefig(
    FIGURES / "residuals_vs_humidity_linear.png"
)

plt.close()

print("Diagnostic plots saved.")