from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# ==================================================
# Load Data
# ==================================================

X_train = np.load("data/processed/X_train.npy")
X_test = np.load("data/processed/X_test.npy")

y_train = np.load("data/processed/y_train.npy")
y_test = np.load("data/processed/y_test.npy")

# ==================================================
# Models
# ==================================================

rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

lin = LinearRegression()

# ==================================================
# TimeSeries Cross Validation
# ==================================================

tscv = TimeSeriesSplit(n_splits=5)

rf_cv_scores = -cross_val_score(
    rf,
    X_train,
    y_train,
    cv=tscv,
    scoring="neg_mean_absolute_error"
)

lin_cv_scores = -cross_val_score(
    lin,
    X_train,
    y_train,
    cv=tscv,
    scoring="neg_mean_absolute_error"
)

print("\n===== CROSS VALIDATION RESULTS =====")

print(
    f"RF CV MAE: {rf_cv_scores.mean():.3f} "
    f"+/- {rf_cv_scores.std():.3f}"
)

print(
    f"Linear CV MAE: {lin_cv_scores.mean():.3f} "
    f"+/- {lin_cv_scores.std():.3f}"
)

# ==================================================
# Hold-Out Test Evaluation
# ==================================================

rf.fit(X_train, y_train)
lin.fit(X_train, y_train)

rf_train_pred = rf.predict(X_train)
rf_test_pred = rf.predict(X_test)

lin_train_pred = lin.predict(X_train)
lin_test_pred = lin.predict(X_test)

rf_train_mae = mean_absolute_error(y_train, rf_train_pred)
rf_test_mae = mean_absolute_error(y_test, rf_test_pred)

lin_train_mae = mean_absolute_error(y_train, lin_train_pred)
lin_test_mae = mean_absolute_error(y_test, lin_test_pred)

# ==================================================
# Comparison Table
# ==================================================

results = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest"],
    "CV Mean MAE": [
        lin_cv_scores.mean(),
        rf_cv_scores.mean()
    ],
    "CV Std": [
        lin_cv_scores.std(),
        rf_cv_scores.std()
    ],
    "Train MAE": [
        lin_train_mae,
        rf_train_mae
    ],
    "Test MAE": [
        lin_test_mae,
        rf_test_mae
    ]
})

print("\n===== MODEL COMPARISON =====")
print(results.round(3))

# ==================================================
# Overfitting Analysis
# ==================================================

print("\n===== OVERFITTING ANALYSIS =====")

for model_name, train_mae, test_mae in [
    ("Linear Regression", lin_train_mae, lin_test_mae),
    ("Random Forest", rf_train_mae, rf_test_mae),
]:
    print(f"\n{model_name}")
    print(f"Train MAE: {train_mae:.3f}")
    print(f"Test MAE : {test_mae:.3f}")

    if train_mae < (0.5 * test_mae):
        print("Potential overfitting detected.")
    else:
        print("No strong evidence of overfitting.")

# ==================================================
# CV Score Plot
# ==================================================

Path("reports/figures").mkdir(
    parents=True,
    exist_ok=True
)

plt.figure(figsize=(8, 4))

folds = np.arange(1, len(rf_cv_scores) + 1)

plt.plot(
    folds,
    rf_cv_scores,
    marker="o",
    label="Random Forest"
)

plt.plot(
    folds,
    lin_cv_scores,
    marker="o",
    label="Linear Regression"
)

plt.xlabel("Fold")
plt.ylabel("MAE")
plt.title("TimeSeriesSplit Cross-Validation MAE")
plt.legend()

plt.tight_layout()

plt.savefig(
    "reports/figures/cv_mae_scores.png",
    dpi=150
)

plt.close()

print(
    "\nCV plot saved to "
    "reports/figures/cv_mae_scores.png"
)

# ==================================================
# Save Markdown Report
# ==================================================

Path("reports").mkdir(
    parents=True,
    exist_ok=True
)

with open(
    "reports/cv_results.md",
    "w",
    encoding="utf-8"
) as f:

    f.write("# Cross Validation Results\n\n")

    f.write("## TimeSeriesSplit Configuration\n")
    f.write("- n_splits = 5\n")
    f.write("- No test data used during CV\n\n")

    f.write("## Model Performance\n\n")

    f.write(results.round(3).to_markdown(index=False))
    f.write("\n\n")

    f.write("## Interpretation\n\n")

    f.write(
        f"- Random Forest CV MAE: "
        f"{rf_cv_scores.mean():.3f} ± "
        f"{rf_cv_scores.std():.3f}\n"
    )

    f.write(
        f"- Linear Regression CV MAE: "
        f"{lin_cv_scores.mean():.3f} ± "
        f"{lin_cv_scores.std():.3f}\n"
    )

    f.write(
        "\nHigher standard deviation across folds "
        "indicates greater variability and less stable "
        "performance.\n"
    )

    f.write(
        "\nOverfitting was assessed by comparing "
        "training MAE and test MAE.\n"
    )

print(
    "\nReport saved to reports/cv_results.md"
)