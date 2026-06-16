from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==========================================
# Create report directories
# ==========================================

Path("reports").mkdir(exist_ok=True)
Path("reports/figures").mkdir(parents=True, exist_ok=True)

# ==========================================
# Load test data
# ==========================================

X_test = np.load("data/processed/X_test.npy")
y_test = np.load("data/processed/y_test.npy")

# ==========================================
# Load models
# ==========================================

models = {
    "Linear Regression":
        joblib.load("models/linear_regression.joblib"),

    "Random Forest Default":
        joblib.load("models/random_forest.joblib"),

    "Random Forest Tuned":
        joblib.load("models/random_forest_tuned.joblib")
}

# ==========================================
# Evaluate models
# ==========================================

results = []

predictions = {}

for name, model in models.items():

    pred = model.predict(X_test)

    predictions[name] = pred

    mae = mean_absolute_error(y_test, pred)

    rmse = np.sqrt(
        mean_squared_error(y_test, pred)
    )

    r2 = r2_score(y_test, pred)

    if name == "Linear Regression":
        interpretability = "High"
    elif name == "Random Forest Default":
        interpretability = "Medium"
    else:
        interpretability = "Medium-Low"

    results.append({
        "Model": name,
        "CV MAE": "See training logs",
        "Test MAE": round(mae, 3),
        "RMSE": round(rmse, 3),
        "R²": round(r2, 3),
        "Training Time (s)": "N/A",
        "Interpretability": interpretability
    })

comparison = pd.DataFrame(results)

comparison = comparison.sort_values(
    by="Test MAE",
    ascending=True
)

# ==========================================
# Save CSV
# ==========================================

comparison.to_csv(
    "reports/model_comparison.csv",
    index=False
)

print("\nMODEL COMPARISON\n")
print(comparison.to_markdown(index=False))

# ==========================================
# Champion Selection
# ==========================================

best_model = comparison.iloc[0]["Model"]

best_mae = comparison.iloc[0]["Test MAE"]

linear_mae = comparison.loc[
    comparison["Model"] == "Linear Regression",
    "Test MAE"
].values[0]

# Tie rule
if abs(best_mae - linear_mae) < 0.05:
    best_model = "Linear Regression"

print(f"\nChampion Model: {best_model}")

# ==========================================
# Predicted vs Actual Plot
# ==========================================

champ_pred = predictions[best_model]

plt.figure(figsize=(8,6))

plt.scatter(
    y_test,
    champ_pred,
    alpha=0.7
)

low = min(y_test.min(), champ_pred.min())
high = max(y_test.max(), champ_pred.max())

plt.plot(
    [low, high],
    [low, high],
    "r--"
)

plt.xlabel("Actual Yield (kg)")
plt.ylabel("Predicted Yield (kg)")
plt.title(
    f"Predicted vs Actual Yield\nChampion: {best_model}"
)

plt.tight_layout()

plt.savefig(
    "reports/figures/pred_vs_actual.png",
    dpi=150
)

plt.close()

# ==========================================
# Markdown Report
# ==========================================

with open(
    "reports/model_comparison.md",
    "w",
    encoding="utf-8"
) as f:

    f.write("# Model Comparison and Champion Selection\n\n")

    f.write("## Comparison Table\n\n")

    f.write(
        comparison.to_markdown(index=False)
    )

    f.write("\n\n")

    f.write("## Champion Model\n\n")

    f.write(
        f"**Selected Model:** {best_model}\n\n"
    )

    f.write(
        "The champion model was selected based on "
        "lowest Test MAE while considering RMSE, "
        "R² score, and interpretability. "
        "If performance differences were minimal, "
        "Linear Regression was preferred due to "
        "greater transparency and stakeholder trust.\n\n"
    )

    f.write("## Predicted vs Actual Plot\n\n")

    f.write(
        "Saved at:\n\n"
        "`reports/figures/pred_vs_actual.png`\n\n"
    )

    f.write("## Limitations and Edge Cases\n\n")

    f.write(
        "- Predictions outside observed sensor ranges may be unreliable.\n"
        "- Seasonal effects may not be fully captured.\n"
        "- Environmental changes not represented in training data may reduce accuracy.\n"
        "- Synthetic datasets may not perfectly reflect real mushroom farms.\n"
        "- Extreme temperature, humidity, or CO₂ values may lead to prediction errors.\n"
        "- Model output is advisory only and should not replace grower judgment.\n"
    )

print("\nSaved:")
print("reports/model_comparison.csv")
print("reports/model_comparison.md")
print("reports/figures/pred_vs_actual.png")