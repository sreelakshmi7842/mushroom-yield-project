# src/eda.py
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 1. Ensure the output directory exists so the script doesn't crash
Path("reports/figures").mkdir(parents=True, exist_ok=True)

# Load data
df = pd.read_parquet("data/interim/02_cleaned.parquet")
features = ["temperature_c", "humidity_pct", "co2_ppm", "yield_kg"]

# 2. Correlation Heatmap
fig, ax = plt.subplots(figsize=(6, 5))
corr_matrix = df[features].corr()
im = ax.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)

# Add tick labels
ax.set_xticks(range(len(features)), features, rotation=45, ha="right")
ax.set_yticks(range(len(features)), features)

# Add correlation text inside the heatmap blocks for maximum readability
for i in range(len(features)):
    for j in range(len(features)):
        text_color = "black" if abs(corr_matrix.iloc[i, j]) < 0.5 else "white"
        ax.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}",
                ha="center", va="center", color=text_color)

fig.colorbar(im, ax=ax, label="Pearson r")
ax.set_title("Sensor & Yield Correlations")
plt.tight_layout()
plt.savefig("reports/figures/corr_heatmap.png", dpi=150)
print("✅ Saved heatmap to reports/figures/corr_heatmap.png")

# 3. Scatter Plots
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].scatter(df["humidity_pct"], df["yield_kg"], alpha=0.5, s=15)
axes[0].set(xlabel="Humidity (%)", ylabel="Yield (kg)", title="Humidity vs Yield")

axes[1].scatter(df["temperature_c"], df["yield_kg"], alpha=0.5, s=15)
axes[1].set(xlabel="Temperature (°C)", ylabel="Yield (kg)", title="Temperature vs Yield")

axes[2].scatter(df["co2_ppm"], df["yield_kg"], alpha=0.5, s=15)
axes[2].set(xlabel="CO₂ (ppm)", ylabel="Yield (kg)", title="CO₂ vs Yield")

plt.tight_layout()
plt.savefig("reports/figures/scatter_yield.png", dpi=150)
print("✅ Saved scatter plots to reports/figures/scatter_yield.png")