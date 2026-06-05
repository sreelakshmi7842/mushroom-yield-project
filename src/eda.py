import pandas as pd
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_parquet("data/interim/02_cleaned.parquet")
features = ["temperature_c", "humidity_pct", "co2_ppm", "yield_kg"]

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(df[features].corr(), cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(features)), features, rotation=45, ha="right")
ax.set_yticks(range(len(features)), features)
fig.colorbar(im, ax=ax, label="Pearson r")
ax.set_title("Sensor & Yield Correlations")
plt.tight_layout()
plt.savefig("reports/figures/corr_heatmap.png", dpi=150)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].scatter(df["humidity_pct"], df["yield_kg"], alpha=0.5, s=15)
axes[0].set(xlabel="Humidity (%)", ylabel="Yield (kg)")
axes[1].scatter(df["temperature_c"], df["yield_kg"], alpha=0.5, s=15)
axes[1].set(xlabel="Temperature (°C)", ylabel="Yield (kg)")
axes[2].scatter(df["co2_ppm"], df["yield_kg"], alpha=0.5, s=15)
axes[2].set(xlabel="CO₂ (ppm)", ylabel="Yield (kg)")
plt.tight_layout()
plt.savefig("reports/figures/scatter_yield.png", dpi=150)
# Load cleaned data
df = pd.read_parquet("data/interim/02_cleaned.parquet")

features = [
    "temperature_c",
    "humidity_pct",
    "co2_ppm",
    "yield_kg"
]

# Create correlation matrix
corr_matrix = df[features].corr()

# Create reports folder if needed
Path("reports").mkdir(exist_ok=True)

# Save CSV
corr_matrix.to_csv(
    "reports/correlation_matrix.csv"
)

# Save Markdown
with open(
    "reports/correlation_matrix.md",
    "w",
    encoding="utf-8"
) as f:
    f.write("# Correlation Matrix\n\n")
    f.write(corr_matrix.to_markdown())

print(corr_matrix)
print("Correlation matrix saved.")