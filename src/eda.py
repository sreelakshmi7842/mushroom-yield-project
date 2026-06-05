import pandas as pd
from pathlib import Path

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