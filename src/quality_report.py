# src/quality_report.py
import pandas as pd
from pathlib import Path

# Load the clean data
df = pd.read_parquet("data/interim/02_cleaned.parquet")

# Compute summary statistics
cols = ["temperature_c", "humidity_pct", "co2_ppm", "yield_kg"]
summary = df[cols].describe().T

# Calculate Coefficient of Variation (CV)
summary["cv"] = summary["std"] / summary["mean"]

# Build the markdown report
report = []
report.append("# Polyhouse Data Quality Report\n")
report.append(f"**Total Valid Rows:** {len(df)}")
report.append(f"**Date Range:** {df['timestamp'].min()} -> {df['timestamp'].max()}\n")

report.append("## Summary Statistics")
report.append(summary.to_markdown())

# Add the mandatory insight section required by your checklist
report.append("\n## Key Insights")
report.append("* **Yield Distribution:** Compare the mean and median `yield_kg` above. A significant difference suggests bumper harvest days skew the data. Use the median for typical flush size reporting.")
report.append("* **Environmental Control:** Check the `cv` (Coefficient of Variation) for `humidity_pct`. A low CV confirms the polyhouse is successfully maintaining a narrow, stable humidity band critical for oyster mushrooms.")

# Create the reports directory and save the file
Path("reports").mkdir(exist_ok=True)
Path("reports/data_quality.md").write_text("\n".join(report), encoding="utf-8")

print("Data quality report successfully generated in reports/data_quality.md")