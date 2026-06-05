# src/clean.py

import pandas as pd
from pathlib import Path

# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

INTERIM = BASE_DIR / "data" / "interim"
DOCS = BASE_DIR / "docs"

INTERIM.mkdir(parents=True, exist_ok=True)
DOCS.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Load data
# --------------------------------------------------
df = pd.read_parquet(INTERIM / "01_loaded.parquet")

# Save original snapshot
df.to_csv(
    INTERIM / "01_loaded_snapshot.csv",
    index=False
)

# --------------------------------------------------
# Missing value report BEFORE cleaning
# --------------------------------------------------
print("=== Missing Values Before Cleaning ===")
missing_report = df.isna().sum()
print(missing_report)

missing_report.to_csv(
    INTERIM / "missing_value_report.csv",
    header=["missing_count"]
)

# --------------------------------------------------
# Record initial row count
# --------------------------------------------------
initial_rows = len(df)

# --------------------------------------------------
# Explicit threshold rules
# --------------------------------------------------
valid = (
    df["temperature_c"].between(10, 35)
    & df["humidity_pct"].between(50, 100)
    & df["co2_ppm"].between(400, 2000)
    & df["yield_kg"].notna()
)

invalid_rows_removed = (~valid).sum()

df = df[valid].copy()

# --------------------------------------------------
# Fill short sensor gaps
# --------------------------------------------------
sensor_cols = [
    "temperature_c",
    "humidity_pct",
    "co2_ppm"
]

df[sensor_cols] = df[sensor_cols].ffill(limit=2)

# --------------------------------------------------
# Remove rows with null target
# --------------------------------------------------
df = df.dropna(subset=["yield_kg"])

# --------------------------------------------------
# Remove duplicate timestamps
# --------------------------------------------------
before_duplicates = len(df)

df = df.drop_duplicates(
    subset=["timestamp"],
    keep="last"
)

duplicates_removed = before_duplicates - len(df)

# --------------------------------------------------
# Final row count
# --------------------------------------------------
final_rows = len(df)

print("\n=== Cleaning Summary ===")
print(f"Initial rows      : {initial_rows}")
print(f"Invalid removed   : {invalid_rows_removed}")
print(f"Duplicates removed: {duplicates_removed}")
print(f"Final rows        : {final_rows}")

# --------------------------------------------------
# Save cleaned data
# --------------------------------------------------
df.to_parquet(
    INTERIM / "02_cleaned.parquet",
    index=False
)

df.to_csv(
    INTERIM / "02_cleaned.csv",
    index=False
)

# --------------------------------------------------
# Verification
# --------------------------------------------------
check_df = pd.read_parquet(
    INTERIM / "02_cleaned.parquet"
)

assert check_df["yield_kg"].isna().sum() == 0

print(
    "\nVerification Passed:"
    " 02_cleaned.parquet has zero nulls in yield_kg"
)

# --------------------------------------------------
# Save cleaning summary
# --------------------------------------------------
with open(
    INTERIM / "cleaning_summary.txt",
    "w",
    encoding="utf-8"
) as f:
    f.write("DATA CLEANING SUMMARY\n")
    f.write("=====================\n\n")
    f.write(f"Initial rows: {initial_rows}\n")
    f.write(f"Invalid rows removed: {invalid_rows_removed}\n")
    f.write(f"Duplicates removed: {duplicates_removed}\n")
    f.write(f"Final rows: {final_rows}\n")
    f.write(
        f"Null values in yield_kg: "
        f"{check_df['yield_kg'].isna().sum()}\n"
    )

print("\nFiles Generated:")
print(INTERIM / "missing_value_report.csv")
print(INTERIM / "01_loaded_snapshot.csv")
print(INTERIM / "02_cleaned.parquet")
print(INTERIM / "02_cleaned.csv")
print(INTERIM / "cleaning_summary.txt")