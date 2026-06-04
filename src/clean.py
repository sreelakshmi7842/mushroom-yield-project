# src/clean.py
import pandas as pd
from pathlib import Path

df = pd.read_csv("data/raw/polyhouse_sensors.csv")

# Missing report
print(df.isna().sum())

# Valid ranges for oyster polyhouse
valid = (
    df["humidity_pct"].between(50, 100)
    & df["temperature_c"].between(10, 35)
    & df["co2_ppm"].between(400, 2000)
    & df["yield_kg"].notna()
)
df = df[valid].copy()

# Short gap: forward-fill sensor columns only
cols = ["temperature_c", "humidity_pct", "co2_ppm"]
df[cols] = df[cols].ffill(limit=2)

# Drop remaining rows with null target
df = df.dropna(subset=["yield_kg"])

# Duplicates by timestamp
df = df.drop_duplicates(subset=["timestamp"], keep="last")

df.to_parquet("data/processed/02_cleaned.parquet", index=False)
print(f"Clean rows: {len(df)}")