import pandas as pd
from pathlib import Path

RAW = Path("data/raw/polyhouse_sensors.csv")
INTERIM = Path("data/interim")
INTERIM.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(
    RAW,
    parse_dates=["timestamp"],
    dtype={
        "temperature_c": "float64",
        "humidity_pct": "float64",
        "co2_ppm": "float64",
        "yield_kg": "float64",
    },
)

print(df.shape)
print(df.dtypes)
print(df.head())

df.to_parquet(INTERIM / "01_loaded.parquet", index=False)