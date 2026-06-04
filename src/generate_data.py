import numpy as np
import pandas as pd
from pathlib import Path

# Create folders
Path("data/raw").mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)

n = 365

temp = rng.normal(22, 1.5, n)
hum = np.clip(rng.normal(87, 3, n), 75, 98)
co2 = rng.normal(900, 80, n)

yield_kg = (
    8
    + 0.3 * temp
    + 0.05 * hum
    - 0.002 * co2
    + rng.normal(0, 0.5, n)
)

df = pd.DataFrame({
    "timestamp": pd.date_range(
        "2024-01-01",
        periods=n,
        freq="D"
    ),
    "temperature_c": temp.round(2),
    "humidity_pct": hum.round(1),
    "co2_ppm": co2.round(0),
    "yield_kg": yield_kg.round(2)
})

df.to_csv(
    "data/raw/polyhouse_sensors.csv",
    index=False
)

print("Dataset generated successfully!")
print(df.head())