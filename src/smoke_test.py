import pandas as pd

df = pd.read_csv(
    "data/raw/polyhouse_sensors.csv"
)

print("Rows:", len(df))
print("Columns:", len(df.columns))

assert len(df) > 0

print("Smoke Test Passed!")