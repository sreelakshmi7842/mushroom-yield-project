import pandas as pd

df = pd.read_csv("data/raw/polyhouse_sensors.csv")

print(df.columns.tolist())
print()
print(df.head())