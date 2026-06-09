import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler

# Load cleaned dataset
df = pd.read_parquet("data/interim/02_cleaned.parquet")

print("Dataset Loaded")
print(df.head())

# Feature Engineering
df["temp_humidity_interaction"] = (
    df["temperature_c"] * df["humidity_pct"]
)

# Feature columns
feature_cols = [
    "temperature_c",
    "humidity_pct",
    "co2_ppm",
    "temp_humidity_interaction"
]

# X and y
X = df[feature_cols]
y = df["yield_kg"]

print("X Shape:", X.shape)
print("Y Shape:", y.shape)

# Scaling
scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(X)

X_scaled = pd.DataFrame(
    X_scaled,
    columns=feature_cols
)

print("\nMinimum Values")
print(X_scaled.min())

print("\nMaximum Values")
print(X_scaled.max())

# Save scaler
joblib.dump(
    scaler,
    "models/minmax_scaler.pkl"
)

# Save processed data
processed_df = X_scaled.copy()
processed_df["yield_kg"] = y

processed_df.to_parquet(
    "data/processed/features.parquet",
    index=False
)

print("\nFiles Saved Successfully")
print("Scaler: models/minmax_scaler.pkl")
print("Dataset: data/processed/features.parquet")