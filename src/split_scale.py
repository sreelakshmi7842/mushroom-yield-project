from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load cleaned data and sort chronologically

df = (
    pd.read_parquet("data/interim/02_cleaned.parquet")
    .sort_values("timestamp")
    .reset_index(drop=True)
)

feature_cols = [
    "temperature_c",
    "humidity_pct",
    "co2_ppm",
]

target_col = "yield_kg"

# Create processed directory

PROCESSED = Path("data/processed")
PROCESSED.mkdir(parents=True, exist_ok=True)

MODELS = Path("models")
MODELS.mkdir(parents=True, exist_ok=True)

# Chronological 80/20 split


split_idx = int(len(df) * 0.8)

train = df.iloc[:split_idx].copy()
test = df.iloc[split_idx:].copy()

# Verify no future leakage

train_end_date = train["timestamp"].max()
test_start_date = test["timestamp"].min()

assert (
    test_start_date > train_end_date
), "Data leakage detected: test rows overlap train dates."

# Features and target

X_train_raw = train[feature_cols]
X_test_raw = test[feature_cols]

y_train = train[target_col].values
y_test = test[target_col].values

# Fit scaler on TRAIN ONLY

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

# Save scaler

joblib.dump(
    scaler,
    MODELS / "minmax_scaler_train.joblib"
)

# Save train/test datasets

train.to_csv(
    PROCESSED / "train.csv",
    index=False
)

test.to_csv(
    PROCESSED / "test.csv",
    index=False
)

# Save arrays for modeling

np.save(PROCESSED / "X_train.npy", X_train)
np.save(PROCESSED / "X_test.npy", X_test)

np.save(PROCESSED / "y_train.npy", y_train)
np.save(PROCESSED / "y_test.npy", y_test)

# Log split information

print("\n===== DATA SPLIT SUMMARY =====")

print(f"Total rows : {len(df)}")
print(f"Train rows : {len(train)}")
print(f"Test rows  : {len(test)}")

print(
    f"\nTrain Period : "
    f"{train['timestamp'].min()} "
    f"→ "
    f"{train['timestamp'].max()}"
)

print(
    f"Test Period  : "
    f"{test['timestamp'].min()} "
    f"→ "
    f"{test['timestamp'].max()}"
)

print(
    f"\nCutoff Date : {train_end_date}"
)

print("\nLeakage Check Passed ✓")

print("\nArray Shapes")
print(f"X_train : {X_train.shape}")
print(f"X_test  : {X_test.shape}")
print(f"y_train : {y_train.shape}")
print(f"y_test  : {y_test.shape}")

print("\nArtifacts saved successfully.")
print("\n===== ARTIFACT LOCATIONS =====")

print(f"Scaler   : {MODELS / 'minmax_scaler_train.joblib'}")

print(f"Train CSV: {PROCESSED / 'train.csv'}")
print(f"Test CSV : {PROCESSED / 'test.csv'}")

print(f"X_train  : {PROCESSED / 'X_train.npy'}")
print(f"X_test   : {PROCESSED / 'X_test.npy'}")

print(f"y_train  : {PROCESSED / 'y_train.npy'}")
print(f"y_test   : {PROCESSED / 'y_test.npy'}")