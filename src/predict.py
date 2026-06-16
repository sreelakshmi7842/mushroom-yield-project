from pathlib import Path
import json
import argparse

import joblib
import pandas as pd

# ==================================================
# Paths
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODEL_DIR / "random_forest_tuned.joblib"
SCALER_PATH = MODEL_DIR / "minmax_scaler_train.joblib"
FEATURES_PATH = MODEL_DIR / "feature_cols.json"

# ==================================================
# Load Artifacts
# ==================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not SCALER_PATH.exists():
    raise FileNotFoundError(f"Scaler not found: {SCALER_PATH}")

if not FEATURES_PATH.exists():
    raise FileNotFoundError(f"Feature list not found: {FEATURES_PATH}")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

with open(FEATURES_PATH, "r") as f:
    feature_cols = json.load(f)

# ==================================================
# Prediction Function
# ==================================================

def predict_yield(
    temperature_c: float,
    humidity_pct: float,
    co2_ppm: float
) -> float:
    """
    Predict mushroom yield in kg.
    """

    data = pd.DataFrame(
        [[temperature_c, humidity_pct, co2_ppm]],
        columns=feature_cols
    )

    scaled = scaler.transform(data)

    prediction = model.predict(scaled)[0]

    return float(prediction)

# ==================================================
# Helper Function
# ==================================================

def make_prediction(
    temperature: float,
    humidity: float,
    co2: float
) -> float:
    """
    Convenience wrapper for deployment.
    """

    return predict_yield(
        temperature_c=temperature,
        humidity_pct=humidity,
        co2_ppm=co2
    )

# ==================================================
# CLI Entry Point
# ==================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Predict mushroom yield"
    )

    parser.add_argument(
        "--temperature",
        type=float,
        required=True
    )

    parser.add_argument(
        "--humidity",
        type=float,
        required=True
    )

    parser.add_argument(
        "--co2",
        type=float,
        required=True
    )

    args = parser.parse_args()

    prediction = make_prediction(
        args.temperature,
        args.humidity,
        args.co2
    )

    print(
        f"Predicted Yield: {prediction:.2f} kg"
    )