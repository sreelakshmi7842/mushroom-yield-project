import streamlit as st
import numpy as np
import pandas as pd
import json
from pathlib import Path

# ==================================================
# Page Config
# ==================================================

st.set_page_config(
    page_title="Mushroom Yield Forecast",
    page_icon="🍄",
    layout="centered"
)

# ==================================================
# Safe Import
# ==================================================

try:
    from src.predict import make_prediction

except FileNotFoundError as e:
    st.error(
        f"""
        Model artifacts could not be loaded.

        {e}

        Please verify:

        - random_forest_tuned.joblib
        - minmax_scaler_train.joblib
        - feature_cols.json

        exist inside the models folder.
        """
    )
    st.stop()

# ==================================================
# Cached Predictor
# ==================================================

@st.cache_resource
def load_predictor():
    return make_prediction

predictor = load_predictor()

# ==================================================
# Metadata
# ==================================================

metadata_file = Path("models/model_metadata.json")

if metadata_file.exists():

    with open(metadata_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)

else:

    metadata = {
        "version": "v0.1-model",
        "last_training_date": "12 Jun 2026",
        "test_mae": "Unknown",
        "champion_model": "Random Forest Tuned"
    }

# ==================================================
# Header
# ==================================================

st.title("🍄 Mushroom Yield Forecast")

st.markdown(
    """
Estimate expected mushroom yield using environmental
sensor readings from a controlled polyhouse environment.

### Input Units

- Temperature → °C
- Humidity → %
- CO₂ → ppm

### Output

Predicted mushroom yield in kilograms.
"""
)

# ==================================================
# Sidebar Inputs
# ==================================================

with st.sidebar:

    st.header("Sensor Readings")

    temperature = st.slider(
        "Temperature (°C)",
        10.0,
        35.0,
        22.0,
        0.1
    )

    humidity = st.slider(
        "Humidity (%)",
        50.0,
        100.0,
        88.0,
        0.5
    )

    co2 = st.slider(
        "CO₂ (ppm)",
        400,
        2000,
        900,
        10
    )

# ==================================================
# Range Validation
# ==================================================

warnings = []

if not (15 <= temperature <= 30):
    warnings.append(
        "Temperature is outside the training range."
    )

if not (60 <= humidity <= 95):
    warnings.append(
        "Humidity is outside the training range."
    )

if not (500 <= co2 <= 1500):
    warnings.append(
        "CO₂ is outside the training range."
    )

for warning in warnings:
    st.warning(warning)

# ==================================================
# Prediction
# ==================================================

if st.button(
    "Predict Yield",
    use_container_width=True
):

    with st.spinner(
        "Generating forecast..."
    ):

        prediction = predictor(
            temperature=temperature,
            humidity=humidity,
            co2=co2
        )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Estimated Yield",
            f"{prediction:,.2f} kg"
        )

    with col2:

        st.metric(
            "Humidity",
            f"{humidity:,.1f}%"
        )

    st.success(
        "Prediction completed successfully."
    )

    # ==========================================
    # Sensitivity Chart
    # ==========================================

    st.subheader(
        "What-if Analysis: Humidity Sensitivity"
    )

    st.markdown(
        """
Shows how predicted yield changes when
humidity varies while temperature and CO₂
remain fixed.
"""
    )

    humidity_range = np.linspace(
        60,
        98,
        39
    )

    predictions = [
        predictor(
            temperature=temperature,
            humidity=h,
            co2=co2
        )
        for h in humidity_range
    ]

    chart_df = pd.DataFrame({
        "Humidity (%)": humidity_range,
        "Predicted Yield (kg)": predictions
    })

    st.line_chart(
        chart_df,
        x="Humidity (%)",
        y="Predicted Yield (kg)"
    )

# ==================================================
# Model Information
# ==================================================

with st.expander(
    "Model Information"
):

    st.markdown(
        f"""
### Metadata

**Model Version:** {metadata["version"]}

**Champion Model:** {metadata["champion_model"]}

**Last Training Date:** {metadata["last_training_date"]}

**Test MAE:** {metadata["test_mae"]}

### Features

- Temperature (°C)
- Humidity (%)
- CO₂ (ppm)

### Notes

Predictions are generated using the
saved champion model and scaler.
"""
    )

# ==================================================
# Methodology
# ==================================================

with st.expander(
    "Methodology"
):

    st.markdown(
        """
1. Raw sensor readings are collected.
2. Inputs are scaled using the saved scaler.
3. The trained model predicts yield.
4. Results are returned in kilograms.

For additional details see:

`reports/model_comparison.md`
"""
    )

# ==================================================
# Footer
# ==================================================

st.markdown("---")

st.caption(
    "This tool provides advisory forecasts only. "
    "Predictions should support, not replace, "
    "grower expertise and field observations."
)

