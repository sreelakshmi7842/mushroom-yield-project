import streamlit as st
import numpy as np
import pandas as pd

from src.predict import make_prediction

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Mushroom Yield Forecast",
    page_icon="🍄",
    layout="centered"
)

# ==================================================
# CACHED PREDICTOR
# ==================================================

@st.cache_resource
def load_predictor():
    return make_prediction

predictor = load_predictor()

# ==================================================
# MODEL METADATA
# ==================================================

MODEL_VERSION = "v0.1-model"

LAST_TRAINING_DATE = "12 Jun 2026"

TEST_MAE = "Replace with actual MAE"

# ==================================================
# HEADER
# ==================================================

st.title("🍄 Mushroom Yield Forecast")

st.markdown(
    """
    Estimate expected mushroom yield using environmental
    sensor readings collected inside the polyhouse.

    Inputs:
    - Temperature (°C)
    - Relative Humidity (%)
    - CO₂ Concentration (ppm)

    Output:
    - Estimated Yield (kg)

    This tool is intended for planning and operational support.
    """
)

# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.header("Sensor Inputs")

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
# VALIDATION WARNINGS
# ==================================================

if not (15 <= temperature <= 30):
    st.warning(
        "Temperature is outside the typical training range."
    )

if not (60 <= humidity <= 95):
    st.warning(
        "Humidity is outside the typical training range."
    )

if not (500 <= co2 <= 1500):
    st.warning(
        "CO₂ is outside the typical training range."
    )

# ==================================================
# PREDICTION SECTION
# ==================================================

if st.button("Predict Yield", use_container_width=True):

    prediction = predictor(
        temperature=temperature,
        humidity=humidity,
        co2=co2
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Estimated Yield",
            f"{prediction:.2f} kg"
        )

    with col2:

        st.metric(
            "Humidity",
            f"{humidity:.1f}%"
        )

    st.success(
        "Prediction generated successfully."
    )

    # ==========================================
    # SENSITIVITY ANALYSIS
    # ==========================================

    st.subheader("What-if Analysis")

    st.markdown(
        """
        This chart shows how predicted yield changes
        as humidity varies while temperature and CO₂
        remain fixed at the selected values.
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
# MODEL INFORMATION
# ==================================================

with st.expander("Model Information"):

    st.markdown(
        f"""
### Model Metadata

- Version: **{MODEL_VERSION}**
- Last Training Date: **{LAST_TRAINING_DATE}**
- Test MAE: **{TEST_MAE}**
- Features:
  - Temperature (°C)
  - Humidity (%)
  - CO₂ (ppm)

### Notes

The model was trained using historical
polyhouse sensor observations and yield data.

Predictions should be interpreted as guidance
rather than guaranteed outcomes.
"""
    )

# ==================================================
# METHODOLOGY
# ==================================================

with st.expander("Methodology"):

    st.markdown(
        """
1. Sensor readings are scaled using the
   saved MinMaxScaler.

2. The champion machine-learning model
   predicts expected mushroom yield.

3. Yield estimates are returned in kilograms.

For technical details, see:

`reports/model_comparison.md`
"""
    )

# ==================================================
# FOOTER
# ==================================================

st.markdown("---")

st.caption(
    "Forecasts are advisory only and should not "
    "replace grower judgment or field observations."
)

