import streamlit as st

from src.predict import make_prediction

# ==================================================
# Page Config
# ==================================================

st.set_page_config(
    page_title="Mushroom Yield Forecast",
    layout="centered"
)

# ==================================================
# Cache Model Resources
# ==================================================

@st.cache_resource
def load_predictor():
    """
    Load prediction function once.
    Cached across Streamlit reruns.
    """
    return make_prediction

predictor = load_predictor()

# ==================================================
# App Header
# ==================================================

st.title("🍄 Polyhouse Yield Predictor")

st.caption(
    "Agritech environmental forecasting using "
    "temperature, humidity, and CO₂ sensor readings."
)

# ==================================================
# Sidebar Inputs
# ==================================================

with st.sidebar:

    st.header("Sensor Readings")

    temp = st.slider(
        "Temperature (°C)",
        min_value=10.0,
        max_value=35.0,
        value=22.0,
        step=0.1
    )

    humid = st.slider(
        "Humidity (%)",
        min_value=50.0,
        max_value=100.0,
        value=88.0,
        step=0.5
    )

    co2 = st.slider(
        "CO₂ (ppm)",
        min_value=400,
        max_value=2000,
        value=900,
        step=10
    )

# ==================================================
# Training Range Warnings
# ==================================================

if not (15 <= temp <= 30):
    st.warning(
        "Temperature is outside the typical "
        "training data range."
    )

if not (60 <= humid <= 95):
    st.warning(
        "Humidity is outside the typical "
        "training data range."
    )

if not (500 <= co2 <= 1500):
    st.warning(
        "CO₂ is outside the typical "
        "training data range."
    )

# ==================================================
# Prediction
# ==================================================

if st.button("Predict Yield"):

    prediction = predictor(
        temperature=temp,
        humidity=humid,
        co2=co2
    )

    st.metric(
        label="Estimated Yield",
        value=f"{prediction:.2f} kg"
    )

    st.info(
        "This forecast is advisory only and "
        "should not replace grower judgment."
    )

