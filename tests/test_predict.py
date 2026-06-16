from src.predict import predict_yield
import pandas as pd

def test_predict_returns_float_in_range():
    result = predict_yield(22.0, 88.0, 920)

    assert isinstance(result, float)
    assert 0 < result < 50


def test_higher_humidity_changes_prediction():
    low = predict_yield(22.0, 75.0, 920)
    high = predict_yield(22.0, 92.0, 920)

    assert high != low
scenarios = [
    ("Optimal", 22, 88, 900),
    ("Dry Spell", 22, 65, 900),
    ("Heat Spike", 32, 88, 900),
    ("High CO2", 22, 88, 1800),
]

rows = []

for name, temp, hum, co2 in scenarios:
    pred = predict_yield(temp, hum, co2)

    rows.append({
        "Scenario": name,
        "Temperature (°C)": temp,
        "Humidity (%)": hum,
        "CO₂ (ppm)": co2,
        "Predicted Yield (kg)": round(pred, 2)
    })

df = pd.DataFrame(rows)

with open("reports/test_scenarios.md", "w", encoding="utf-8") as f:
    f.write("# Prediction Validation Scenarios\n\n")
    f.write(df.to_markdown(index=False))