# Model Comparison and Champion Selection

## Comparison Table

| Model                 | CV MAE            |   Test MAE |   RMSE |    R² | Training Time (s)   | Interpretability   |
|:----------------------|:------------------|-----------:|-------:|------:|:--------------------|:-------------------|
| Linear Regression     | See training logs |      0.419 |  0.535 | 0.427 | N/A                 | High               |
| Random Forest Tuned   | See training logs |      0.445 |  0.562 | 0.369 | N/A                 | Medium-Low         |
| Random Forest Default | See training logs |      0.45  |  0.58  | 0.327 | N/A                 | Medium             |

## Champion Model

**Selected Model:** Linear Regression

The champion model was selected based on lowest Test MAE while considering RMSE, R² score, and interpretability. If performance differences were minimal, Linear Regression was preferred due to greater transparency and stakeholder trust.

## Predicted vs Actual Plot

Saved at:

`reports/figures/pred_vs_actual.png`

## Limitations and Edge Cases

- Predictions outside observed sensor ranges may be unreliable.
- Seasonal effects may not be fully captured.
- Environmental changes not represented in training data may reduce accuracy.
- Synthetic datasets may not perfectly reflect real mushroom farms.
- Extreme temperature, humidity, or CO₂ values may lead to prediction errors.
- Model output is advisory only and should not replace grower judgment.
