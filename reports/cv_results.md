# Cross Validation Results

## TimeSeriesSplit Configuration
- n_splits = 5
- No test data used during CV

## Model Performance

| Model             |   CV Mean MAE |   CV Std |   Train MAE |   Test MAE |
|:------------------|--------------:|---------:|------------:|-----------:|
| Linear Regression |         0.441 |    0.034 |       0.415 |      0.419 |
| Random Forest     |         0.475 |    0.057 |       0.167 |      0.45  |

## Interpretation

- Random Forest CV MAE: 0.475 ± 0.057
- Linear Regression CV MAE: 0.441 ± 0.034

Higher standard deviation across folds indicates greater variability and less stable performance.

Overfitting was assessed by comparing training MAE and test MAE.
