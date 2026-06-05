# Polyhouse Data Quality Report

**Total Valid Rows:** 365
**Date Range:** 2024-01-01 -> 2024-12-30

## Summary Statistics
|               |   count |     mean |       std |    min |    25% |    50% |    75% |     max |        cv |
|:--------------|--------:|---------:|----------:|-------:|-------:|-------:|-------:|--------:|----------:|
| temperature_c |     365 |  21.9867 |  1.41241  |  18.15 |  21.01 |  21.97 |  22.88 |   26.37 | 0.0642392 |
| humidity_pct  |     365 |  86.7433 |  3.06779  |  78.1  |  84.6  |  86.7  |  88.7  |   94.8  | 0.0353664 |
| co2_ppm       |     365 | 901.162  | 78.2652   | 608    | 854    | 904    | 949    | 1154    | 0.0868493 |
| yield_kg      |     365 |  17.1394 |  0.679041 |  15.31 |  16.7  |  17.13 |  17.63 |   18.85 | 0.0396187 |

## Key Insights
* **Yield Distribution:** Compare the mean and median `yield_kg` above. A significant difference suggests bumper harvest days skew the data. Use the median for typical flush size reporting.
* **Environmental Control:** Check the `cv` (Coefficient of Variation) for `humidity_pct`. A low CV confirms the polyhouse is successfully maintaining a narrow, stable humidity band critical for oyster mushrooms.