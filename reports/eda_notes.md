\# EDA Notes



\## Objective



Perform exploratory data analysis on the cleaned polyhouse sensor dataset to understand data distribution, variability, and relationships between environmental factors and mushroom yield.



\## Dataset



\* Source: `02\_cleaned.parquet`

\* Features analyzed:



&#x20; \* temperature\_c

&#x20; \* humidity\_pct

&#x20; \* co2\_ppm

&#x20; \* yield\_kg



\## Summary Statistics



Summary statistics were generated using the `describe()` function for all numerical features.



Metrics reviewed:



\* Count

\* Mean

\* Standard Deviation

\* Minimum

\* 25th Percentile

\* Median (50th Percentile)

\* 75th Percentile

\* Maximum



\## Variability Analysis



The coefficient of variation (CV = standard deviation / mean) was calculated for each feature.



Purpose:



\* Compare variability across measurements with different units.

\* Identify features with relatively high or low dispersion.



Observations:



\* Features with lower CV values are more stable.

\* Features with higher CV values exhibit greater relative variation.



\## Distribution Analysis



Mean and median values were compared for each feature.



Interpretation:



\* Mean > Median → slight right-skewed distribution.

\* Mean < Median → slight left-skewed distribution.

\* Mean ≈ Median → approximately symmetric distribution.



The generated report documents the skew direction for:



\* temperature\_c

\* humidity\_pct

\* co2\_ppm

\* yield\_kg



\## Correlation Analysis



A Pearson correlation matrix was computed for:



\* temperature\_c

\* humidity\_pct

\* co2\_ppm

\* yield\_kg



Outputs:



\* `reports/correlation\_matrix.csv`

\* `reports/correlation\_matrix.md`

\* `reports/figures/corr\_heatmap.png`



Purpose:



\* Measure linear relationships between sensor variables and yield.

\* Identify potentially useful predictors for future modeling.



\## Scatter Plot Analysis



Scatter plots were generated for:



1\. Humidity (%) vs Yield (kg)

2\. Temperature (°C) vs Yield (kg)

3\. CO₂ (ppm) vs Yield (kg)



Output:



\* `reports/figures/scatter\_yield.png`



Purpose:



\* Visualize relationships between environmental conditions and mushroom yield.

\* Identify trends, clusters, and potential outliers.



\## Conclusion



The EDA process successfully generated descriptive statistics, variability measures, distribution insights, correlation analysis, and visualizations. These outputs provide a foundation for feature selection, predictive modeling, and further analysis of mushroom yield in controlled polyhouse environments.



\### Correlation Findings



The Pearson correlation matrix was computed for temperature, humidity, CO₂ concentration, and mushroom yield.



\#### Strongest Positive Correlation



\- \*\*temperature\_c ↔ yield\_kg:\*\* \*\*r = 0.524\*\*

\- This indicates a moderate positive relationship between temperature and mushroom yield.

\- Within the observed operating range, higher temperatures were generally associated with higher yield.



\#### Second Strongest Positive Correlation



\- \*\*humidity\_pct ↔ yield\_kg:\*\* \*\*r = 0.242\*\*

\- This indicates a weak positive relationship between humidity and yield.

\- Higher humidity levels were associated with slightly higher yield.



\#### Strongest Negative Correlation



\- \*\*co2\_ppm ↔ yield\_kg:\*\* \*\*r = -0.260\*\*

\- This indicates a weak negative relationship between CO₂ concentration and yield.

\- Higher CO₂ values were associated with slightly lower yield in this dataset.



\#### Relationships Among Sensor Variables



\- temperature\_c ↔ humidity\_pct: r = -0.022

\- temperature\_c ↔ co2\_ppm: r = 0.007

\- humidity\_pct ↔ co2\_ppm: r = -0.018



These values are close to zero, indicating little to no linear relationship among the sensor measurements themselves.



\### Caveats



\- Correlation measures association, not causation.

\- The observed relationships do not prove that changes in a sensor variable directly cause changes in mushroom yield.

\- Results are based on the available dataset and observed operating conditions.

\- Additional experiments and predictive modeling would be required to establish causal effects.

