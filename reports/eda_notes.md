\# Exploratory Data Analysis (EDA) Takeaways



\## Key Biological Insights \& Nonlinearity

\* \*\*Optimal Humidity Band:\*\* The scatter plot for humidity vs. yield reveals a non-linear relationship. Yields appear to peak within a specific optimal humidity band and drop off if the air is too dry or too saturated, which perfectly aligns with the strict moisture requirements of fruiting oyster mushrooms.

\* \*\*Temperature Clustering:\*\* Temperature exhibits a clustering effect rather than a strict linear trend. Extreme heat or cold stifles fruiting body development. This non-linearity suggests that tree-based models (like Random Forests) might perform better than simple linear regression moving forward.

\* \*\*CO₂ Accumulation:\*\* High CO₂ levels generally cluster with lower yields. As mushrooms respire heavily during growth, CO₂ builds up; without proper HVAC ventilation to remove it, cap growth is severely stunted.



\## Correlation Analysis \& Caveats

\* \*\*Strongest Correlations:\*\* Humidity typically shows the strongest positive correlation with yield (up to its optimal threshold), while CO₂ exhibits a negative correlation.

\* \*\*Caveat (Correlation ≠ Causation):\*\* The apparent statistical relationship between temperature and CO₂ may merely be a side-effect of the polyhouse's HVAC system turning on and off to regulate seasonal weather, rather than a direct biological mechanism. We must be cautious of this multicollinearity when selecting features for our final model.

