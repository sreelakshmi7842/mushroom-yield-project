\# Linear Regression Diagnostics



\## Residual Definition



Residual = Actual Yield − Predicted Yield



Positive residuals indicate the model under-predicted yield.



Negative residuals indicate the model over-predicted yield.



\## Figures



\* reports/figures/residuals\_linear.png

\* reports/figures/residuals\_vs\_humidity\_linear.png



\## Findings



\### Finding 1



Residuals are generally centered around zero, indicating limited systematic bias.



\### Finding 2



Some residual spread increases at higher predicted yields, suggesting mild heteroscedasticity.



\### Finding 3



Residuals versus humidity show evidence of possible nonlinear behavior, indicating that humidity effects may not be perfectly captured by a linear model.



\## Outliers



Several observations have larger residuals than the majority of samples. These may correspond to unusual environmental conditions or data collection issues and should be reviewed before removal.



\## Recommendation



Linear Regression provides an interpretable baseline model.



However, residual diagnostics suggest that nonlinear relationships may exist.



The next recommended step is to train a Random Forest model and compare MAE, RMSE, and R² against this baseline.



\## Conclusion



The baseline model is useful for interpretation, but a nonlinear model may improve predictive performance.



