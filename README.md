Markdown
# Mushroom Yield Prediction Project

## Overview & Problem Statement
Predicting daily mushroom yield (kg) in a climate-controlled polyhouse using real-world sensor readings for temperature (°C), relative humidity (%), and CO₂ (ppm). This repository serves as a version-controlled data pipeline resilient against model breakdown due to sudden hardware or data drift updates.

This project is structured to support data ingestion, validation, preprocessing, model training, and deployment-ready model storage.

---

## Dataset Fields

| Column        | Description                                  |
| ------------- | -------------------------------------------- |
| timestamp     | Sensor reading timestamp                     |
| temperature_c | Temperature in Celsius                       |
| humidity_pct  | Relative humidity (%)                        |
| co2_ppm       | Carbon dioxide concentration (ppm)           |
| yield_kg      | Mushroom yield (kg)                          |

---

## Workflow

1. Store raw sensor files inside `data/raw/`
2. Clean and transform data into `data/interim/`
3. Perform analysis using notebooks
4. Train machine learning models
5. Save trained models in `models/`
6. Evaluate and deploy prediction pipeline

---

## cleaning log ## -day4

loaded clean.py file into src

This script performs data cleaning and preprocessing using several common techniques:
1 Missing value analysis
2 Range-based filtering (data validation)
3 Null target removal
4 Forward-fill imputation
5 Deletion of missing targets
6 Deduplication

Overall Cleaning Strategy

This is a combination of:

1 Data Validation Cleaning
  Filters out out-of-range sensor readings.
2 Missing Value Treatment
  Forward-fill imputation for sensor data.
  Row deletion for missing target values.
3 Data Deduplication
  Removes duplicate timestamps.
4 Quality-Based Filtering
  Retains only records that satisfy predefined oyster polyhouse environmental conditions.

Duplicate records were identified using the timestamp column and removed while retaining the latest occurrence. A total of 0 duplicate records were removed, resulting in a final cleaned dataset containing 365 rows.

02_cleaned.parquet was successfully loaded and validated. The target column (yield_kg) contains 0 missing values, confirming that all records are suitable for downstream analysis and model training.

## Data Quality Report Generation ## -day5

### Objective

The objective of this script is to perform an exploratory assessment of the cleaned polyhouse sensor dataset and automatically generate a data quality report.

### Tasks Performed

1. Loads the cleaned dataset (`02_cleaned.parquet`).
2. Calculates summary statistics for:

   * Temperature
   * Humidity
   * CO₂ concentration
   * Mushroom yield
3. Computes the coefficient of variation (CV) to measure relative variability.
4. Compares mean and median values to identify potential data skewness.
5. Generates human-readable insights describing the distribution of each feature.
6. Creates a Markdown report containing:

   * Dataset size
   * Date range
   * Summary statistics table
   * Distribution insights
7. Saves the report as:

```text
reports/data_quality.md
```

### Output

The generated report provides a concise overview of data quality and feature distributions, helping validate the dataset before feature engineering, visualization, and machine learning model development.


## Day 6: Exploratory Data Analysis (EDA) & Feature Validation

**Objective:** Transform clean statistics into visual hypotheses to identify non-linear relationships and feature correlations before building the machine learning model.

### What We Accomplished
We successfully completed Task 3 by writing `src/eda.py` to generate publication-ready visualizations and documenting our findings in `reports/eda_notes.md`. 

#### 1. Key Biological Insights (Scatter Plots)
* **Optimal Humidity Band:** The scatter plots revealed a clear curve (non-linear relationship). Yields peak tightly around a specific humidity "Goldilocks zone." Too dry or too saturated drastically reduces harvest weight.
* **Temperature Clustering:** Growth drops to zero outside the 10-35°C bounds, confirming that tree-based models (which handle thresholds well) will likely outperform strict linear regression.

#### 2. Correlation Analysis (Heatmap)
* **Feature Importance:** Humidity exhibited the strongest positive correlation with yield, while CO₂ showed a strong negative correlation (high CO₂ suffocates cap growth).
* **Important Caveat (Correlation ≠ Causation):** The heatmap showed a relationship between Temperature and CO₂. We noted that this is likely due to the polyhouse HVAC system turning on/off seasonally, rather than a direct biological link (a key multicollinearity insight for future modeling).

### Files Generated & Tracked
```text
├── reports/
│   ├── eda_notes.md                 # Written biological takeaways and caveats
│   └── figures/
│       ├── corr_heatmap.png         # Pearson matrix with readable feature labels
│       └── scatter_yield.png        # Labeled subplots for Temp, Humidity, and CO2 vs. Yield
├── src/
│   └── eda.py                       # The script used to generate the figures


✅ Checklist Completion Status
[x] Heatmap saved with readable feature labels.

[x] At least three scatter plots with axis labels and units.

[x] Written EDA takeaways committed to reports/eda_notes.md.

[x] Figures folder tracked in Git (PNG size reasonable at 150 DPI).

[x] Identified strongest positive/negative correlations with caveats.

# # Feature Engineering & Scaling## -day7

## Objective

Prepare machine learning features from the cleaned polyhouse dataset and scale them to a common range using Min-Max Scaling. The resulting feature set will be used for model training in later tasks.

## Input Dataset

Source:

data/interim/02_cleaned.parquet


Target Variable:

yield_kg

## Feature Definitions

### 1. Temperature

Column:

temperature_c

Description:

Average temperature inside the polyhouse in degrees Celsius.

Biological Importance:

Temperature directly affects mushroom growth rate, metabolism, and fruiting body development.

### 2. Humidity

Column:

humidity_pct

Description:

Relative humidity percentage inside the polyhouse.

Biological Importance:

Oyster mushrooms require high humidity for healthy growth and yield production. Low humidity can reduce productivity and affect mushroom quality.

### 3. Carbon Dioxide

Column:

co2_ppm

Description:

Carbon dioxide concentration measured in parts per million (ppm).

Biological Importance:

CO₂ levels influence mushroom respiration and growth conditions. Extremely high or low concentrations may affect yield.

### 4. Temperature–Humidity Interaction Feature

Column:

temp_humid_interaction

Formula:

temp_humid_interaction =
(temperature_c × humidity_pct) / 100

Example:

temperature_c = 25
humidity_pct = 80

temp_humid_interaction =
(25 × 80) / 100
= 20

Biological Importance:

Mushroom growth depends on the combined effect of temperature and humidity rather than either variable independently. This engineered feature helps the model capture interactions between these environmental factors.


## Feature Matrix and Target

Feature Matrix (X):

[
    "temperature_c",
    "humidity_pct",
    "co2_ppm",
    "temp_humid_interaction"
]


Target Variable (y):

yield_kg

## Scaling Method

Scaler Used:

MinMaxScaler()

Scaling Formula:

x_scaled =
(x - x_min) / (x_max - x_min)

Output Range:

[0, 1]

Purpose:

* Prevents large-scale variables from dominating smaller-scale variables.
* Improves compatibility with many machine learning algorithms.
* Produces comparable feature ranges.

## Saved Outputs

Processed Features:

data/processed/features.parquet


Saved Scaler:

models/minmax_scaler.joblib

## Validation Checks

The following checks are performed after feature engineering:

* Feature and target row counts match.
* No missing values remain after processing.
* All scaled features lie within the range [0, 1].
* Scaler object is successfully saved for future inference.

## Future Improvement

For learning purposes, the scaler is currently fitted on the full cleaned dataset.

##### DAY 8
## Chronological Train/Test Split

### Objective

To prepare the mushroom yield dataset for machine learning by creating a chronological train/test split while preventing data leakage.

### Methodology

1. Loaded the cleaned dataset from:

   `data/interim/02_cleaned.parquet`

2. Sorted records by timestamp.

3. Applied an 80/20 chronological split:

   * First 80% of records → Training set
   * Last 20% of records → Test set

4. Verified that no test record occurred before the training cutoff date.

5. Applied MinMaxScaler:

   * Fitted only on training data
   * Applied to both training and test data

### Features Used

* temperature_c
* humidity_pct
* co2_ppm

### Target Variable

* yield_kg

### Leakage Prevention

The following assertion verifies that all test observations occur after the training period:

`assert test_start_date > train_end_date`

### Saved Artifacts

#### Model Assets

* models/minmax_scaler_train.joblib

#### Processed Data

* data/processed/train.csv
* data/processed/test.csv

#### NumPy Arrays

* data/processed/X_train.npy
* data/processed/X_test.npy
* data/processed/y_train.npy
* data/processed/y_test.npy

### Output Information Logged

The script logs:

* Train and test row counts
* Train period dates
* Test period dates
* Split cutoff date
* Leakage validation status
* X and y array shapes

### Execution

Run the script using:

`python src/split_scale.py`

### Timeline Diagram

The chronological split can be visualized as:

|------------------- Training Set (80%) -------------------|------ Test Set (20%) ------|

2024-01-01                                            2024-10-18              2024-10-19                    2024-12-30
                                                        ↑
                                                   Split Cutoff

This timeline illustrates the separation between training and test windows and confirms that future observations are not used during model training.

### Seasonality Consideration

Because the dataset is split chronologically, the test period represents future observations that the model has not seen during training. If the average value of `yield_kg` in the test period differs significantly from the training period, evaluation metrics may decrease. Such differences can occur due to seasonality, environmental changes, or shifts in growing conditions over time.

This behavior is expected in real-world forecasting scenarios and does not indicate data leakage. Instead, it reflects the model's ability to generalize to future data under changing conditions.

## Baseline Linear Regression### -day 9

### Objective

Train a baseline Linear Regression model to predict mushroom yield using environmental sensor measurements and evaluate its performance on unseen test data.

### Features

| Feature | Description |
|----------|-------------|
| temperature_c | Temperature inside the polyhouse (°C) |
| humidity_pct | Relative humidity (%) |
| co2_ppm | Carbon dioxide concentration (ppm) |

### Target

| Target | Description |
|----------|-------------|
| yield_kg | Mushroom yield (kg) |

### Methodology

1. Loaded preprocessed train and test datasets.
2. Trained a Linear Regression model using `X_train` and `y_train`.
3. Generated predictions on the test set.
4. Computed evaluation metrics:
   - Mean Absolute Error (MAE)
   - Root Mean Squared Error (RMSE)
   - R² Score
5. Inspected model coefficients to understand feature influence.
6. Saved the trained model and evaluation reports.

### Coefficient Interpretation

Since all features were scaled using MinMaxScaler, coefficient magnitudes can be compared directly.

- Positive coefficient → Higher feature value tends to increase yield.
- Negative coefficient → Higher feature value tends to decrease yield.
- Larger absolute coefficient → Greater influence on model predictions.

### Evaluation Metrics

- **MAE** measures average prediction error in kilograms.
- **RMSE** penalizes larger prediction errors.
- **R²** measures how much variation in yield is explained by the model.

### Saved Artifacts

#### Model

- `models/linear_regression.joblib`

#### Reports

- `reports/metrics_linear.json`
- `reports/metrics_linear.md`

### Execution

```bash
python src/train_linear_model.py
```

### Baseline Assessment

R² interpretation:

| R² Score | Assessment |
|-----------|------------|
| > 0.70 | Strong baseline |
| 0.50 – 0.70 | Reasonable baseline |
| < 0.50 | Additional feature engineering or advanced models recommended |

### Output

The script prints:

- MAE
- RMSE
- R² Score
- Feature coefficients
- Saved artifact locations

The resulting model serves as a baseline benchmark for future machine learning experiments on mushroom yield prediction.

# Day 10 – Linear Regression Diagnostics

## Objective

The objective of Day 10 was to evaluate the quality of the Linear Regression baseline model beyond standard metrics.

While MAE, RMSE, and R² provide overall performance information, residual diagnostics help determine whether the model errors behave randomly or exhibit patterns that indicate missing relationships, feature engineering opportunities, or the need for a more complex model.

---

## Tasks Completed

### 1. Loaded Trained Linear Regression Model

Model file:

```text
models/linear_regression.joblib
```

The previously trained Linear Regression model was loaded and used to generate predictions on the test dataset.

---

### 2. Generated Predictions

Predictions were produced for:

* Training set
* Test set

These predictions were used to calculate residuals.

---

### 3. Calculated Residuals

Residuals were computed using:

```text
Residual = Actual Yield − Predicted Yield
```

Interpretation:

* Positive residual → model under-predicted yield
* Negative residual → model over-predicted yield
* Residual near zero → accurate prediction

---

### 4. Created Diagnostic Plots

#### Residuals vs Predicted Yield

Saved as:

```text
reports/figures/residuals_linear.png
```

Purpose:

* Detect heteroscedasticity
* Detect systematic bias
* Check whether residuals are centered around zero

---

#### Residuals vs Humidity

Saved as:

```text
reports/figures/residuals_vs_humidity_linear.png
```

Purpose:

* Investigate whether humidity has nonlinear effects on yield
* Detect feature-specific residual patterns

---

## Diagnostic Findings

### Finding 1: Residuals Centered Around Zero

Most residuals were distributed around the zero line.

Interpretation:

The model does not show strong systematic over-prediction or under-prediction.

---

### Finding 2: Variance Changes at Higher Predictions

The spread of residuals increases slightly for some larger predicted yields.

Interpretation:

This may indicate mild heteroscedasticity, meaning prediction uncertainty increases for larger harvests.

---

### Finding 3: Potential Nonlinear Relationship with Humidity

The residual-versus-humidity plot suggests that the effect of humidity may not be perfectly linear.

Interpretation:

A linear model may not fully capture the relationship between humidity and mushroom yield.

---

## Outlier Investigation

Several observations showed larger residual values than the majority of samples.

Possible explanations:

* Environmental anomalies
* Harvest recording inconsistencies
* Unusual growing conditions
* Sensor measurement noise

These observations should be reviewed before considering removal.

---

## Model Assessment

### Strengths

* Easy to interpret
* Fast training
* Clear coefficient explanations
* Useful benchmark model

### Limitations

* Assumes linear relationships
* Sensitive to nonlinear patterns
* May not capture interactions among environmental variables

---

## Recommendation

The Linear Regression model should be retained as the baseline model.

However, diagnostic analysis suggests that nonlinear relationships may exist within the dataset.

Recommended next step:

### Train a Random Forest Regressor

Reasons:

* Captures nonlinear relationships
* Handles feature interactions automatically
* Often improves predictive accuracy
* Robust to complex environmental effects

---

## Deliverables

Generated during Day 10:

```text
reports/
│
├── linear_diagnostics.md
│
└── figures/
    ├── residuals_linear.png
    └── residuals_vs_humidity_linear.png
```

Model used:

```text
models/linear_regression.joblib
```

---

## Conclusion

Day 10 focused on evaluating model behavior rather than model accuracy alone.

Residual diagnostics revealed that the Linear Regression baseline is useful and interpretable, but some evidence of nonlinearity remains.

# Day 11 – Random Forest Regression Model

## Objective

To train and evaluate a Random Forest Regression model for mushroom yield prediction and compare its performance against the Linear Regression baseline.

---

## Dataset

### Training Data

* Samples: **292**

### Test Data

* Samples: **73**

Input files used:

* `data/processed/X_train.npy`
* `data/processed/y_train.npy`
* `data/processed/X_test.npy`
* `data/processed/y_test.npy`

The Random Forest model was trained exclusively on the training dataset and evaluated on the held-out test dataset.

---

## Methodology

### Baseline Model

A Linear Regression model was trained using the training data and evaluated on the test set.

Metrics computed:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* Coefficient of Determination (R²)

### Random Forest Model

A Random Forest Regressor was trained with the following configuration:

```python
RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
```

Parameters:

| Parameter    | Value |
| ------------ | ----- |
| n_estimators | 100   |
| random_state | 42    |
| n_jobs       | -1    |

The model was fitted using:

```python
rf.fit(X_train, y_train)
```

Predictions were generated on the held-out test set and evaluated using the same metrics as the baseline model.

---

## Model Comparison

Performance of the Random Forest model was compared directly against the Linear Regression baseline.

Metrics included:

* MAE
* RMSE
* R²

Comparison results were exported for future reference.

Saved file:

```text
reports/model_comparison.csv
```

---

## Feature Importance Analysis

Random Forest feature importances were extracted using:

```python
rf.feature_importances_
```

The following environmental variables were evaluated:

* Temperature (`temperature_c`)
* Humidity (`humidity_pct`)
* CO₂ (`co2_ppm`)

Feature importance values indicate the relative contribution of each predictor to the model's predictions.

---

## Visualization

A horizontal bar chart was created to visualize feature importance rankings.

Saved figure:

```text
reports/figures/rf_feature_importance.png
```

The chart allows quick identification of the most influential environmental factor affecting mushroom yield predictions.

---

## Model Artifact

The trained Random Forest model was serialized using Joblib.

Saved model:

```text
models/random_forest.joblib
```

This artifact can be loaded later for inference, validation, or deployment.

---

## Interpretation

The feature with the highest importance score was identified as the strongest contributor to yield prediction.

Feature importance analysis provides insight into which environmental conditions have the greatest influence on the model's decisions.

---

## Complexity Assessment

Random Forest performance was compared against Linear Regression to determine whether the additional model complexity was justified.

Decision rule:

* Higher R² and lower prediction error → Random Forest likely justified.
* Similar performance → Linear Regression may remain preferable due to simplicity and interpretability.

This comparison helps balance predictive performance against model complexity.

---

## Saved Artifacts

### Model

```text
models/random_forest.joblib
```

### Comparison Table

```text
reports/model_comparison.csv
```

### Feature Importance Plot

```text
reports/figures/rf_feature_importance.png
```

---

## Deliverables Completed

* Random Forest trained using training data only.
* Test set evaluation completed.
* Performance compared against Linear Regression baseline.
* Feature importance values computed.
* Feature importance visualization generated.
* Trained model saved for reuse.
* Model comparison table exported.
* Complexity justification documented.

---

## Conclusion

A Random Forest Regression model was successfully trained and evaluated on the mushroom yield dataset. The model's predictive performance was compared with a Linear Regression baseline using MAE, RMSE, and R² metrics. Feature importance analysis provided insight into the influence of temperature, humidity, and CO₂ on yield prediction. The trained model, evaluation outputs, and visualization artifacts were saved to ensure reproducibility and future model analysis.



# Day 12 Time Series Cross-Validation

## Objective

To evaluate model stability and generalization performance using time-aware cross-validation while preventing data leakage from future observations.

## Methodology

1. Loaded the preprocessed training and test datasets:

   * `data/processed/X_train.npy`
   * `data/processed/X_test.npy`
   * `data/processed/y_train.npy`
   * `data/processed/y_test.npy`

2. Configured `TimeSeriesSplit` with 5 folds:

   * Training data was split chronologically.
   * Earlier observations were used to predict later observations.
   * No test data was used during cross-validation.

3. Evaluated two models:

   * Linear Regression
   * Random Forest Regressor (`n_estimators=100`)

4. Computed cross-validated Mean Absolute Error (MAE) for each fold.

5. Calculated:

   * Mean CV MAE
   * Standard deviation of CV MAE
   * Training MAE
   * Hold-out Test MAE

6. Compared cross-validation performance with final test-set performance.

7. Assessed overfitting by comparing training MAE and test MAE.

## Results

Cross-validation results were summarized for both models using:

* Mean CV MAE
* CV MAE Standard Deviation
* Training MAE
* Test MAE

Lower MAE values indicate better predictive performance.

The standard deviation across folds was used to assess model stability. Higher variance suggests model performance changes significantly across different time periods.

## Overfitting Analysis

Overfitting was evaluated by comparing training and test errors.

Guideline:

* Train MAE much lower than Test MAE → Potential overfitting
* Similar Train and Test MAE → Better generalization

Random Forest and Linear Regression were both examined using this criterion.

## Deliverables

### Saved Report

Cross-validation summary:

`reports/cv_results.md`

### Saved Visualization

Cross-validation MAE plot:

`reports/figures/cv_mae_scores.png`

### Input Files

Training and test datasets:

* `data/processed/X_train.npy`
* `data/processed/X_test.npy`
* `data/processed/y_train.npy`
* `data/processed/y_test.npy`

## Key Findings

* TimeSeriesSplit was used instead of random K-Fold to preserve chronological order.
* Cross-validation was performed exclusively on training data.
* Model performance was evaluated across multiple folds to estimate robustness.
* Variability across folds was analyzed using MAE standard deviation.
* Hold-out test performance was compared against cross-validation results.
* Overfitting risk was assessed by comparing training and test errors.
* Results provide a more reliable estimate of future model performance than a single train/test split.

# DAY 13 Hyperparameter Tuning with GridSearchCV

## Objective

To improve Random Forest model performance by tuning key hyperparameters using time-aware cross-validation while preventing data leakage from future observations.

## Methodology

### Data Used

The following preprocessed datasets were loaded:

* `data/processed/X_train.npy`
* `data/processed/X_test.npy`
* `data/processed/y_train.npy`
* `data/processed/y_test.npy`

Only the training data was used during hyperparameter tuning.

### Cross-Validation Strategy

A `TimeSeriesSplit` cross-validator with 3 splits was used to preserve chronological ordering of observations.

This approach ensures that:

* Earlier observations are used to predict later observations.
* Future information is never used during training.
* Data leakage is prevented.

###  DAY 13 Hyperparameter Grid

A small parameter grid was selected to keep runtime reasonable while exploring meaningful model configurations.

| Parameter          | Values Tested | Purpose                                                                                                  |
| ------------------ | ------------- | -------------------------------------------------------------------------------------------------------- |
| `n_estimators`     | 50, 100, 200  | Controls the number of trees in the forest. More trees generally improve stability but increase runtime. |
| `max_depth`        | None, 8, 16   | Limits tree depth and helps control model complexity.                                                    |
| `min_samples_leaf` | 1, 3, 5       | Controls minimum observations per leaf node and helps reduce overfitting.                                |

Total parameter combinations evaluated:

```text
3 × 3 × 3 = 27 combinations
```

### Grid Search Configuration

The search was performed using:

* `GridSearchCV`
* `TimeSeriesSplit(n_splits=3)`
* Scoring metric: Mean Absolute Error (MAE)
* `refit=True`
* `n_jobs=-1`

The model was automatically refit using the best parameter combination found during cross-validation.

## Evaluation Procedure

1. Perform Grid Search using training data only.
2. Select the best parameter combination based on cross-validated MAE.
3. Refit the best estimator on the full training dataset.
4. Evaluate the tuned model once on the held-out test dataset.
5. Record final test metrics.

The test set was not used during tuning.

## Results

The Grid Search produced:

* Best parameter combination
* Best cross-validation MAE
* Final test MAE
* Final test RMSE
* Final test R² score

Runtime was also recorded to ensure the search remained practical for a standard laptop environment.

## Saved Artifacts

### Tuned Model

Best Random Forest model:

```text
models/random_forest_tuned.joblib
```

### Best Parameters

Optimal hyperparameters:

```text
models/rf_best_params.json
```

### Search Transparency

First rows of GridSearchCV results:

```text
reports/gridsearch_cv_results_head.csv
```

### Performance Summary

Summary of tuning results and evaluation metrics:

```text
reports/gridsearch_summary.csv
```

## Validation Checks

The following validation criteria were satisfied:

* TimeSeriesSplit used instead of random K-Fold.
* Hyperparameter search performed exclusively on training data.
* Mean Absolute Error used as the optimization metric.
* Best estimator automatically refit after tuning.
* Test set evaluated only once after model selection.
* Best parameters saved for reproducibility.
* Model artifact saved for deployment and future evaluation.
* Search results exported for mentor review and transparency.

## Key Findings

* Time-aware cross-validation provided a realistic estimate of future performance.
* Hyperparameter tuning explored multiple Random Forest configurations efficiently.
* The selected model represents the best-performing configuration within the defined search space.
* Exported artifacts allow full reproducibility of tuning results and model evaluation.
* Search runtime remained practical for an internship-scale machine learning project.


## Random Forest Hyperparameter Tuning

### Objective

Optimize Random Forest performance using GridSearchCV with TimeSeriesSplit.

### Parameter Grid

- n_estimators: [50, 100, 200]
- max_depth: [None, 8, 16]
- min_samples_leaf: [1, 3, 5]

### Validation Strategy

- TimeSeriesSplit (3 folds)
- MAE scoring
- Refit best estimator

### Outputs

- models/random_forest_tuned.joblib
- models/rf_best_params.json
- reports/gridsearch_results.md

### Execution

```bash
python src/tune_random_forest.py
```


# ##Model Comparison and Champion Selection### --day14

## Objective

Evaluate and compare the performance of three machine learning models for mushroom yield prediction:

1. Linear Regression
2. Random Forest (Default)
3. Random Forest (Tuned)

The comparison uses the same untouched chronological test set to ensure a fair evaluation and prevent data leakage.

---

## Evaluation Methodology

The following metrics were used to assess model performance:

* Cross-Validation MAE (CV MAE)
* Test MAE (Mean Absolute Error)
* RMSE (Root Mean Squared Error)
* R² Score
* Training Time
* Model Interpretability

All models were evaluated on the same test dataset generated during the chronological train/test split.

---

## Model Comparison Table

| Model                 | CV MAE  | Test MAE | RMSE    | R²      | Training Time (s) | Interpretability |
| --------------------- | ------- | -------- | ------- | ------- | ----------------- | ---------------- |
| Linear Regression     | Replace | Replace  | Replace | Replace | Replace           | High             |
| Random Forest Default | Replace | Replace  | Replace | Replace | Replace           | Medium           |
| Random Forest Tuned   | Replace | Replace  | Replace | Replace | Replace           | Medium-Low       |

---

## Champion Model## --day14

*Selected Model:* Replace with actual champion model

### Selection Rationale

The champion model was selected based primarily on the lowest Test MAE while also considering RMSE, R² score, model complexity, and interpretability.

If multiple models achieved nearly identical MAE values, the simpler Linear Regression model would be preferred due to:

* Greater transparency
* Easier stakeholder communication
* Simpler maintenance
* Reduced deployment complexity

In this project, the selected champion model demonstrated the best balance between predictive performance and practical deployment considerations.

---

## Predicted vs Actual Yield

A scatter plot comparing actual yield values against model predictions was generated for the champion model.

*Saved Figure:*

reports/figures/pred_vs_actual.png

### Interpretation

* Points close to the diagonal line indicate accurate predictions.
* Larger deviations from the diagonal represent prediction errors.
* A strong clustering around the diagonal suggests good model performance on unseen data.

---

## Deployment Recommendation

The selected champion model is recommended for deployment as a decision-support tool for mushroom yield forecasting.

The model can assist growers by providing estimated yield predictions based on environmental sensor inputs.

---

## Known Limitations and Edge Cases

### Sensor Range Limitations

The model was trained on a limited range of environmental conditions. Predictions for temperature, humidity, or CO₂ values outside the observed training range may be unreliable.

### Seasonality

The dataset covers a limited time period and may not fully capture long-term seasonal effects or environmental variations.

### Unseen Conditions

Extreme growing conditions not represented in the training data may reduce prediction accuracy.

### Synthetic Dataset Constraints

The project uses generated data for development purposes. Real-world mushroom farms may exhibit additional variability not captured by the synthetic dataset.

### Operational Use

Model predictions should be considered advisory only.

The model is intended to support decision-making and should not replace grower expertise, operational experience, or field observations.
---

## Deliverables

Generated artifacts:

* reports/model_comparison.csv
* reports/model_comparison.md
* reports/figures/pred_vs_actual.png

These files provide a complete summary of model evaluation, champion selection, and deployment readiness assessment.



# ## DAY 15 Inference and Deployment### --day15

## Objective

Deploy the champion machine learning model as a reusable inference module capable of predicting mushroom yield from environmental sensor readings.

---

## Saved Artifacts

The following artifacts are required for inference:

text
models/
├── random_forest_tuned.joblib
├── minmax_scaler_train.joblib
├── feature_cols.json


### Artifact Description

| File                       | Purpose                     |
| -------------------------- | --------------------------- |
| random_forest_tuned.joblib | Champion prediction model   |
| minmax_scaler_train.joblib | Feature scaling transformer |
| feature_cols.json          | Training feature order      |

---

## Run Inference

### Example Command

From the project root:

bash
python src/predict.py --temperature 22 --humidity 88 --co2 920


### Example Output

text
Predicted Yield: 16.42 kg


Actual output will vary depending on the trained model.

---

## Python API

The module exposes a public prediction function:

python
from src.predict import predict_yield

prediction = predict_yield(
    temperature_c=22,
    humidity_pct=88,
    co2_ppm=920
)

print(prediction)


### Helper Function

python
from src.predict import make_prediction

prediction = make_prediction(
    temperature=22,
    humidity=88,
    co2=920
)


---

## Reproducibility Notes

### Random Seeds

The following seed was used throughout model development:

python
np.random.seed(42)
random_state=42


### Library Versions

Generate exact versions using:

bash
pip freeze > requirements.txt


Typical core libraries:

* numpy
* pandas
* scikit-learn
* matplotlib
* joblib
* pyarrow

---

## Dependency Installation

Create a clean virtual environment and install dependencies:

bash
pip install -r requirements.txt


---

## Validation

Inference was validated by:

1. Loading saved artifacts from the models directory.
2. Running predictions through predict.py.
3. Comparing results against manual model calls.
4. Confirming identical predictions for the same inputs.

---

## Deployment Notes

* All paths are relative to the project root.
* Compatible with Streamlit deployment.
* Predictions are advisory only.
* Outputs should support grower decision-making and not replace operational judgment.


DAY 16
# Streamlit Application

## Run the App

Activate your virtual environment:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch Streamlit:

```bash
streamlit run app.py
```

The application will open in your browser.

---

## Features

* Temperature input slider
* Humidity input slider
* CO₂ input slider
* Yield prediction in kilograms
* Cached model loading using `@st.cache_resource`
* Out-of-range sensor warnings
* Advisory prediction disclaimer

---

## Example Usage

Input:

* Temperature: 22°C
* Humidity: 88%
* CO₂: 900 ppm

Output:

```text
Estimated Yield: XX.XX kg
```

The exact value depends on the trained champion model.

---

## Screenshot

Save a screenshot after successful local execution:

```text
reports/streamlit_app.png
```

This screenshot demonstrates:

* Successful application startup
* Sensor input controls
* Yield prediction display
* Working inference pipeline


DAY 17
# Enhanced Streamlit Dashboard

## Overview

The Streamlit application provides an interactive interface for mushroom yield forecasting using environmental sensor readings.

The dashboard is designed for farm managers and operations teams rather than machine learning practitioners.

---

## Features

### Yield Prediction

Users can enter:

* Temperature (°C)
* Relative Humidity (%)
* CO₂ Concentration (ppm)

and receive an estimated yield prediction in kilograms.

---

### Input Validation

Warnings are displayed when sensor readings fall outside the ranges observed during model training.

This helps communicate uncertainty and improve trust in predictions.

---

### What-if Analysis

The dashboard includes a sensitivity analysis chart showing:

**Predicted Yield vs Humidity**

while holding temperature and CO₂ constant.

This helps users understand how environmental adjustments may affect expected production.

---

### Model Metadata

An expandable information section displays:

* Model Version
* Last Training Date
* Test MAE
* Input Features

This improves transparency and reproducibility.

---

### Methodology Section

A methodology expander explains:

* Feature scaling
* Model inference workflow
* Output interpretation

and links users to the technical project documentation.

---

## Running the Dashboard

Activate the virtual environment:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run Streamlit:

```bash
streamlit run app.py
```

---

## Screenshot

Save a screenshot after successful execution:

```text
reports/streamlit_dashboard_v2.png
```

The screenshot should display:

* Input controls
* Yield prediction
* Sensitivity chart
* Metadata expander

---

## Notes

The dashboard is intended as a decision-support tool.

Predictions should be used alongside operational expertise, environmental monitoring, and grower judgment.





