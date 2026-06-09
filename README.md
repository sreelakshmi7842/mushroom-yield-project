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

