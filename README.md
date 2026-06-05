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