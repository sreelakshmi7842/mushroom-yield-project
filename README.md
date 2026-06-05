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