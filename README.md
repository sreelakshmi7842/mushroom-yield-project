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

## Data Cleaning Strategy (Phase 1, Task 2)

**1. Outliers & Anomalies (Threshold Rules)**
Filtered humidity (50-100%), temperature (10-35°C), and CO2 (400-2000 ppm) to remove hard sensor failures (e.g., a dead humidity probe reading 0% or environmental spikes outside biological survival ranges). 

**2. Missing Values (Imputation vs. Row Removal)**
Handled short sensor dropouts (power blips, calibration gaps) using forward-fill (`ffill`) with a strict limit of 2 periods, assuming short-term microclimate stability. Rows completely missing the `yield_kg` target variable were dropped entirely, as we cannot train or evaluate models on missing ground-truth labels.

**3. Duplicates**
Removed exact timestamp duplicates, keeping the `last` entry under the assumption it represents the most recent or corrected system export.

### Initial Missing Value Report (Pre-Cleaning)
Before applying the imputation and filtering rules, the dataset exhibited the following missing values:

```text
timestamp         0
temperature_c    14
humidity_pct     14
co2_ppm          14
yield_kg         42
dtype: int64
(Note: Be sure to update these placeholder numbers with the actual output from your terminal).

Project Structure
Plaintext
mushroom-yield-project/
│
├── data/
│   ├── raw/              # Raw sensor data uploads (excluded from Git)
│   ├── interim/          # Intermediate data that has been cleaned
│   └── processed/        # Standardized datasets ready for modeling
│
├── models/               # Serialized production-ready model files
│
├── notebooks/            # Jupyter notebooks for exploratory analysis
│
├── src/
│   ├── clean.py          # Data cleaning and imputation script
│   └── smoke_test.py     # Base validation environment script
│
├── docs/
│   └── cleaning_log.md   # Documentation for data processing decisions
│
├── .gitignore            # Excludes virtual environments and large data files
├── README.md             # Project documentation and workflow guide
└── requirements.txt      # Project dependencies
Installation
Clone the repository:

Bash
git clone https://github.com/sreelakshmi7842/mushroom-yield-project
cd mushroom-yield-project
Create and activate a virtual environment:

Windows
Bash
python -m venv venv
venv\Scripts\activate
Install dependencies:

Bash
pip install -r requirements.txt
Running the Project
Run Environment Validation
Bash
python src/smoke_test.py
Run Data Cleaning Pipeline
Bash
python src/clean.py
Future Enhancements
Automated data ingestion

