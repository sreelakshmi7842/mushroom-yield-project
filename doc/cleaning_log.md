\# Cleaning Log — Polyhouse Sensor Dataset



\## Dataset

\- Source: data/raw/polyhouse\_sensors.csv

\- Rows before cleaning: 365

\- Columns: timestamp, temperature\_c, humidity\_pct, co2\_ppm, yield\_kg



\## Validity Rules Applied

\- humidity\_pct: must be between 50 and 100

\- temperature\_c: must be between 10 and 35

\- co2\_ppm: must be between 400 and 2000

\- yield\_kg: must not be null



\## Strategy

\- Invalid rows filtered using explicit threshold rules

\- Sensor columns forward-filled for short gaps (max 2)

\- Rows with null yield\_kg dropped (never imputed)

\- Duplicate timestamps removed, keeping last entry



\## Results

\- Rows after cleaning: TBD

\- Rows dropped: TBD

