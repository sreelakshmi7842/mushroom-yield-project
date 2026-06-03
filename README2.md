\## Problem Statement 

Predicting daily mushroom yield (kg) in a climate-controlled polyhouse using real-world sensor 

readings for temperature (°C), relative humidity (%), and CO₂ (ppm). This repository serves as a 

version-controlled data pipeline resilient against model breakdown due to sudden hardware or 

data drift updates. 

\## Project Structure 

```text 

├── data/ 

│   ├── processed/        

│   └── raw/              

├── models/               

├── notebooks/            

├── src/ 

\# Standardized datasets ready for modeling 

\# Raw sensor data uploads (Excluded from Git) 

\# Serialized production-ready model files 

\# Jupyter notebooks for exploratory data analysis 

│   └── smoke\_test.py     # Base validation environment script 

├── .gitignore            

\# Explicitly excludes environment and large log assets 

├── README.md             

└── requirements.txt      

\# Project roadmap and run protocols 

\# Pinned infrastructure dependencies

