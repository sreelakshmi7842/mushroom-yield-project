import pandas as pd

try:

    df = pd.read_parquet(
        "data/processed/02_cleaned.parquet"
    )

    print("SUCCESS")

    print(df.shape)

    print(
        "Null yield values:",
        df["yield_kg"].isna().sum()
    )

except Exception as e:

    print("FAILED")

    print(e)