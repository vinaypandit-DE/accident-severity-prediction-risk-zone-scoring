from pathlib import Path
import pandas as pd

csv_file = next(Path("cleaned_data/road_accident_cleaned").glob("*.csv"))

df = pd.read_csv(csv_file)

print(df.head())

print(df.columns.tolist())

print(df.dtypes)