import pandas as pd
df = pd.read_parquet('data/gsm8k/test.parquet')
print("Columns:", df.columns.tolist())
print("\nFirst row:")
print(df.iloc[0])
