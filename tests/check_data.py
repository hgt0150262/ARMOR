import pandas as pd

df = pd.read_parquet("data/gsm8k/train.parquet")
p = df.iloc[0]["prompt"]
print(f"Type: {type(p)}")
print(f"Content: {p}")
print(f"---")
if isinstance(p, list) and len(p) > 0:
    print(f"First element type: {type(p[0])}")
    print(f"First element: {p[0]}")
