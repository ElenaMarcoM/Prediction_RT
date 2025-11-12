from src_elena import *
import os
import pandas as pd
import numpy as np

path = os.path.join("resources", "smrt_fingerprints.csv")
df = pd.read_csv(path)

print(type(df))
print(df.shape)
print(df.head(5))

v_cols = [c for c in df.columns if c.lower().startswith('v')]
v_cols = sorted(v_cols, key=lambda c: int(c[1:]) if c[1:].isdigit() else float('inf'))

for c in v_cols + ['rt']:
    df[c] = pd.to_numeric(df[c], errors='coerce')

X = df[v_cols].to_numpy(dtype=float)
y = df['rt'].to_numpy(dtype=float)

print(X.shape, y.shape)
print(X[1])
print(y[1])