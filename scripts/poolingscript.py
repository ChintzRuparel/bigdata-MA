import pandas as pd
import numpy as np

# Load data
df = pd.read_csv("data.csv")

# ---- Average pooling imputation ----
# Fill numeric NaNs with column mean
df = df.apply(
    lambda col: col.fillna(col.mean()) 
    if np.issubdtype(col.dtype, np.number) 
    else col
)

print("Filled missing numeric values using average pooling!")

# Save output
df.to_csv("pooled_health_data.csv", index=False)
print("Saved pooled dataset as pooled_health_data.csv")
