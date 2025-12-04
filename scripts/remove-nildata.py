import pandas as pd

df = pd.read_csv("data.csv")

# Set threshold (e.g., keep columns that have at least 5% non-NaN values)
threshold = 0.05

df = df.loc[:, df.notna().mean() >= threshold]

df.to_csv("cleaned_health_data.csv", index=False)

print("Removed sparse columns! New shape:", df.shape)
