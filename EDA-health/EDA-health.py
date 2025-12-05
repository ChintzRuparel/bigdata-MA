import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------------------
# LOAD DATA
# ------------------------------
df = pd.read_csv("/Users/chintzruparel/Documents/GitHub/bigdata-MA/datasets/Final_dataset.csv")

# Convert timestamps
df["Date/Time"] = pd.to_datetime(df["Date/Time"])

# Sort just in case
df = df.sort_values("Date/Time")

print("\n====================")
print("Basic Dataset Info")
print("====================")
print(df.info())
print(df.describe())
print("\n")

# ------------------------------
# CORRELATION MATRIX
# ------------------------------
numeric_df = df.select_dtypes(include=[np.number])

corr = numeric_df.corr()
plt.figure(figsize=(18,12))
sns.heatmap(corr, cmap="coolwarm", annot=False)
plt.title("Correlation Matrix of Physiological Metrics")
plt.tight_layout()
plt.savefig("correlation_matrix.png")
print("Correlation matrix saved as correlation_matrix.png\n")

# ------------------------------
# TIME SERIES PLOTS
# ------------------------------
metrics_to_plot = [
    "Active Energy (kcal)",
    "Apple Exercise Time (min)",
    "Blood Oxygen Saturation (%)",
    "Heart Rate [Avg] (count/min)",
    "Heart Rate Variability (ms)",
    "Respiratory Rate (count/min)",
    "Environmental Audio Exposure (dBASPL)",
]

for col in metrics_to_plot:
    plt.figure(figsize=(12,4))
    plt.plot(df["Date/Time"], df[col])
    plt.xlabel("Time")
    plt.ylabel(col)
    plt.title(f"Time Trend: {col}")
    plt.tight_layout()
    plt.savefig(f"time_{col.replace(' ','_')}.png")
    plt.close()

print("Time-series plots saved.\n")

# ------------------------------
# ACTIVITY VS HEART RESPONSE
# ------------------------------
activity_cols = [
    "Active Energy (kcal)",
    "Apple Exercise Time (min)",
    "Physical Effort (kcal/hr·kg)",
    "Step Count (count)",
]

heart_cols = [
    "Heart Rate [Min] (count/min)",
    "Heart Rate [Max] (count/min)",
    "Heart Rate [Avg] (count/min)",
    "Heart Rate Variability (ms)",
]

print("============= Activity ↔ Heart Response Analysis =============")

for a in activity_cols:
    for h in heart_cols:
        correlation = df[a].corr(df[h])
        print(f"Correlation between {a} and {h}: {correlation:.3f}")

print("\n")

# ------------------------------
# DETECT SPIKES OR ANOMALIES
# ------------------------------
def detect_spikes(series, z_thresh=2.5):
    z_scores = (series - series.mean()) / series.std()
    return series[z_scores.abs() > z_thresh]

print("============= Spike Detection =============")
for col in metrics_to_plot:
    spikes = detect_spikes(df[col])
    if len(spikes) > 0:
        print(f"{col} has {len(spikes)} anomaly points:")
        print(spikes)
    else:
        print(f"No anomalies in {col}")
