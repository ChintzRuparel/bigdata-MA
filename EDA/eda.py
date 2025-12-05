import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ------------------------------
# 0. CONFIG
# ------------------------------
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# Create output folder
os.makedirs("eda_outputs", exist_ok=True)

# ------------------------------
# 1. LOAD DATA
# ------------------------------
df = pd.read_csv("/Users/chintzruparel/Documents/GitHub/bigdata-MA/Final Dataset/Multiset-Dataset.csv", parse_dates=["Date/Time"])

# ------------------------------
# 2. BASIC CLEANING
# ------------------------------
df = df.sort_values("Date/Time")
df = df.drop_duplicates()

# Convert screen_on into categorical label
df["screen_state"] = df["screen_on"].map({1: "ON", 0: "OFF"})

# Automatically detect numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns

# ------------------------------
# 3. BASIC DESCRIPTIVE STATS
# ------------------------------
print("\n===== BASIC STATISTICS =====\n")
print(df.describe())

df.describe().to_csv("eda_outputs/descriptive_stats.csv")

# ------------------------------
# 4. CORRELATION MATRIX (HEATMAP)
# ------------------------------
corr = df[numeric_cols].corr()

plt.figure(figsize=(18, 15))
sns.heatmap(corr, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("eda_outputs/correlation_heatmap.png")
plt.close()

# Save correlation matrix to CSV
corr.to_csv("eda_outputs/correlation_matrix.csv")

# ------------------------------
# 5. HISTOGRAMS FOR ALL NUMERIC FEATURES
# ------------------------------
for col in numeric_cols:
    plt.figure()
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.tight_layout()
    plt.savefig(f"eda_outputs/hist_{col}.png")
    plt.close()

# ------------------------------
# 6. BOX PLOTS FOR OUTLIERS
# ------------------------------
for col in numeric_cols:
    plt.figure()
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot: {col}")
    plt.tight_layout()
    plt.savefig(f"eda_outputs/box_{col}.png")
    plt.close()

# ------------------------------
# 7. TIME SERIES PLOTS (KEY METRICS)
# ------------------------------
time_series_cols = [
    "Active Energy (kcal)",
    "Heart Rate [Avg] (count/min)",
    "Step Count (count)",
    "Sleep Analysis [Asleep] (hr)",
    "screen_on",
    "temperature"
]

for col in time_series_cols:
    if col in df.columns:
        plt.figure()
        plt.plot(df["Date/Time"], df[col])
        plt.title(f"Time Series: {col}")
        plt.xlabel("Time")
        plt.ylabel(col)
        plt.tight_layout()
        plt.savefig(f"eda_outputs/timeseries_{col}.png")
        plt.close()

# ------------------------------
# 8. SCATTER PLOTS: RELATIONSHIPS
# ------------------------------
scatter_pairs = [
    ("Step Count (count)", "temperature"),
    ("Heart Rate [Avg] (count/min)", "screen_on"),
    ("Sleep Analysis [Asleep] (hr)", "screen_on"),
    ("Active Energy (kcal)", "Step Count (count)"),
]

for x, y in scatter_pairs:
    if x in df.columns and y in df.columns:
        plt.figure()
        sns.scatterplot(x=df[x], y=df[y], hue=df["screen_state"])
        plt.title(f"{x} vs {y}")
        plt.tight_layout()
        plt.savefig(f"eda_outputs/scatter_{x}_vs_{y}.png")
        plt.close()

# ------------------------------
# 9. PAIRPLOT (MULTI-VARIATE)
# ------------------------------
selected_for_pairplot = [
    "Active Energy (kcal)",
    "Step Count (count)",
    "Heart Rate [Avg] (count/min)",
    "Sleep Analysis [Asleep] (hr)",
    "temperature",
    "screen_on"
]

pair_df = df[selected_for_pairplot].dropna()

sns.pairplot(pair_df, corner=True)
plt.savefig("eda_outputs/pairplot.png")
plt.close()

# ------------------------------
# 10. WEATHER VS HEALTH ANALYSIS
# ------------------------------
weather_cols = ["temperature", "humidity", "rain", "wind_speed"]
health_cols = ["Step Count (count)", "Heart Rate [Avg] (count/min)", "Sleep Analysis [Asleep] (hr)"]

for w in weather_cols:
    if w not in df.columns:
        continue
    for h in health_cols:
        if h not in df.columns:
            continue
        plt.figure()
        sns.scatterplot(x=df[w], y=df[h], hue=df["screen_state"])
        plt.title(f"{h} vs {w}")
        plt.tight_layout()
        plt.savefig(f"eda_outputs/weather_{h}_vs_{w}.png")
        plt.close()

# ------------------------------
# 11. SCREEN TIME IMPACT ANALYSIS
# ------------------------------
plt.figure()
sns.boxplot(x=df["screen_state"], y=df["Heart Rate [Avg] (count/min)"])
plt.title("Heart Rate vs Screen State (ON/OFF)")
plt.savefig("eda_outputs/hr_vs_screen.png")
plt.close()

plt.figure()
sns.boxplot(x=df["screen_state"], y=df["Step Count (count)"])
plt.title("Steps vs Screen State (ON/OFF)")
plt.savefig("eda_outputs/steps_vs_screen.png")
plt.close()

plt.figure()
sns.boxplot(x=df["screen_state"], y=df["Sleep Analysis [Asleep] (hr)"])
plt.title("Sleep vs Screen State (ON/OFF)")
plt.savefig("eda_outputs/sleep_vs_screen.png")
plt.close()

# ------------------------------
# DONE
# ------------------------------
print("\n\n=======================================")
print("EDA COMPLETE! Check the 'eda_outputs/' folder.")
print("=======================================\n")
