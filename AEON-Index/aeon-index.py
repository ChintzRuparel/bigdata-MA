import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def normalize(series):
    return (series - series.min()) / (series.max() - series.min() + 1e-9)


df = pd.read_csv("/Users/chintzruparel/Documents/GitHub/bigdata-MA/Final Dataset/Multiset-Dataset.csv")
df["Date/Time"] = pd.to_datetime(df["Date/Time"])

# ======== COMPONENTS ========

BEM = (
    0.40 * normalize(df["Heart Rate Variability (ms)"]) +
    0.25 * (1 - normalize(df["Resting Heart Rate (count/min)"])) +
    0.20 * normalize(df["Blood Oxygen Saturation (%)"]) +
    0.15 * (1 - normalize(df["Respiratory Rate (count/min)"]))
)

KEF = (
    0.35 * normalize(df["Step Count (count)"]) +
    0.35 * normalize(df["Active Energy (kcal)"]) +
    0.20 * normalize(df["Walking + Running Distance (mi)"]) +
    0.10 * normalize(df["VO2 Max (ml/(kg·min))"])
)

CSC = (
    0.40 * normalize(df["Sleep Analysis [Total] (hr)"]) +
    0.30 * normalize(df["Sleep Analysis [Deep] (hr)"]) +
    0.20 * normalize(df["Sleep Analysis [REM] (hr)"]) +
    0.10 * (1 - normalize(df["Sleep Analysis [Awake] (hr)"]))
)

ASG = (
    0.40 * (1 - normalize(abs(df["feels_like"] - 22))) +
    0.30 * (1 - normalize(df["humidity"])) +
    0.20 * (1 - normalize(df["rain"])) +
    0.10 * (1 - normalize(df["wind_speed"]))
)

CLI = 1 - normalize(df["screen_on"])


df["AEON_Index"] = 100 * (
    0.35 * BEM +
    0.25 * CSC +
    0.20 * KEF +
    0.15 * ASG +
    0.05 * CLI
)

# ======== PLOTTING ========

plt.figure(figsize=(15,6))
plt.plot(df["Date/Time"], df["AEON_Index"], label="AEON Index", linewidth=3, color="cyan")
plt.plot(df["Date/Time"], normalize(df["screen_on"]) * 100, label="Screen Time (scaled)", alpha=0.6)
plt.plot(df["Date/Time"], normalize(df["feels_like"]) * 100, label="Feels Like Temp (scaled)", alpha=0.6)
plt.plot(df["Date/Time"], normalize(df["Resting Heart Rate (count/min)"]) * 100, label="Resting HR (scaled)", alpha=0.6)

plt.title("A.E.O.N. Index vs Weather, Screen Time & Physiological Signals")
plt.ylabel("Score (0–100)")
plt.legend()
plt.grid(True)
plt.show()
