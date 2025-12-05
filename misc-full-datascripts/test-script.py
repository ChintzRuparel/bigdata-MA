import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =======================================
# 1. LOAD & CLEAN DATA
# =======================================
df = pd.read_csv("Final Dataset/Multiset-Dataset.csv", parse_dates=["Date/Time"])
df = df.sort_values("Date/Time").drop_duplicates()

# Add readable label for screen state
df["screen_state"] = df["screen_on"].map({1: "ON", 0: "OFF"})

# Daily aggregation
daily = df.groupby(df["Date/Time"].dt.date).agg({
    # Activity
    "Active Energy (kcal)": "sum",
    "Resting Energy (kcal)": "sum",
    "Apple Exercise Time (min)": "sum",
    "Step Count (count)": "sum",
    "Walking + Running Distance (mi)": "sum",

    # Sleep
    "Sleep Analysis [Total] (hr)": "mean",
    "Sleep Analysis [Asleep] (hr)": "mean",

    # Screen
    "screen_on": "sum",

    # Weather
    "temperature": "mean",
    "feels_like": "mean",
    "humidity": "mean",
    "rain": "sum",
    "wind_speed": "mean"
})

daily.rename(columns={"screen_on": "Screen On Minutes"}, inplace=True)

# =======================================
# 2. HIGH-LEVEL SUMMARY
# =======================================
print("\n===== GENERAL SUMMARY =====\n")
print("Average Temperature:", round(daily["temperature"].mean(), 2), "°C")
print("Average Humidity:", round(daily["humidity"].mean(), 2), "%")
print("Average Daily Rain:", round(daily["rain"].mean(), 2), "mm")
print("Average Daily Steps:", round(daily["Step Count (count)"].mean(), 2))
print("Average Sleep Hours:", round(daily["Sleep Analysis [Asleep] (hr)"].mean(), 2))

# Activity ratio
daily["Activity Ratio"] = (
    daily["Active Energy (kcal)"] /
    (daily["Active Energy (kcal)"] + daily["Resting Energy (kcal)"])
)

# =======================================
# 3. CORRELATION MATRIX INCLUDING WEATHER
# =======================================
corr_features = [
    "Active Energy (kcal)",
    "Step Count (count)",
    "Apple Exercise Time (min)",
    "Walking + Running Distance (mi)",
    "Sleep Analysis [Asleep] (hr)",
    "Screen On Minutes",
    "temperature",
    "humidity",
    "rain",
    "wind_speed"
]

corr = daily[corr_features].corr()
print("\n===== CORRELATION MATRIX (Health + Weather) =====\n")
print(corr)

# =======================================
# 4. WEATHER EFFECTS ON BEHAVIOR
# =======================================
print("\n===== WEATHER → ACTIVITY INSIGHTS =====\n")

# Temperature vs activity
temp_steps_corr = corr.loc["temperature", "Step Count (count)"]
print("Correlation: Temperature ↔ Steps:", round(temp_steps_corr, 3))

# Humidity vs walking
humid_walk_corr = corr.loc["humidity", "Walking + Running Distance (mi)"]
print("Correlation: Humidity ↔ Walking Distance:", round(humid_walk_corr, 3))

# Rain vs exercise
rain_exercise_corr = corr.loc["rain", "Apple Exercise Time (min)"]
print("Correlation: Rain ↔ Exercise Time:", round(rain_exercise_corr, 3))

# Weather vs screen time
temp_screen_corr = corr.loc["temperature", "Screen On Minutes"]
print("Correlation: Temperature ↔ Screen Time:", round(temp_screen_corr, 3))

print("\n--- INTERPRETATION GUIDE ---")
print("""
Positive correlation   → Both increase together  
Negative correlation   → One increases while the other decreases  
Near zero              → No meaningful effect  
""")

# =======================================
# 5. SCREEN TIME RELATIONSHIPS
# =======================================
print("\n===== SCREEN TIME RELATIONSHIPS =====\n")

print("Screen Time ↔ Sleep:", 
      round(np.corrcoef(daily["Screen On Minutes"], daily["Sleep Analysis [Asleep] (hr)"])[0,1], 3))

print("Screen Time ↔ Steps:", 
      round(np.corrcoef(daily["Screen On Minutes"], daily["Step Count (count)"])[0,1], 3))

print("Screen Time ↔ Temperature:", 
      round(np.corrcoef(daily["Screen On Minutes"], daily["temperature"])[0,1], 3))

# =======================================
# 6. WEATHER IMPACT DETECTION (ANOMALIES)
# =======================================
print("\n===== WEATHER ANOMALIES =====\n")

hot_days = daily[daily["temperature"] > daily["temperature"].mean() + 2*daily["temperature"].std()]
cold_days = daily[daily["temperature"] < daily["temperature"].mean() - 2*daily["temperature"].std()]
rain_spikes = daily[daily["rain"] > daily["rain"].mean() + 2*daily["rain"].std()]

print("🔥 Extreme Hot Days:")
print(hot_days[["temperature", "Step Count (count)", "Screen On Minutes"]])

print("\n❄️ Extreme Cold Days:")
print(cold_days[["temperature", "Step Count (count)", "Screen On Minutes"]])

print("\n🌧️ Heavy Rain Days:")
print(rain_spikes[["rain", "Step Count (count)", "Screen On Minutes"]])

# =======================================
# 7. SIMPLE WEATHER → STEP COUNT MODEL
# =======================================
print("\n===== WEATHER-BASED STEP PREDICTION =====\n")
from sklearn.linear_model import LinearRegression

model = LinearRegression()
X = daily[["temperature", "humidity", "rain", "wind_speed"]]
y = daily["Step Count (count)"]

model.fit(X, y)
coefs = dict(zip(X.columns, model.coef_))

print("Model Coefficients (Impact on Step Count):")
for k, v in coefs.items():
    print(f"{k}: {round(v, 2)} steps per unit change")

# =======================================
# 8. OPTIONAL PLOT — TEMP VS STEPS
# =======================================
plt.scatter(daily["temperature"], daily["Step Count (count)"])
plt.title("Temperature vs Step Count")
plt.xlabel("Temperature (°C)")
plt.ylabel("Steps")
plt.show()
