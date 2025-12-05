import requests
import pandas as pd

def get_weather_data(latitude, longitude, start_date, end_date, output_file="weather_data.csv"):
    """
    Fetch historical weather between start_date and end_date (YYYY-MM-DD)
    and save to CSV.
    """

    url = "https://archive-api.open-meteo.com/v1/archive"  # historical endpoint

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "temperature_2m",
            "apparent_temperature",
            "relativehumidity_2m",
            "rain",
            "windspeed_10m",
            "pm2_5"
        ]
    }

    response = requests.get(url, params=params)
    data = response.json()

    if "hourly" not in data:
        print("Error: No weather data returned.")
        print(data)
        return

    hourly = data["hourly"]

    df = pd.DataFrame({
        "time": hourly["time"],
        "temperature": hourly["temperature_2m"],
        "feels_like": hourly["apparent_temperature"],
        "humidity": hourly["relativehumidity_2m"],
        "rain": hourly["rain"],
        "wind_speed": hourly["windspeed_10m"],
        "pm2_5": hourly.get("pm2_5", [None] * len(hourly["time"]))
    })

    df.to_csv(output_file, index=False)
    print(f"Weather data saved to {output_file}")


# -------------------------
# Example Usage With Your Dates
# -------------------------

if __name__ == "__main__":
    latitude = 40.7128    # NYC example
    longitude = -74.0060

    start_date = "2025-10-14"
    end_date = "2025-11-20"

    get_weather_data(latitude, longitude, start_date, end_date)
