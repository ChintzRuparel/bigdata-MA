import pandas as pd

def hourly_to_minute(
    input_csv="/Users/chintzruparel/Documents/GitHub/bigdata-MA/WeatherData/weather_data.csv",
    output_csv="/Users/chintzruparel/Documents/GitHub/bigdata-MA/WeatherData/weather_data_minute.csv"
):
    # Load hourly CSV
    df = pd.read_csv(input_csv)

    # Convert 'time' column to datetime and set as index
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time')

    # Resample to 1-minute intervals and interpolate values
    df_minute = df.resample("1T").interpolate(method="linear")

    # Save to CSV
    df_minute.to_csv(output_csv)
    print(f"Minute-by-minute data saved to: {output_csv}")
    print(f"Total rows generated: {len(df_minute)}")


# ----------------------------
# Run Conversion
# ----------------------------
if __name__ == "__main__":
    hourly_to_minute()
