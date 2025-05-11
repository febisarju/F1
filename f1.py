import fastf1
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import os

# Create cache directory if it doesn't exist
os.makedirs("f1_cache", exist_ok=True)

# Enable FastF1 cache
fastf1.Cache.enable_cache("f1_cache")

def load_sector_data():
    session = fastf1.get_session(2024, "China", "R")
    session.load()
    laps = session.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]].copy()
    laps.dropna(inplace=True)

    for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
        laps[f"{col} (s)"] = laps[col].dt.total_seconds()

    sector_times = laps.groupby("Driver")[["Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]].mean().reset_index()
    avg_laptimes = laps.groupby("Driver")["LapTime (s)"].mean().reset_index()
    return sector_times, avg_laptimes

def get_qualifying_data():
    df = pd.DataFrame({
        "Driver": ["Oscar Piastri", "George Russell", "Lando Norris", "Max Verstappen", "Lewis Hamilton",
                   "Charles Leclerc", "Isack Hadjar", "Andrea Kimi Antonelli", "Yuki Tsunoda", "Alexander Albon",
                   "Esteban Ocon", "Nico Hülkenberg", "Fernando Alonso", "Lance Stroll", "Carlos Sainz Jr.",
                   "Pierre Gasly", "Oliver Bearman", "Jack Doohan", "Gabriel Bortoleto", "Liam Lawson"],
        "QualifyingTime (s)": [90.641, 90.723, 90.793, 90.817, 90.927,
                               91.021, 91.079, 91.103, 91.638, 91.706,
                               91.625, 91.632, 91.688, 91.773, 91.840,
                               91.992, 92.018, 92.092, 92.141, 92.174]
    })
    driver_map = {
        "Oscar Piastri": "PIA", "George Russell": "RUS", "Lando Norris": "NOR", "Max Verstappen": "VER",
        "Lewis Hamilton": "HAM", "Charles Leclerc": "LEC", "Isack Hadjar": "HAD", "Andrea Kimi Antonelli": "ANT",
        "Yuki Tsunoda": "TSU", "Alexander Albon": "ALB", "Esteban Ocon": "OCO", "Nico Hülkenberg": "HUL",
        "Fernando Alonso": "ALO", "Lance Stroll": "STR", "Carlos Sainz Jr.": "SAI", "Pierre Gasly": "GAS",
        "Oliver Bearman": "BEA", "Jack Doohan": "DOO", "Gabriel Bortoleto": "BOR", "Liam Lawson": "LAW"
    }
    df["DriverCode"] = df["Driver"].map(driver_map)
    return df

def predict_with_sector_times():
    qualifying_df = get_qualifying_data()
    sector_times, avg_laptimes = load_sector_data()

    merged = qualifying_df.merge(sector_times, left_on="DriverCode", right_on="Driver", how="left")
    X = merged[["QualifyingTime (s)", "Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]].fillna(0)
    y = avg_laptimes["LapTime (s)"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=38)
    model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=38)
    model.fit(X_train, y_train)

    qualifying_df["PredictedRaceTime (s)"] = model.predict(X)
    qualifying_df = qualifying_df.sort_values(by="PredictedRaceTime (s)")

    print("\n🏁 2025 Predictions with Sector Times 🏁")
    print(qualifying_df[["Driver", "PredictedRaceTime (s)"]])
    print(f"\nModel MAE: {mean_absolute_error(y_test, model.predict(X_test)):.2f} s")

def predict_without_sector_times():
    qualifying_df = get_qualifying_data()
    session = fastf1.get_session(2024, "China", "R")
    session.load()

    laps = session.laps[["Driver", "LapTime"]].copy()
    laps.dropna(subset=["LapTime"], inplace=True)
    laps["LapTime (s)"] = laps["LapTime"].dt.total_seconds()

    merged = qualifying_df.merge(laps, left_on="DriverCode", right_on="Driver")
    if merged.empty:
        raise ValueError("Merged data is empty.")

    X = merged[["QualifyingTime (s)"]]
    y = merged["LapTime (s)"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=39)
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=39)
    model.fit(X_train, y_train)

    qualifying_df["PredictedRaceTime (s)"] = model.predict(qualifying_df[["QualifyingTime (s)"]])
    qualifying_df = qualifying_df.sort_values(by="PredictedRaceTime (s)")

    print("\n🏁 2025 Predictions with Only Qualifying Time 🏁")
    print(qualifying_df[["Driver", "PredictedRaceTime (s)"]])
    print(f"\nModel MAE: {mean_absolute_error(y_test, model.predict(X_test)):.2f} s")

# Run both predictions
if __name__ == "__main__":
    predict_with_sector_times()
    predict_without_sector_times()