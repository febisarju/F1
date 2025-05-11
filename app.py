import streamlit as st
import pandas as pd
from f1 import get_qualifying_data, load_sector_data
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import fastf1

# Enable FastF1 cache
fastf1.Cache.enable_cache("f1_cache")

st.set_page_config(page_title="F1 2025 Race Predictions", layout="wide")
st.title("🏎️ F1 2025 Race Time Predictor")
st.write("Choose prediction mode and view predicted race times based on qualifying and/or sector data.")

option = st.selectbox("Select Prediction Mode", ("With Sector Times", "Without Sector Times"))

@st.cache_data(show_spinner=True)
def load_lap_data():
    session = fastf1.get_session(2024, "China", "R")
    session.load()
    return session.laps

def run_with_sector_times():
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

    mae = mean_absolute_error(y_test, model.predict(X_test))
    return qualifying_df[["Driver", "PredictedRaceTime (s)"]], mae

def run_without_sector_times():
    qualifying_df = get_qualifying_data()
    laps = load_lap_data()
    laps = laps[["Driver", "LapTime"]].dropna(subset=["LapTime"]).copy()
    laps["LapTime (s)"] = laps["LapTime"].dt.total_seconds()

    merged = qualifying_df.merge(laps, left_on="DriverCode", right_on="Driver")
    if merged.empty:
        return pd.DataFrame(), None

    X = merged[["QualifyingTime (s)"]]
    y = merged["LapTime (s)"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=39)
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=39)
    model.fit(X_train, y_train)

    qualifying_df["PredictedRaceTime (s)"] = model.predict(qualifying_df[["QualifyingTime (s)"]])
    qualifying_df = qualifying_df.sort_values(by="PredictedRaceTime (s)")

    mae = mean_absolute_error(y_test, model.predict(X_test))
    return qualifying_df[["Driver", "PredictedRaceTime (s)"]], mae

# Run selected option
with st.spinner("Running model and generating predictions..."):
    if option == "With Sector Times":
        df, mae = run_with_sector_times()
    else:
        df, mae = run_without_sector_times()

if df.empty:
    st.error("Prediction failed. Possibly missing data.")
else:
    st.success(f"Prediction completed! Mean Absolute Error: {mae:.2f} seconds")
    st.dataframe(df.reset_index(drop=True), use_container_width=True)
    st.bar_chart(df.set_index("Driver"))