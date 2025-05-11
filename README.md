# RaceCraft 2025: The F1 Prediction Engine
F1 2025 Race Time Predictor
A machine learning project to predict Formula 1 race times using qualifying and sector data from the 2024 Chinese Grand Prix, featuring an interactive Streamlit dashboard for visualizing predictions.
Features

Machine Learning Model: Uses Gradient Boosting Regressor to predict race times based on qualifying times and sector performance.
Data Integration: Pulls real-time F1 data using the FastF1 API.
Interactive Dashboard: Streamlit app allows users to toggle between two prediction modes (with or without sector times) and view results in a table and bar chart.
Performance Evaluation: Reports Mean Absolute Error (MAE) to assess model accuracy.

Demo
Try the live app: F1 Race Predictor (Note: Replace with actual URL after deployment)
Installation

Clone the repository:git clone https://github.com/<your-username>/F1-Race-Predictor
cd F1-Race-Predictor


Create a virtual environment (optional but recommended):python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install dependencies:pip install -r requirements.txt


Run the Streamlit app:streamlit run app.py



Dependencies
Key libraries (full list in requirements.txt):

fastf1 - F1 data API
pandas - Data manipulation
scikit-learn - Machine learning
streamlit - Web app framework
numpy - Numerical operations

Usage

Launch the Streamlit app:streamlit run app.py


Open your browser to http://localhost:8501.
Select a prediction mode:
With Sector Times: Uses qualifying times and average sector times for more accurate predictions.
Without Sector Times: Uses only qualifying times for a simpler model.


View the predicted race times in a table and bar chart, along with the model's Mean Absolute Error (MAE).

Data Source

Race and sector data are sourced from the FastF1 API for the 2024 Chinese Grand Prix.
Qualifying times for 2025 are hardcoded as sample data for demonstration purposes.

Results

Model Performance:
With Sector Times: MAE ~0.XX seconds (varies with data split)
Without Sector Times: MAE ~0.XX seconds


Top Predicted Drivers: Max Verstappen, Lando Norris, Oscar Piastri (based on sample data).
The bar chart visualizes predicted race times, sorted from fastest to slowest.

Limitations

Hardcoded 2025 qualifying times are used as placeholders, not real data.
Predictions are based on a single race (2024 Chinese Grand Prix), limiting generalizability.
Missing sector data is filled with zeros, which may affect prediction accuracy.
Basic error handling; empty data merges may cause failures.

Future Work

Integrate dynamic qualifying data from FastF1 or external sources.
Support multiple races or full seasons for broader analysis.
Implement hyperparameter tuning (e.g., GridSearchCV) and compare additional models (e.g., Random Forest, XGBoost).
Enhance the Streamlit app with custom inputs (e.g., select race or input times) and advanced visualizations (e.g., scatter plots).
Add unit tests for data loading and model predictions.

License
MIT License
Contact

Email: your-email@example.com
LinkedIn: Your LinkedIn Profile
GitHub: Your GitHub Profile


Built with passion for Formula 1 and data science!