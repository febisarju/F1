# RaceCraft 2025: The F1 Prediction Engine

A machine learning project to predict Formula 1 race times using qualifying and sector data from the 2024 Chinese Grand Prix, featuring an interactive Streamlit dashboard for visualizing predictions.

## Features:

- **Machine Learning Model**: Uses Gradient Boosting Regressor to predict race times based on qualifying times and sector performance.
- **Data Integration**: Pulls real-time F1 data using the FastF1 API.
- **Interactive Dashboard**: Streamlit app allows users to toggle between two prediction modes (with or without sector times) and view results in a table and bar chart.
- **Performance Evaluation**: Reports Mean Absolute Error (MAE) to assess model accuracy.

## Tools and Technologies Used:

- **Python**: fastf1 - F1 data API , pandas - Data manipulation, scikit-learn - Machine learning, numpy - Numerical operations.
- **Streamlit**: Web app framework

## Results & Insights: 

- Model Performance:       
With Sector Times(Uses qualifying times and average sector times for more accurate predictions): MAE ~0.XX seconds (varies with data split).
Without Sector Times(Uses only qualifying times for a simpler model): MAE ~0.XX seconds.
The bar chart visualizes predicted race times, sorted from fastest to slowest.

## Limitations:

- Hardcoded 2025 qualifying times are used as placeholders, not real data.
- Predictions are based on a single race (2024 Chinese Grand Prix), limiting generalizability.

## Future Work:

- Integrate dynamic qualifying data from FastF1 or external sources.
- Support multiple races or full seasons for broader analysis.

## Contact:     

For any questions or collaboration, feel free to reach out!                                 
Github- https://github.com/febisarju
