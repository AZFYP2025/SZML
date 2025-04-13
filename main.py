from typing import Optional

from fastapi import FastAPI
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64
import io
import joblib
from fastapi import FastAPI
from firebase_admin import credentials, initialize_app, db
from sklearn.preprocessing import OneHotEncoder
import json
import os

firebase_json = os.environ.get("FIREBASE_CREDENTIALS")
cred_dict = json.loads(firebase_json)
cred = credentials.Certificate(cred_dict)

initialize_app(cred, {
    'databaseURL': "https://safezone-660a9-default-rtdb.asia-southeast1.firebasedatabase.app/"
})


# Load model and encoder
model = joblib.load("model.pkl")
encoder: OneHotEncoder = joblib.load("encoder.pkl")

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

# Fetch and aggregate data from Firebase
def fetch_and_aggregate_firebase(path: str) -> pd.DataFrame:
    # Connect to the Firebase path
    ref = db.reference(path)
    raw_data = ref.get()

    # If no data, return empty DataFrame
    if not raw_data:
        return pd.DataFrame()

    # Convert data to list of records
    records = []
    
    if isinstance(raw_data, dict):
        # Handle dictionary format (likely Firebase's default)
        for key, value in raw_data.items():
            if isinstance(value, dict):
                records.append(value)
            elif isinstance(value, str):
                # Handle case where value might be a JSON string
                try:
                    parsed = json.loads(value)
                    if isinstance(parsed, dict):
                        records.append(parsed)
                except json.JSONDecodeError:
                    pass
    elif isinstance(raw_data, list):
        # Handle list format
        for item in raw_data:
            if isinstance(item, dict):
                records.append(item)
            elif isinstance(item, str):
                try:
                    parsed = json.loads(item)
                    if isinstance(parsed, dict):
                        records.append(parsed)
                except json.JSONDecodeError:
                    pass

    if not records:
        return pd.DataFrame()

    # Convert to DataFrame
    try:
        df = pd.DataFrame(records)
    except ValueError as e:
        print(f"Error converting to DataFrame: {e}")
        return pd.DataFrame()

    # Check if required columns exist
    required_columns = ['date', 'category', 'type']
    if not all(col in df.columns for col in required_columns):
        return pd.DataFrame()

    # Process the data
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(subset=['date', 'category', 'type'], inplace=True)

    if df.empty:
        return df

    # Extract month and year from date
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year

    # Group by category, type, month, year to count crimes
    summary = df.groupby(['category', 'type', 'month', 'year']).size().reset_index(name='crimes')

    return summary


@app.get("/predict_firebase_summary")
def predict_firebase_summary():
    df_summary = fetch_and_aggregate_firebase("crime_data")
    if df_summary.empty:
        return {"error": "No data found to summarize"}

    X_cat = df_summary[['category', 'type']]
    X_num = df_summary[['month', 'year']]

    try:
        X_encoded = encoder.transform(X_cat)
    except:
        return {"error": "OneHotEncoder categories mismatch – ensure encoder.pkl matches Firebase data"}

    X_input = np.hstack([X_encoded, X_num])
    predictions = model.predict(X_input)
    df_summary['predicted_crimes'] = predictions

    return df_summary.to_dict(orient="records")

@app.get("/plot_by_crime_type")
def plot_by_crime_type():
    df_summary = fetch_and_aggregate_firebase("crime_data")
    if df_summary.empty:
        return {"error": "No data found to summarize"}

    # Keep only synthetic-source data for actuals
    df_synth = df_summary[df_summary['source'] == 'synth'].copy()

    # Extract month and year from date if not already done
    if 'month' not in df_synth.columns or 'year' not in df_synth.columns:
        df_synth['date'] = pd.to_datetime(df_synth['date'])
        df_synth['month'] = df_synth['date'].dt.month
        df_synth['year'] = df_synth['date'].dt.year

    # Only keep 2023 actual synth data
    df_2023 = df_synth[df_synth['year'] == 2023]

    results = {}

    for crime_type in df_2023['type'].unique():
        df_type = df_2023[df_2023['type'] == crime_type]

        # Prepare prediction input for 2024 and 2025
        input_rows = []
        for year in [2024, 2025]:
            for month in range(1, 13):
                input_rows.append({
                    'category': df_type['category'].iloc[0],  # assume consistent category
                    'type': crime_type,
                    'month': month,
                    'year': year
                })

        df_future = pd.DataFrame(input_rows)

        # Encode input
        X_cat = df_future[['category', 'type']]
        X_num = df_future[['month', 'year']]
        X_encoded = encoder.transform(X_cat)
        X_input = np.hstack([X_encoded, X_num])

        # Predict
        preds = model.predict(X_input)

        # Plot
        plt.figure(figsize=(10, 5))

        # Actual 2023 crimes per month using 'crimes' column
        df_actual_grouped = df_type.groupby('month')['crimes'].sum()
        plt.plot(df_actual_grouped.index, df_actual_grouped.values, label="2023 Actual", marker='o')

        # Predictions for 2024 and 2025
        plt.plot(range(1, 13), preds[:12], label="2024 Predicted", marker='x')
        plt.plot(range(1, 13), preds[12:], label="2025 Predicted", marker='^')

        plt.title(f"Crime Forecast for '{crime_type}'")
        plt.xlabel("Month")
        plt.ylabel("Crime Count")
        plt.legend()
        plt.grid(True)

        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        results[crime_type] = base64.b64encode(buf.read()).decode("utf-8")
        plt.close()

    return results

