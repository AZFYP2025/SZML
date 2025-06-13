from typing import Optional
from fastapi import FastAPI
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64
import io
import joblib
from firebase_admin import credentials, initialize_app, db
import json
import os
from datetime import datetime

# Initialize Firebase
firebase_json = os.environ.get("FIREBASE_CREDENTIALS")
cred_dict = json.loads(firebase_json)
cred = credentials.Certificate(cred_dict)
initialize_app(cred, {
    'databaseURL': "https://safezone-660a9-default-rtdb.asia-southeast1.firebasedatabase.app/"
})

# Load model and scalers
model = joblib.load("model.pkl")
x_scaler = joblib.load("x_scaler.pkl")
y_scaler = joblib.load("y_scaler.pkl")

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "XGBoost Forecast API Active"}

def fetch_firebase_data():
    ref = db.reference("crime_data")
    data = ref.get()
    if not data:
        return pd.DataFrame()

    rows = []
    for _, v in data.items():
        if isinstance(v, dict) and v.get("source") == "synth":
            rows.append({
                "category": v.get("category"),
                "type": v.get("type"),
                "date": v.get("date"),
                "crimes": v.get("crimes"),
                "source": v.get("source")
            })
    return pd.DataFrame(rows)

def extract_features(df):
    df['sin_week'] = np.sin(2 * np.pi * df['week'] / 52)
    df['cos_week'] = np.cos(2 * np.pi * df['week'] / 52)
    df['is_festive'] = df['month'].isin([1, 5, 6, 11, 12]).astype(int)
    df['is_monsoon'] = df['month'].isin([10, 11, 12, 1]).astype(int)
    return df

@app.get("/plot_by_crime_type")
def plot_by_crime_type():
    df_summary = fetch_firebase_data()

    if df_summary.empty:
        return {"error": "No data found in Firebase."}

    df_summary['date'] = pd.to_datetime(df_summary['date'], format='%m/%d/%Y', errors='coerce')
    df_summary['month'] = df_summary['date'].dt.month
    df_summary['year'] = df_summary['date'].dt.year
    df_summary['week'] = df_summary['date'].dt.isocalendar().week.astype(int)

    df_2023 = df_summary[df_summary['year'] == 2023]

    results = {}

    for crime_type in df_2023['type'].dropna().unique():
        df_type = df_2023[df_2023['type'] == crime_type]

        if df_type.empty:
            continue

        # Generate future rows
        input_rows = []
        for year in [2024, 2025]:
            for month in range(1, 13):
                dt = datetime(year, month, 1)
                week = dt.isocalendar().week
                input_rows.append({
                    'year': year,
                    'month': month,
                    'week': week
                })

        df_future = pd.DataFrame(input_rows)
        df_future = extract_features(df_future)
        X_input = df_future[['year', 'week', 'month', 'sin_week', 'cos_week', 'is_festive', 'is_monsoon']]
        X_scaled = x_scaler.transform(X_input)

        y_scaled_pred = model.predict(X_scaled)
        y_log_pred = y_scaler.inverse_transform(y_scaled_pred.reshape(-1, 1)).ravel()
        y_pred = np.expm1(y_log_pred)

        # Plotting
        plt.figure(figsize=(10, 5))
        df_actual_grouped = df_type.groupby('month')['crimes'].sum()
        plt.plot(df_actual_grouped.index, df_actual_grouped.values, label="2023 Actual", marker='o')
        plt.plot(range(1, 13), y_pred[:12], label="2024 Predicted", marker='x')
        plt.plot(range(1, 13), y_pred[12:], label="2025 Predicted", marker='^')
        plt.title(f"Crime Forecast for '{crime_type}'")
        plt.xlabel("Month")
        plt.ylabel("Crime Count")
        plt.legend()
        plt.grid(True)

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        results[crime_type] = base64.b64encode(buf.read()).decode("utf-8")
        plt.close()

    return results
