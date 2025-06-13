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
    return {"message": "Hello World"}

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

@app.get("/plot_by_crime_type")
def plot_by_crime_type():
    df_summary = fetch_firebase_data()

    if df_summary.empty:
        return {"error": "No data found in Firebase."}

    df_summary['date'] = pd.to_datetime(df_summary['date'], format='%m/%d/%Y', errors='coerce')
    df_summary['year'] = df_summary['date'].dt.year
    df_summary = df_summary.dropna(subset=['date'])
    df_summary['week'] = df_summary['date'].dt.isocalendar().week.astype(int)
    df_summary['month'] = df_summary['date'].dt.month
    df_summary['sin_week'] = np.sin(2 * np.pi * df_summary['week'] / 52)
    df_summary['cos_week'] = np.cos(2 * np.pi * df_summary['week'] / 52)
    df_summary['is_festive'] = df_summary['month'].isin([1, 5, 6, 11, 12]).astype(int)
    df_summary['is_monsoon'] = df_summary['month'].isin([10, 11, 12, 1]).astype(int)

    results = {}

    for crime_type in df_summary['type'].dropna().unique():
        df_type = df_summary[df_summary['type'] == crime_type].copy()
        df_type = df_type.sort_values('date')

        # Feature engineering
        for lag in [1, 2, 3, 4, 52]:
            df_type[f'lag_{lag}'] = df_type['crimes'].shift(lag)
        df_type['rolling_4wk_mean'] = df_type['crimes'].rolling(4).mean().shift(1)
        df_type['rolling_4wk_std'] = df_type['crimes'].rolling(4).std().shift(1)
        df_type['rolling_52wk_mean'] = df_type['crimes'].rolling(52).mean().shift(1)
        df_type['yoy_change'] = df_type['crimes'] / df_type['lag_52'] - 1
        df_type = df_type.dropna()

        if df_type.empty:
            continue

        feature_cols = ['year', 'week', 'month', 'sin_week', 'cos_week',
                        'is_festive', 'is_monsoon',
                        'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_52',
                        'rolling_4wk_mean', 'rolling_4wk_std', 'rolling_52wk_mean',
                        'yoy_change']

        X = df_type[feature_cols]
        X_scaled = x_scaler.transform(X)

        # Forecast for 2024 and 2025 (mock last row reuse for simplicity)
        last_row = df_type.iloc[-1:]
        future_rows = []
        for year in [2024, 2025]:
            for month in range(1, 13):
                row = last_row.copy()
                row['year'] = year
                row['month'] = month
                row['week'] = 1  # default week for prediction
                row['sin_week'] = np.sin(2 * np.pi * row['week'] / 52)
                row['cos_week'] = np.cos(2 * np.pi * row['week'] / 52)
                row['is_festive'] = int(month in [1, 5, 6, 11, 12])
                row['is_monsoon'] = int(month in [10, 11, 12, 1])
                future_rows.append(row[feature_cols])

        df_future = pd.concat(future_rows, ignore_index=True)
        X_future_scaled = x_scaler.transform(df_future)

        preds_scaled = model.predict(X_future_scaled)
        preds_log = y_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).ravel()
        preds = np.expm1(preds_log)

        # Plot
        plt.figure(figsize=(10, 5))
        df_actual_grouped = df_type[df_type['year'] == 2023].groupby('month')['crimes'].sum()
        plt.plot(df_actual_grouped.index, df_actual_grouped.values, label="2023 Actual", marker='o')
        plt.plot(range(1, 13), preds[:12], label="2024 Predicted", marker='x')
        plt.plot(range(1, 13), preds[12:], label="2025 Predicted", marker='^')
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
