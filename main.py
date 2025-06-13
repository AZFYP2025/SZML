from fastapi import FastAPI
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64
import io
import joblib
from firebase_admin import credentials, initialize_app, db
import os
import json
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
def root():
    return {"message": "Hello from SafeZone"}

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
            })
    return pd.DataFrame(rows)

@app.get("/plot_by_crime_type")
def plot_by_crime_type():
    df = fetch_firebase_data()
    print("Raw data:", df.shape)

    if df.empty:
        return {"error": "No data found"}

    # Parse and preprocess
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df['year'] = df['date'].dt.year
    df['week'] = df['date'].dt.isocalendar().week.astype(int)
    df['month'] = df['date'].dt.month
    df['sin_week'] = np.sin(2 * np.pi * df['week'] / 52)
    df['cos_week'] = np.cos(2 * np.pi * df['week'] / 52)
    df['is_festive'] = df['month'].isin([1, 5, 6, 11, 12]).astype(int)
    df['is_monsoon'] = df['month'].isin([10, 11, 12, 1]).astype(int)

    df = df[df['year'] == 2023]
    print("Filtered 2023 data:", df.shape)

    if df.empty:
        return {"error": "No data from 2023"}

    results = {}
    for crime_type in df['type'].unique():
        df_type = df[df['type'] == crime_type].sort_values('date')
        if df_type.empty:
            continue

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

        # Prepare historical data for plotting
        actual_2023 = df_type.groupby('month')['crimes'].sum()

        # Generate future data
        base_row = df_type.iloc[-1]
        future = []
        for year in [2024, 2025]:
            for month in range(1, 13):
                week = (month - 1) * 4 + 1  # rough estimate
                sin_week = np.sin(2 * np.pi * week / 52)
                cos_week = np.cos(2 * np.pi * week / 52)
                is_festive = int(month in [1, 5, 6, 11, 12])
                is_monsoon = int(month in [10, 11, 12, 1])

                row = {
                    "year": year,
                    "week": week,
                    "month": month,
                    "sin_week": sin_week,
                    "cos_week": cos_week,
                    "is_festive": is_festive,
                    "is_monsoon": is_monsoon,
                    "lag_1": base_row['crimes'],
                    "lag_2": base_row['crimes'],
                    "lag_3": base_row['crimes'],
                    "lag_4": base_row['crimes'],
                    "lag_52": base_row['crimes'],
                    "rolling_4wk_mean": base_row['crimes'],
                    "rolling_4wk_std": 0,
                    "rolling_52wk_mean": base_row['crimes'],
                    "yoy_change": 0
                }
                future.append(row)

        df_future = pd.DataFrame(future)

        # Predict
        feature_cols = df_future.columns
        X_future_scaled = x_scaler.transform(df_future[feature_cols])
        y_scaled_pred = model.predict(X_future_scaled)
        y_log_pred = y_scaler.inverse_transform(y_scaled_pred.reshape(-1, 1)).ravel()
        y_pred = np.expm1(y_log_pred)

        # Plot
        plt.figure(figsize=(10, 5))
        plt.plot(actual_2023.index, actual_2023.values, label="2023 Actual", marker='o')
        plt.plot(range(1, 13), y_pred[:12], label="2024 Predicted", marker='x')
        plt.plot(range(1, 13), y_pred[12:], label="2025 Predicted", marker='^')
        plt.title(f"Forecast for {crime_type}")
        plt.xlabel("Month")
        plt.ylabel("Crime Count")
        plt.grid(True)
        plt.legend()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("utf-8")
        results[crime_type] = img_b64

    print("Returning results:", len(results))
    return results
