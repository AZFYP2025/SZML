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
from datetime import datetime, timedelta

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

# Feature columns used during training
feature_cols = ['year', 'week', 'month', 'sin_week', 'cos_week',
                'is_festive', 'is_monsoon',
                'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_52',
                'rolling_4wk_mean', 'rolling_4wk_std', 'rolling_52wk_mean',
                'yoy_change']

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Hello from SafeZone API"}

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
    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    return df.dropna(subset=['date'])

def prepare_features(df_type):
    df = df_type.copy()
    df = df[['date', 'crimes']].sort_values('date')
    df['crimes'] = df['crimes'].astype(float)

    future_weeks = pd.date_range(df['date'].max() + timedelta(days=7), periods=104, freq='W')
    future_df = pd.DataFrame({'date': future_weeks})

    df = pd.concat([df, future_df], ignore_index=True)
    df['year'] = df['date'].dt.year
    df['week'] = df['date'].dt.isocalendar().week.astype(int)
    df['month'] = df['date'].dt.month
    df['sin_week'] = np.sin(2 * np.pi * df['week'] / 52)
    df['cos_week'] = np.cos(2 * np.pi * df['week'] / 52)
    df['is_festive'] = df['month'].isin([1, 5, 6, 11, 12]).astype(int)
    df['is_monsoon'] = df['month'].isin([10, 11, 12, 1]).astype(int)

    for lag in [1, 2, 3, 4, 52]:
        df[f'lag_{lag}'] = df['crimes'].shift(lag)
    df['rolling_4wk_mean'] = df['crimes'].rolling(4).mean().shift(1)
    df['rolling_4wk_std'] = df['crimes'].rolling(4).std().shift(1)
    df['rolling_52wk_mean'] = df['crimes'].rolling(52).mean().shift(1)
    df['yoy_change'] = df['crimes'] / df['lag_52'] - 1

    df = df.dropna()
    df = df[df['year'].isin([2024, 2025])]
    return df

@app.get("/plot_by_crime_type")
def plot_by_crime_type():
    df_summary = fetch_firebase_data()
    if df_summary.empty:
        return {"error": "No data found in Firebase."}

    df_summary['year'] = df_summary['date'].dt.year
    df_2023 = df_summary[df_summary['year'] == 2023]

    results = {}

    for crime_type in df_2023['type'].dropna().unique():
        df_type = df_2023[df_2023['type'] == crime_type]
        if df_type.empty:
            continue

        df_feat = prepare_features(df_type)
        if df_feat.empty:
            continue

        X = x_scaler.transform(df_feat[feature_cols])
        y_pred_scaled = model.predict(X)
        y_pred_log = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        y_pred = np.expm1(y_pred_log)

        df_feat['prediction'] = y_pred
        df_feat['month'] = df_feat['date'].dt.month
        df_feat['year'] = df_feat['date'].dt.year
        monthly_pred = df_feat.groupby(['year', 'month'])['prediction'].sum().reset_index()

        actual_grouped = df_type.groupby(df_type['date'].dt.month)['crimes'].sum()

        plt.figure(figsize=(10, 5))
        plt.plot(actual_grouped.index, actual_grouped.values, label="2023 Actual", marker='o')
        for yr in [2024, 2025]:
            ydata = monthly_pred[monthly_pred['year'] == yr]
            plt.plot(ydata['month'], ydata['prediction'], label=f"{yr} Predicted", marker='x')

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
