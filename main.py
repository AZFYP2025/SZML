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
    # Load the saved scalers
    x_scaler = joblib.load("x_scaler.pkl")
    y_scaler = joblib.load("y_scaler.pkl")
    
    df_summary = fetch_firebase_data()
    if df_summary.empty:
        return {"error": "No data found in Firebase."}

    # Preprocessing (must match training exactly)
    df_summary['date'] = pd.to_datetime(df_summary['date'], format='%m/%d/%Y', errors='coerce')
    df_summary['crimes'] = pd.to_numeric(df_summary['crimes'], errors='coerce')
    df_summary.dropna(subset=['crimes', 'date'], inplace=True)
    
    # Must match training feature engineering exactly
    df_summary['month'] = df_summary['date'].dt.month
    df_summary['week'] = df_summary['date'].dt.isocalendar().week.astype(int)
    df_summary['year'] = df_summary['date'].dt.year
    df_summary['sin_week'] = np.sin(2 * np.pi * df_summary['week'] / 52)
    df_summary['cos_week'] = np.cos(2 * np.pi * df_summary['week'] / 52)
    df_summary['is_festive'] = df_summary['month'].isin([1, 5, 6, 11, 12]).astype(int)
    df_summary['is_monsoon'] = df_summary['month'].isin([10, 11, 12, 1]).astype(int)

    results = {}

    for crime_type in df_summary['type'].dropna().unique():
        df_type = df_summary[df_summary['type'] == crime_type].sort_values("date")
        if df_type.empty:
            continue

        # Must match training exactly
        for lag in [1, 2, 3, 4, 52]:
            df_type[f'lag_{lag}'] = df_type['crimes'].shift(lag)
        df_type['rolling_4wk_mean'] = df_type['crimes'].rolling(4).mean().shift(1)
        df_type['rolling_4wk_std'] = df_type['crimes'].rolling(4).std().shift(1)
        df_type['rolling_52wk_mean'] = df_type['crimes'].rolling(52).mean().shift(1)
        df_type['yoy_change'] = df_type['crimes'] / df_type['lag_52'] - 1
        df_type = df_type.dropna()

        if df_type.empty:
            continue

        # Prepare features (must match training order exactly)
        feature_cols = [
            'year', 'week', 'month', 'sin_week', 'cos_week',
            'is_festive', 'is_monsoon',
            'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_52',
            'rolling_4wk_mean', 'rolling_4wk_std', 'rolling_52wk_mean',
            'yoy_change'
        ]
        
        # Future prediction
        last_row = df_type.iloc[-1:].copy()
        preds = []
        
        for _ in range(12):  # predict 12 weeks ahead
            # Create new row for prediction
            row = last_row.copy()
            row['week'] = row['week'] + 1
            if row['week'].values[0] > 52:
                row['week'] = 1
                row['year'] = row['year'] + 1
            
            # Update temporal features
            row['month'] = (row['date'].dt.month + (row['week'] // 4)).apply(lambda x: x % 12 or 12)
            row['sin_week'] = np.sin(2 * np.pi * row['week'] / 52)
            row['cos_week'] = np.cos(2 * np.pi * row['week'] / 52)
            row['is_festive'] = row['month'].isin([1, 5, 6, 11, 12]).astype(int)
            row['is_monsoon'] = row['month'].isin([10, 11, 12, 1]).astype(int)

            # Scale features using the saved scaler
            X_input = row[feature_cols]
            X_scaled = x_scaler.transform(X_input)
            
            # Predict
            y_scaled_pred = model.predict(X_scaled)
            y_pred = np.expm1(y_scaler.inverse_transform(y_scaled_pred.reshape(-1, 1))[0][0])
            preds.append(y_pred)

            # Update for next iteration (must match training logic)
            new_row = last_row.copy()
            new_row['date'] = last_row['date'] + pd.Timedelta(weeks=1)
            new_row['crimes'] = y_pred
            
            # Update lags
            for lag in [52, 4, 3, 2, 1]:
                if lag == 1:
                    new_row[f'lag_{lag}'] = last_row['crimes'].values[0]
                else:
                    if len(df_type) >= lag-1:
                        new_row[f'lag_{lag}'] = df_type['crimes'].iloc[-lag+1] if lag > 1 else y_pred
            
            # Update rolling stats
            window_4 = df_type['crimes'].iloc[-4:].values if len(df_type) >= 4 else np.array([y_pred]*4)
            new_row['rolling_4wk_mean'] = window_4.mean()
            new_row['rolling_4wk_std'] = window_4.std()
            
            window_52 = df_type['crimes'].iloc[-52:].values if len(df_type) >= 52 else np.array([y_pred]*52)
            new_row['rolling_52wk_mean'] = window_52.mean()
            
            new_row['yoy_change'] = y_pred / new_row['lag_52'].values[0] - 1 if new_row['lag_52'].values[0] > 0 else 0
            
            last_row = new_row
            df_type = pd.concat([df_type, new_row], ignore_index=True)
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
