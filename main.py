from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64
import io
import joblib
import os
import json
from datetime import datetime, timedelta
from firebase_admin import credentials, initialize_app, db

# Firebase init
firebase_json = os.environ.get("FIREBASE_CREDENTIALS")
cred_dict = json.loads(firebase_json)
cred = credentials.Certificate(cred_dict)
initialize_app(cred, {
    'databaseURL': "https://safezone-660a9-default-rtdb.asia-southeast1.firebasedatabase.app/"
})

# FastAPI app
app = FastAPI()

# Load model and scalers
model = joblib.load("model.pkl")
x_scaler = joblib.load("x_scaler.pkl")
y_scaler = joblib.load("y_scaler.pkl")

class CrimeRequest(BaseModel):
    type: str
    category: str

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
                "crimes": v.get("crimes")
            })
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["crimes"] = pd.to_numeric(df["crimes"], errors="coerce")
    df.dropna(inplace=True)
    df['year'] = df['date'].dt.year
    df['week'] = df['date'].dt.isocalendar().week.astype(int)
    df['month'] = df['date'].dt.month
    df['sin_week'] = np.sin(2 * np.pi * df['week'] / 52)
    df['cos_week'] = np.cos(2 * np.pi * df['week'] / 52)
    df['is_festive'] = df['month'].isin([1, 5, 6, 11, 12]).astype(int)
    df['is_monsoon'] = df['month'].isin([10, 11, 12, 1]).astype(int)
    return df

@app.post("/plot_by_crime_type")
async def plot_by_crime_type(request: CrimeRequest):
    crime_type = request.type
    crime_category = request.category
    df = fetch_firebase_data()

    df_type = df[(df["type"] == crime_type) & (df["category"] == crime_category)].copy()
    df_type = df_type.sort_values("date").reset_index(drop=True)

    for lag in [1, 2, 3, 4, 52]:
        df_type[f'lag_{lag}'] = df_type['crimes'].shift(lag)
    df_type['rolling_4wk_mean'] = df_type['crimes'].rolling(4).mean().shift(1)
    df_type['rolling_4wk_std'] = df_type['crimes'].rolling(4).std().shift(1)
    df_type['rolling_52wk_mean'] = df_type['crimes'].rolling(52).mean().shift(1)
    df_type['yoy_change'] = df_type['crimes'] / df_type['lag_52'] - 1
    df_type.dropna(inplace=True)

    last_date = df_type['date'].max()
    future_weeks = []

    for _ in range(104):  # 2 years = 104 weeks
        next_date = last_date + timedelta(weeks=1)
        week = next_date.isocalendar().week
        month = next_date.month
        sin_week = np.sin(2 * np.pi * week / 52)
        cos_week = np.cos(2 * np.pi * week / 52)
        is_festive = int(month in [1, 5, 6, 11, 12])
        is_monsoon = int(month in [10, 11, 12, 1])

        new_row = {
            'date': next_date,
            'year': next_date.year,
            'week': week,
            'month': month,
            'sin_week': sin_week,
            'cos_week': cos_week,
            'is_festive': is_festive,
            'is_monsoon': is_monsoon,
        }

        for lag in [1, 2, 3, 4, 52]:
            new_row[f'lag_{lag}'] = df_type['crimes'].iloc[-lag]
        new_row['rolling_4wk_mean'] = df_type['crimes'].rolling(4).mean().iloc[-1]
        new_row['rolling_4wk_std'] = df_type['crimes'].rolling(4).std().iloc[-1]
        new_row['rolling_52wk_mean'] = df_type['crimes'].rolling(52).mean().iloc[-1]
        new_row['yoy_change'] = new_row['lag_52'] and df_type['crimes'].iloc[-1] / new_row['lag_52'] - 1 or 0

        feature_cols = ['year', 'week', 'month', 'sin_week', 'cos_week',
                        'is_festive', 'is_monsoon',
                        'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_52',
                        'rolling_4wk_mean', 'rolling_4wk_std', 'rolling_52wk_mean',
                        'yoy_change']
        
        x = pd.DataFrame([new_row])[feature_cols]
        x_scaled = x_scaler.transform(x)
        y_pred_scaled = model.predict(x_scaled).reshape(-1, 1)
        y_pred_log = y_scaler.inverse_transform(y_pred_scaled).ravel()
        y_pred = np.expm1(y_pred_log)[0]

        new_row['crimes'] = y_pred
        df_type = pd.concat([df_type, pd.DataFrame([new_row])], ignore_index=True)
        future_weeks.append({'date': next_date, 'crimes': y_pred})
        last_date = next_date

    # Forecasted
    df_future = pd.DataFrame(future_weeks)
    df_future['month'] = df_future['date'].dt.month
    df_future['year'] = df_future['date'].dt.year
    preds_2024 = df_future[df_future['year'] == 2024].groupby('month')['crimes'].sum().reindex(range(1, 13), fill_value=0)
    preds_2025 = df_future[df_future['year'] == 2025].groupby('month')['crimes'].sum().reindex(range(1, 13), fill_value=0)

    # Actual 2023
    actual_2023 = df_type[df_type['year'] == 2023]
    actual_monthly = actual_2023.groupby('month')['crimes'].sum().reindex(range(1, 13), fill_value=0)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(actual_monthly.index, actual_monthly.values, label="2023 Actual", marker='o')
    plt.plot(range(1, 13), preds_2024.values, label="2024 Predicted", marker='x')
    plt.plot(range(1, 13), preds_2025.values, label="2025 Predicted", marker='^')
    plt.title(f"Crime Forecast for '{crime_type}'")
    plt.xlabel("Month")
    plt.ylabel("Crime Count")
    plt.legend()
    plt.grid(True)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()

    return {"type": crime_type, "category": crime_category, "plot": plot_base64}
