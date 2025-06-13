from fastapi import FastAPI, Query
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

# Firebase setup
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

def generate_features(df, category, crime_type):
    df = df.copy()
    df = df[df['category'] == category]
    df = df[df['type'] == crime_type]
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    df['year'] = df['date'].dt.year
    df['week'] = df['date'].dt.isocalendar().week.astype(int)
    df['month'] = df['date'].dt.month
    df['sin_week'] = np.sin(2 * np.pi * df['week'] / 52)
    df['cos_week'] = np.cos(2 * np.pi * df['week'] / 52)
    df['is_festive'] = df['month'].isin([1, 5, 6, 11, 12]).astype(int)
    df['is_monsoon'] = df['month'].isin([10, 11, 12, 1]).astype(int)
    df['crimes'] = df['crimes'].astype(float)

    # Weekly aggregation
    df = df.groupby('date').agg({
        'crimes': 'sum',
        'year': 'first',
        'week': 'first',
        'month': 'first',
        'sin_week': 'first',
        'cos_week': 'first',
        'is_festive': 'first',
        'is_monsoon': 'first'
    }).reset_index()

    # Feature engineering
    for lag in [1, 2, 3, 4, 52]:
        df[f'lag_{lag}'] = df['crimes'].shift(lag)
    df['rolling_4wk_mean'] = df['crimes'].rolling(4).mean().shift(1)
    df['rolling_4wk_std'] = df['crimes'].rolling(4).std().shift(1)
    df['rolling_52wk_mean'] = df['crimes'].rolling(52).mean().shift(1)
    df['yoy_change'] = df['crimes'] / df['lag_52'] - 1

    return df.dropna()

def predict(weekly_df):
    feature_cols = ['year', 'week', 'month', 'sin_week', 'cos_week',
                    'is_festive', 'is_monsoon',
                    'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_52',
                    'rolling_4wk_mean', 'rolling_4wk_std', 'rolling_52wk_mean',
                    'yoy_change']

    X = weekly_df[feature_cols]
    X_scaled = x_scaler.transform(X)

    y_pred_scaled = model.predict(X_scaled)
    y_pred_log = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    y_pred = np.expm1(y_pred_log)

    weekly_df['predicted'] = y_pred
    weekly_df['year'] = weekly_df['date'].dt.year
    weekly_df['month'] = weekly_df['date'].dt.month

    return weekly_df.groupby(['year', 'month'])[['predicted']].sum().reset_index()

def plot_predictions(real_df, pred_df, crime_type):
    plt.figure(figsize=(10, 5))

    for year in [2023, 2024, 2025]:
        if year == 2023:
            data_2023 = real_df[real_df['date'].dt.year == 2023]
            monthly = data_2023.groupby(data_2023['date'].dt.month)['crimes'].sum().reset_index()
            plt.plot(monthly['date'], monthly['crimes'], label='2023')
        else:
            data_year = pred_df[pred_df['year'] == year]
            plt.plot(data_year['month'], data_year['predicted'], label=str(year))

    plt.title(f"Monthly Crime Predictions for {crime_type} (2023–2025)")
    plt.xlabel("Month")
    plt.ylabel("Crimes")
    plt.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

@app.get("/plot_by_crime_type")
def plot_by_crime_type(category: str = Query(...), type: str = Query(...)):
    df = fetch_firebase_data()
    if df.empty:
        return {"error": "No Firebase data found"}

    df = df[df['date'] < "2024"]
    df['date'] = pd.to_datetime(df['date'])

    weekly_df = generate_features(df, category, type)
    if weekly_df.empty:
        return {"error": "Not enough data after feature engineering"}

    pred_df = predict(weekly_df)
    image_base64 = plot_predictions(weekly_df, pred_df, type)

    return {"image_base64": image_base64}
