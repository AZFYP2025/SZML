from fastapi import FastAPI, Response
from fastapi.responses import FileResponse
import joblib
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings
import os
import matplotlib.pyplot as plt
import json
import matplotlib.dates as mdates
from firebase_admin import credentials, initialize_app

warnings.filterwarnings("ignore")

# Firebase init
firebase_json = os.environ.get("FIREBASE_CREDENTIALS")
cred_dict = json.loads(firebase_json)
cred = credentials.Certificate(cred_dict)
initialize_app(cred, {
    'databaseURL': "https://safezone-660a9-default-rtdb.asia-southeast1.firebasedatabase.app/"
})

# Create output directory if it doesn't exist
# models directory not needed for prediction only
os.makedirs("plots", exist_ok=True)

app = FastAPI()

@app.get("/forecast_data")
def generate_forecast_plot(category: str, type_: str):
    safe_cat = category.replace(" ", "_").lower()
    safe_typ = type_.replace(" ", "_").lower()

    try:
        model = joblib.load(f"models/{safe_cat}__{safe_typ}__model.pkl")
        x_scaler = joblib.load(f"models/{safe_cat}__{safe_typ}__x_scaler.pkl")
        y_scaler = joblib.load(f"models/{safe_cat}__{safe_typ}__y_scaler.pkl")
    except FileNotFoundError:
        return {"error": "Model or scalers not found."}

    from firebase_admin import db

    ref = db.reference("crime_data")
    snapshots = ref.get()
    df = pd.DataFrame([v for v in snapshots.values() if v.get("source") == "synth"])
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['week'] = df['date'].dt.isocalendar().week.astype(int)
    df['month'] = df['date'].dt.month
    df['sin_week'] = np.sin(2 * np.pi * df['week'] / 52)
    df['cos_week'] = np.cos(2 * np.pi * df['week'] / 52)
    df['is_festive'] = df['month'].isin([1, 5, 6, 11, 12]).astype(int)
    df['is_monsoon'] = df['month'].isin([10, 11, 12, 1]).astype(int)

    data = df[(df['category'] == category) & (df['type'] == type_)].copy()
    data = data.sort_values('date').reset_index(drop=True)

    for lag in [1, 2, 3, 4, 52]:
        data[f'lag_{lag}'] = data['crimes'].shift(lag)
    data['rolling_4wk_mean'] = data['crimes'].rolling(4).mean().shift(1)
    data['rolling_4wk_std'] = data['crimes'].rolling(4).std().shift(1)
    data['rolling_52wk_mean'] = data['crimes'].rolling(52).mean().shift(1)
    data['yoy_change'] = data['crimes'] / data['lag_52'] - 1
    data = data.dropna().reset_index(drop=True)

    forecast_weeks = pd.date_range(start="2024-01-01", end="2024-12-31", freq="W-MON")
    latest = data.iloc[-52:].copy()
    predictions = []

    for date in forecast_weeks:
        year = date.year
        week = date.isocalendar().week
        month = date.month
        sin_week = np.sin(2 * np.pi * week / 52)
        cos_week = np.cos(2 * np.pi * week / 52)
        is_festive = int(month in [1, 5, 6, 11, 12])
        is_monsoon = int(month in [10, 11, 12, 1])

        row = {
            'year': year,
            'week': week,
            'month': month,
            'sin_week': sin_week,
            'cos_week': cos_week,
            'is_festive': is_festive,
            'is_monsoon': is_monsoon
        }

        for lag in [1, 2, 3, 4, 52]:
            row[f'lag_{lag}'] = latest['crimes'].iloc[-lag]

        row['rolling_4wk_mean'] = latest['crimes'].iloc[-4:].mean()
        row['rolling_4wk_std'] = latest['crimes'].iloc[-4:].std()
        row['rolling_52wk_mean'] = latest['crimes'].mean()
        row['yoy_change'] = row['lag_52'] and ((row['lag_1'] / row['lag_52']) - 1) or 0

        X_pred = pd.DataFrame([row])
        X_scaled = x_scaler.transform(X_pred)
        y_pred_scaled = model.predict(X_scaled)
        y_pred_log = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        y_pred = np.expm1(y_pred_log[0])

        row['crimes'] = y_pred
        row['date'] = date
        latest = pd.concat([latest, pd.DataFrame([row])], ignore_index=True)
        predictions.append((date, y_pred))

    actual_2023 = data[data['year'] == 2023][['date', 'crimes']]
    pred_df = pd.DataFrame(predictions, columns=['date', 'predicted_crimes'])

    # Format response
    response = {
        "category": category,
        "type": type_,
        "actual_2023": [
            {"month": int(row["month"]), "date": row["date"].strftime("%Y-%m-%d"), "crimes": row["crimes"]}
            for _, row in actual_2023.iterrows()
        ],
        "predicted_2024": [
            {"month": int(row["month"]), "date": row["date"].strftime("%Y-%m-%d"), "crimes": row["predicted_crimes"]}
            for _, row in pred_df.iterrows()
        ]
    }

    return response
