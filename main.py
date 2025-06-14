from fastapi import FastAPI
import pandas as pd
import numpy as np
import joblib
import os
import json
import matplotlib.pyplot as plt
import io
import base64
from firebase_admin import credentials, initialize_app, db
from pathlib import Path

# Firebase setup
firebase_json = os.environ.get("FIREBASE_CREDENTIALS")
cred_dict = json.loads(firebase_json)
cred = credentials.Certificate(cred_dict)
initialize_app(cred, {
    'databaseURL': "https://safezone-660a9-default-rtdb.asia-southeast1.firebasedatabase.app/"
})

app = FastAPI()

CATEGORY = "assault"
TYPE = "murder"

@app.get("/")
def root():
    return {"message": "Forecasting assault - murder only"}

def safe_name(s): return s.strip().lower().replace(" ", "_")

def load_model_and_scalers(category, typ):
    base = f"models/{safe_name(category)}__{safe_name(typ)}__"
    return (
        joblib.load(base + "model.pkl"),
        joblib.load(base + "x_scaler.pkl"),
        joblib.load(base + "y_scaler.pkl"),
    )

def fetch_data(category, typ):
    ref = db.reference("crime_data")
    raw = ref.get()
    rows = []
    for v in raw.values():
        if isinstance(v, dict) and v.get("category") == category and v.get("type") == typ:
            rows.append({
                "date": v["date"],
                "crimes": v["crimes"]
            })
    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['date'])
    df['crimes'] = pd.to_numeric(df['crimes'], errors="coerce")
    return df.dropna().sort_values('date').reset_index(drop=True)

def engineer(df):
    df['year'] = df['date'].dt.year
    df['week'] = df['date'].dt.isocalendar().week.astype(int)
    df['month'] = df['date'].dt.month
    df['sin_week'] = np.sin(2 * np.pi * df['week'] / 52)
    df['cos_week'] = np.cos(2 * np.pi * df['week'] / 52)
    df['is_festive'] = df['month'].isin([1, 5, 6, 11, 12]).astype(int)
    df['is_monsoon'] = df['month'].isin([10, 11, 12, 1]).astype(int)
    for lag in [1,2,3,4,52]:
        df[f'lag_{lag}'] = df['crimes'].shift(lag)
    df['rolling_4wk_mean'] = df['crimes'].rolling(4).mean().shift(1)
    df['rolling_4wk_std']  = df['crimes'].rolling(4).std().shift(1)
    df['rolling_52wk_mean'] = df['crimes'].rolling(52).mean().shift(1)
    df['yoy_change'] = df['crimes'] / df['lag_52'] - 1
    return df

@app.get("/predict_and_plot")
def forecast_and_plot():
    df = fetch_data(CATEGORY, TYPE)
    df = engineer(df)

    model, x_scaler, y_scaler = load_model_and_scalers(CATEGORY, TYPE)
    feature_cols = [
        'year', 'week', 'month', 'sin_week', 'cos_week',
        'is_festive', 'is_monsoon',
        'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_52',
        'rolling_4wk_mean', 'rolling_4wk_std', 'rolling_52wk_mean',
        'yoy_change'
    ]

    forecast_weeks = 104
    history = df.copy()

    for _ in range(forecast_weeks):
        last = history.iloc[-1]
        next_date = last['date'] + pd.Timedelta(weeks=1)
        new_row = pd.DataFrame([{
            'date': next_date,
            'crimes': np.nan
        }])
        history = pd.concat([history, new_row], ignore_index=True)
        history = engineer(history)

        latest = history.iloc[[-1]]
        if latest[feature_cols].isnull().any(axis=1).values[0]:
            break

        X = x_scaler.transform(latest[feature_cols])
        y_scaled = model.predict(X)
        y_log = y_scaler.inverse_transform(y_scaled.reshape(-1, 1)).ravel()
        y_pred = np.expm1(y_log)[0]
        history.at[history.index[-1], 'crimes'] = y_pred
        history.at[history.index[-1], 'predicted'] = True

    history['predicted'] = history['predicted'].fillna(False)
    history['year'] = history['date'].dt.year
    history['month'] = history['date'].dt.month

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))

    actual = history[(history['year'] == 2023) & (~history['predicted'])]
    future = history[(history['year'] >= 2024) & (history['predicted'])]

    if not actual.empty:
        ax.plot(
            actual.groupby("month")["crimes"].mean(),
            label="2023 Actual", marker='o'
        )
    for y in [2024, 2025]:
        fy = future[future['year'] == y]
        if not fy.empty:
            ax.plot(
                fy.groupby("month")["crimes"].mean(),
                label=f"{y} Forecast", linestyle="--", marker='o'
            )

    ax.set_title("Assault - Murder Forecast")
    ax.set_xlabel("Month")
    ax.set_ylabel("Crimes")
    ax.legend()
    ax.grid(True)
    ax.set_xticks(range(1, 13))

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    return {
        "category": CATEGORY,
        "type": TYPE,
        "image_base64": encoded
    }
