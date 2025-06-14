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

# Firebase init
firebase_json = os.environ.get("FIREBASE_CREDENTIALS")
cred_dict = json.loads(firebase_json)
cred = credentials.Certificate(cred_dict)
initialize_app(cred, {
    'databaseURL': "https://safezone-660a9-default-rtdb.asia-southeast1.firebasedatabase.app/"
})

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

def safe_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_")

def get_model_and_scalers(category: str, typ: str):
    safe_cat = safe_name(category)
    safe_typ = safe_name(typ)
    model_path = Path(f"models/{safe_cat}__{safe_typ}__model.pkl")
    x_scaler_path = Path(f"models/{safe_cat}__{safe_typ}__x_scaler.pkl")
    y_scaler_path = Path(f"models/{safe_cat}__{safe_typ}__y_scaler.pkl")
    if not model_path.exists() or not x_scaler_path.exists() or not y_scaler_path.exists():
        raise FileNotFoundError(f"Missing model/scalers for: {category} - {typ}")
    return (
        joblib.load(model_path),
        joblib.load(x_scaler_path),
        joblib.load(y_scaler_path)
    )

def fetch_data(category: str, typ: str):
    ref = db.reference("crime_data")
    raw = ref.get()
    rows = []
    for v in raw.values():
        if isinstance(v, dict) and v.get("source") == "synth" and v.get("category") == category and v.get("type") == typ:
            rows.append({
                "category": v["category"],
                "type": v["type"],
                "date": v["date"],
                "crimes": v["crimes"]
            })
    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['date'])
    df['crimes'] = pd.to_numeric(df['crimes'], errors="coerce")
    return df.dropna().sort_values("date").reset_index(drop=True)

def engineer(df):
    df = df.copy()
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
    return df

@app.get("/predict_and_plot")
async def predict_and_plot():
    results = []

    category, typ = "property", "theft"

    try:
        model, x_scaler, y_scaler = get_model_and_scalers(category, typ)
        df = fetch_data(category, typ)
        df = engineer(df)
        df['forecast'] = False  # mark all as historical

        feature_cols = [
            'year', 'week', 'month', 'sin_week', 'cos_week',
            'is_festive', 'is_monsoon',
            'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_52',
            'rolling_4wk_mean', 'rolling_4wk_std', 'rolling_52wk_mean',
            'yoy_change'
        ]

        forecast_weeks = 52
        history = df.copy()

        for _ in range(forecast_weeks):
            next_date = history['date'].max() + pd.Timedelta(weeks=1)
            new_row = pd.DataFrame([{'date': next_date, 'crimes': np.nan}])
            temp = pd.concat([history, new_row], ignore_index=True)
            temp = engineer(temp)

            latest = temp.iloc[[-1]].copy()
            if latest[feature_cols].isnull().any(axis=1).values[0]:
                continue  # skip but continue loop

            X = x_scaler.transform(latest[feature_cols])
            y_scaled = model.predict(X)
            y_log = y_scaler.inverse_transform(y_scaled.reshape(-1, 1)).ravel()
            y_pred = np.expm1(y_log)[0]

            latest.at[latest.index[-1], 'crimes'] = y_pred
            latest.at[latest.index[-1], 'forecast'] = True
            history = pd.concat([history.iloc[:-1], latest], ignore_index=True)

        history['year'] = history['date'].dt.year
        history['month'] = history['date'].dt.month

        print(history.tail(10)[['date', 'crimes', 'forecast']])  # DEBUG

        fig, ax = plt.subplots(figsize=(8, 5))

        actual = history[(history['year'] == 2023) & (~history['forecast'])]
        forecast = history[history['forecast']]

        if not actual.empty:
            monthly_actual = actual.groupby("month")["crimes"].mean()
            ax.plot(monthly_actual.index, monthly_actual.values, label="2023 Actual", marker='o')

        for y in sorted(forecast['year'].unique()):
            fy = forecast[forecast['year'] == y]
            if not fy.empty:
                monthly_pred = fy.groupby("month")["crimes"].mean()
                ax.plot(monthly_pred.index, monthly_pred.values, label=f"{y} Forecast", linestyle="--", marker='o')

        ax.set_title(f"{category.title()} - {typ.title()}")
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

        results.append({
            "category": category,
            "type": typ,
            "plot_base64": encoded
        })

    except Exception as e:
        results.append({
            "category": category,
            "type": typ,
            "error": str(e)
        })

    return {"results": results}
