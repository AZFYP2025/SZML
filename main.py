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

# Initialize Firebase
firebase_json = os.environ.get("FIREBASE_CREDENTIALS")
cred_dict = json.loads(firebase_json)
cred = credentials.Certificate(cred_dict)
initialize_app(cred, {
    'databaseURL': "https://safezone-660a9-default-rtdb.asia-southeast1.firebasedatabase.app/"
})

app = FastAPI()

# Route: Hello World
@app.get("/")
async def root():
    return {"message": "Hello World"}

# Utility: Normalize names
def safe_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_")

# Load model + scalers
def get_model_and_scalers(category: str, typ: str):
    safe_cat = safe_name(category)
    safe_typ = safe_name(typ)

    model_path = Path(f"models/{safe_cat}__{safe_typ}__model.pkl")
    x_scaler_path = Path(f"models/{safe_cat}__{safe_typ}__x_scaler.pkl")
    y_scaler_path = Path(f"models/{safe_cat}__{safe_typ}__y_scaler.pkl")

    if not model_path.exists() or not x_scaler_path.exists() or not y_scaler_path.exists():
        raise FileNotFoundError(f"Missing model/scalers for: {category} - {typ}")

    model = joblib.load(model_path)
    x_scaler = joblib.load(x_scaler_path)
    y_scaler = joblib.load(y_scaler_path)
    return model, x_scaler, y_scaler

# Load data from Firebase
def fetch_firebase_data(category: str, typ: str) -> pd.DataFrame:
    ref = db.reference("crime_data")
    data = ref.get()
    if not data:
        return pd.DataFrame()

    rows = []
    for _, v in data.items():
        if isinstance(v, dict) and v.get("source") == "synth" and v.get("category") == category and v.get("type") == typ:
            rows.append({
                "category": v.get("category"),
                "type": v.get("type"),
                "date": v.get("date"),
                "crimes": v.get("crimes"),
                "source": v.get("source")
            })

    df = pd.DataFrame(rows)
    df["crimes"] = pd.to_numeric(df["crimes"], errors="coerce")
    df = df.dropna()
    return df

# Feature engineering
def preprocess_input(df: pd.DataFrame) -> pd.DataFrame:
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['week'] = df['date'].dt.isocalendar().week.astype(int)
    df['month'] = df['date'].dt.month
    df['sin_week'] = np.sin(2 * np.pi * df['week'] / 52)
    df['cos_week'] = np.cos(2 * np.pi * df['week'] / 52)
    df['is_festive'] = df['month'].isin([1, 5, 6, 11, 12]).astype(int)
    df['is_monsoon'] = df['month'].isin([10, 11, 12, 1]).astype(int)

    df = df.sort_values("date").reset_index(drop=True)

    for lag in [1, 2, 3, 4, 52]:
        df[f'lag_{lag}'] = df['crimes'].shift(lag)
    df['rolling_4wk_mean'] = df['crimes'].rolling(4).mean().shift(1)
    df['rolling_4wk_std'] = df['crimes'].rolling(4).std().shift(1)
    df['rolling_52wk_mean'] = df['crimes'].rolling(52).mean().shift(1)
    df['yoy_change'] = df['crimes'] / df['lag_52'] - 1

    return df.dropna()

# Route: All predictions and plots
@app.get("/predict_and_plot")
async def predict_and_plot_all():
    ref = db.reference("crime_data")
    raw = ref.get()
    if not raw:
        return {"error": "No crime data found in Firebase."}

    # Extract unique (category, type) combos
    combos = set()
    for v in raw.values():
        if isinstance(v, dict) and v.get("source") == "synth":
            combos.add((safe_name(v["category"]), safe_name(v["type"])))

    results = []

    for category, typ in sorted(combos):
        try:
            df = fetch_firebase_data(category, typ)
            if df.empty or len(df) < 60:
                continue

            df = preprocess_input(df)
            model, x_scaler, y_scaler = get_model_and_scalers(category, typ)

            feature_cols = ['year', 'week', 'month', 'sin_week', 'cos_week',
                            'is_festive', 'is_monsoon',
                            'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_52',
                            'rolling_4wk_mean', 'rolling_4wk_std', 'rolling_52wk_mean',
                            'yoy_change']

            X = df[feature_cols]
            X_scaled = x_scaler.transform(X)

            y_scaled_pred = model.predict(X_scaled)
            y_log_pred = y_scaler.inverse_transform(y_scaled_pred.reshape(-1, 1)).ravel()
            y_pred = np.expm1(y_log_pred)

            df["predicted_crimes"] = y_pred
            df["year"] = df["date"].dt.year
            df["month"] = df["date"].dt.month

            # Plot
            fig, ax = plt.subplots(figsize=(8, 5))

            actual_2023 = df[df['year'] == 2023]
            if not actual_2023.empty:
                monthly_actual = actual_2023.groupby("month")["crimes"].mean()
                ax.plot(monthly_actual.index, monthly_actual.values, label="2023 Actual", marker='o')

            for yr in [2024, 2025]:
                future = df[df['year'] == yr]
                if not future.empty:
                    monthly_pred = future.groupby("month")["predicted_crimes"].mean()
                    ax.plot(monthly_pred.index, monthly_pred.values, label=f"{yr} Predicted", marker='o', linestyle='--')

            ax.set_title(f"{category.replace('_',' ').title()} - {typ.replace('_',' ').title()}")
            ax.set_xlabel("Month")
            ax.set_ylabel("Crimes")
            ax.set_xticks(range(1, 13))
            ax.legend()
            ax.grid(True)

            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()

            results.append({
                "category": category,
                "type": typ,
                "plot_base64": img_base64
            })

        except Exception as e:
            results.append({
                "category": category,
                "type": typ,
                "error": str(e)
            })

    return {"results": results}
