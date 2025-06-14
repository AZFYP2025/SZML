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


def fetch_firebase_data(category: str, typ: str) -> pd.DataFrame:
    ref = db.reference("crime_data")
    data = ref.get()
    rows = []
    for _, v in data.items():
        if isinstance(v, dict) and v.get("source") == "synth" and v.get("category") == category and v.get("type") == typ:
            rows.append({
                "category": v.get("category"),
                "type": v.get("type"),
                "date": v.get("date"),
                "crimes": v.get("crimes")
            })
    df = pd.DataFrame(rows)
    df["crimes"] = pd.to_numeric(df["crimes"], errors="coerce")
    return df.dropna()


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
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
async def predict_and_plot_all():
    ref = db.reference("crime_data")
    raw = ref.get()
    if not raw:
        return {"error": "No crime data found in Firebase."}

    combos = {(safe_name(v["category"]), safe_name(v["type"]))
              for v in raw.values() if v.get("source") == "synth"}

    results = []

    for category, typ in sorted(combos):
        try:
            df = fetch_firebase_data(category, typ)
            if df.empty or len(df) < 60:
                continue

            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
            df = engineer_features(df)

            model, x_scaler, y_scaler = get_model_and_scalers(category, typ)

            future_weeks = 104
            future_rows = []

            for _ in range(future_weeks):
                last_row = df.iloc[-1].copy()
                new_date = last_row["date"] + pd.Timedelta(weeks=1)
                new_row = {
                    "date": new_date,
                    "crimes": np.nan
                }

                temp_df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                temp_df = engineer_features(temp_df)

                feature_cols = ['year', 'week', 'month', 'sin_week', 'cos_week',
                                'is_festive', 'is_monsoon',
                                'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_52',
                                'rolling_4wk_mean', 'rolling_4wk_std', 'rolling_52wk_mean',
                                'yoy_change']

                row = temp_df.iloc[[-1]]
                if row[feature_cols].isnull().any(axis=1).values[0]:
                    break  # stop if features aren't complete

                X = x_scaler.transform(row[feature_cols])
                y_scaled = model.predict(X)
                y_log = y_scaler.inverse_transform(y_scaled.reshape(-1, 1)).ravel()
                y_pred = np.expm1(y_log)[0]

                new_row["crimes"] = y_pred
                new_row["predicted"] = True
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

            df["predicted"] = df["predicted"] if "predicted" in df else False
            df["year"] = df["date"].dt.year
            df["month"] = df["date"].dt.month

            # Plotting
            fig, ax = plt.subplots(figsize=(8, 5))

            actual = df[(df["year"] == 2023) & (df["predicted"] != True)]
            if not actual.empty:
                monthly_actual = actual.groupby("month")["crimes"].mean()
                ax.plot(monthly_actual.index, monthly_actual.values, label="2023 Actual", marker='o')

            for yr in [2024, 2025]:
                pred = df[(df["year"] == yr) & (df["predicted"] == True)]
                if not pred.empty:
                    monthly_pred = pred.groupby("month")["crimes"].mean()
                    ax.plot(monthly_pred.index, monthly_pred.values, label=f"{yr} Predicted", linestyle='--', marker='o')

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
