from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from io import BytesIO
import base64

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_PATH = "seasonal_weekly_crime_data_2016_2023.csv"
MODEL_DIR = "models"

# Load full historical dataset once
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
df["year"] = df["date"].dt.year
df["week"] = df["date"].dt.isocalendar().week.astype(int)
df["month"] = df["date"].dt.month
df["sin_week"] = np.sin(2 * np.pi * df["week"] / 52)
df["cos_week"] = np.cos(2 * np.pi * df["week"] / 52)
df["is_festive"] = df["month"].isin([1, 5, 6, 11, 12]).astype(int)
df["is_monsoon"] = df["month"].isin([10, 11, 12, 1]).astype(int)


def forecast_and_plot(category: str, crime_type: str):
    try:
        safe_cat = category.replace(" ", "_").lower()
        safe_typ = crime_type.replace(" ", "_").lower()
        model_path = os.path.join(MODEL_DIR, f"{safe_cat}__{safe_typ}__model.pkl")
        x_scaler_path = os.path.join(MODEL_DIR, f"{safe_cat}__{safe_typ}__x_scaler.pkl")
        y_scaler_path = os.path.join(MODEL_DIR, f"{safe_cat}__{safe_typ}__y_scaler.pkl")

        if not os.path.exists(model_path):
            return {"category": category, "type": crime_type, "error": "model not found"}

        group = df[(df["category"] == category) & (df["type"] == crime_type)].copy()
        if group.empty:
            return {"category": category, "type": crime_type, "error": "no historical data"}

        group = group.sort_values("date").reset_index(drop=True)

        # Recreate lag features
        for lag in [1, 2, 3, 4, 52]:
            group[f"lag_{lag}"] = group["crimes"].shift(lag)
        group["rolling_4wk_mean"] = group["crimes"].rolling(4).mean().shift(1)
        group["rolling_4wk_std"] = group["crimes"].rolling(4).std().shift(1)
        group["rolling_52wk_mean"] = group["crimes"].rolling(52).mean().shift(1)
        group["yoy_change"] = group["crimes"] / group["lag_52"] - 1

        group.dropna(inplace=True)
        if group.empty:
            return {"category": category, "type": crime_type, "error": "not enough records after dropna"}

        feature_cols = ['year', 'week', 'month', 'sin_week', 'cos_week',
                        'is_festive', 'is_monsoon',
                        'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_52',
                        'rolling_4wk_mean', 'rolling_4wk_std', 'rolling_52wk_mean',
                        'yoy_change']

        model = joblib.load(model_path)
        x_scaler = joblib.load(x_scaler_path)
        y_scaler = joblib.load(y_scaler_path)

        history = group.copy()
        future_preds = []

        for _ in range(12):  # forecast 12 weeks ahead
            last_row = history.iloc[-1:].copy()

            new_week = last_row["week"].values[0] + 1
            new_year = last_row["year"].values[0]
            if new_week > 52:
                new_week = 1
                new_year += 1
            new_date = last_row["date"].values[0] + pd.Timedelta(weeks=1)
            new_month = pd.to_datetime(new_date).month

            new_data = {
                "year": new_year,
                "week": new_week,
                "month": new_month,
                "sin_week": np.sin(2 * np.pi * new_week / 52),
                "cos_week": np.cos(2 * np.pi * new_week / 52),
                "is_festive": int(new_month in [1, 5, 6, 11, 12]),
                "is_monsoon": int(new_month in [10, 11, 12, 1]),
            }

            for lag in [1, 2, 3, 4, 52]:
                new_data[f"lag_{lag}"] = history["crimes"].iloc[-lag]
            new_data["rolling_4wk_mean"] = history["crimes"].iloc[-4:].mean()
            new_data["rolling_4wk_std"] = history["crimes"].iloc[-4:].std()
            new_data["rolling_52wk_mean"] = history["crimes"].iloc[-52:].mean() if len(history) >= 52 else history["crimes"].mean()
            new_data["yoy_change"] = new_data["lag_52"]
            if new_data["lag_52"] != 0:
                new_data["yoy_change"] = new_data["lag_1"] / new_data["lag_52"] - 1
            else:
                new_data["yoy_change"] = 0

            X_new = pd.DataFrame([new_data])[feature_cols]
            X_scaled = x_scaler.transform(X_new)
            pred_scaled = model.predict(X_scaled)
            pred_log = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
            pred = np.expm1(pred_log)[0]

            pred = max(0, pred)
            pred_row = {
                "date": pd.to_datetime(new_date),
                "crimes": pred
            }

            history = pd.concat([history, pd.DataFrame([pred_row])], ignore_index=True)
            future_preds.append(pred)

        # Plotting
        history["predicted"] = False
        history.loc[history.index[-12]:, "predicted"] = True

        plt.figure(figsize=(10, 5))
        plt.plot(history["date"], history["crimes"], label="Actual", marker='o')
        plt.plot(history[history["predicted"]]["date"], history[history["predicted"]]["crimes"],
                 label="Forecast", linestyle="--", marker='o', color="red")
        plt.title(f"Crime Forecast for {category} - {crime_type}")
        plt.xlabel("Date")
        plt.ylabel("Number of Crimes")
        plt.legend()
        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")

        return {
            "category": category,
            "type": crime_type,
            "forecast": [round(p) for p in future_preds],
            "plot_base64": img_base64
        }

    except Exception as e:
        return {"category": category, "type": crime_type, "error": str(e)}


@app.get("/plot_by_crime_type")
def plot_all():
    all_combinations = df.groupby(['category', 'type']).size().reset_index().values.tolist()
    results = []
    for cat, typ, _ in all_combinations:
        result = forecast_and_plot(cat, typ)
        results.append(result)
    return JSONResponse(content={"results": results})
