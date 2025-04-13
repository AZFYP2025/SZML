from typing import Optional

from fastapi import FastAPI
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64
import io
import joblib
from fastapi import FastAPI
from firebase_admin import credentials, initialize_app, db
from sklearn.preprocessing import OneHotEncoder
import json
import os

firebase_json = os.environ.get("FIREBASE_CREDENTIALS")
cred_dict = json.loads(firebase_json)
cred = credentials.Certificate(cred_dict)

initialize_app(cred, {
    'databaseURL': "https://safezone-660a9-default-rtdb.asia-southeast1.firebasedatabase.app/"
})


# Load model and encoder
model = joblib.load("model.pkl")
encoder: OneHotEncoder = joblib.load("encoder.pkl")

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

# Fetch and aggregate data from Firebase
def fetch_and_aggregate_firebase(path: str) -> pd.DataFrame:
    # Connect to the Firebase path
    ref = db.reference(path)
    raw_data = ref.get()

    # If no data, return empty DataFrame
    if not raw_data:
        return pd.DataFrame()

    # Ensure data is in list-of-dict format
    if isinstance(raw_data, dict):
        records = list(raw_data.values())
    elif isinstance(raw_data, list):
        records = raw_data
    else:
        raise ValueError("Unexpected data format from Firebase")

    # Convert to DataFrame
    df = pd.DataFrame(records)

    # Coerce date column and drop rows with missing essential fields
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(subset=['date', 'category', 'type'], inplace=True)

    # Extract month and year from date
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year

    # Group by category, type, month, year to count crimes
    summary = df.groupby(['category', 'type', 'month', 'year']).size().reset_index(name='crimes')

    return summary


@app.get("/predict_firebase_summary")
def predict_firebase_summary():
    df_summary = fetch_and_aggregate_firebase("crime_data")
    if df_summary.empty:
        return {"error": "No data found to summarize"}

    X_cat = df_summary[['category', 'type']]
    X_num = df_summary[['month', 'year']]

    try:
        X_encoded = encoder.transform(X_cat)
    except:
        return {"error": "OneHotEncoder categories mismatch – ensure encoder.pkl matches Firebase data"}

    X_input = np.hstack([X_encoded, X_num])
    predictions = model.predict(X_input)
    df_summary['predicted_crimes'] = predictions

    return df_summary.to_dict(orient="records")


@app.get("/plot_firebase_prediction")
def plot_firebase_prediction():
    df_summary = fetch_and_aggregate_firebase("crime_data")
    if df_summary.empty:
        return {"error": "No data found to summarize"}

    X_cat = df_summary[['category', 'type']]
    X_num = df_summary[['month', 'year']]
    X_encoded = encoder.transform(X_cat)
    X_input = np.hstack([X_encoded, X_num])
    predictions = model.predict(X_input)

    # Plot predictions
    plt.figure(figsize=(10, 5))
    plt.plot(predictions, marker='o')
    plt.title("Predicted Crime Counts from Firebase Data")
    plt.xlabel("Record Index")
    plt.ylabel("Predicted Crimes")
    plt.grid(True)

    # Save plot to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    return {"image": base64.b64encode(buf.read()).decode("utf-8")}
