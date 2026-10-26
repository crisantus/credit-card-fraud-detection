"""
api.py
Optional: FastAPI endpoint for Credit Card Fraud Detection
"""
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()
ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / 'models' / 'model.pkl'
SCALER_PATH = ROOT_DIR / 'models' / 'scaler.pkl'
FEATURE_NAMES = [f'V{i}' for i in range(1, 29)] + ['Amount']
EXPECTED_FEATURE_COUNT = len(FEATURE_NAMES)

class Transaction(BaseModel):
    features: List[float]

@app.post('/predict')
def predict(transaction: Transaction):
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=500, detail='Model file not found.')
    if not SCALER_PATH.exists():
        raise HTTPException(status_code=500, detail='Scaler file not found.')
    if len(transaction.features) != EXPECTED_FEATURE_COUNT:
        raise HTTPException(
            status_code=400,
            detail=f'Expected {EXPECTED_FEATURE_COUNT} features, received {len(transaction.features)}.',
        )

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    X = pd.DataFrame([transaction.features], columns=FEATURE_NAMES)
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0][1]
    return {'prediction': int(pred), 'probability': float(proba)}
