"""
app.py
Streamlit app for Credit Card Fraud Detection
"""
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / 'models' / 'model.pkl'
SCALER_PATH = ROOT_DIR / 'models' / 'scaler.pkl'
FEATURE_NAMES = [f'V{i}' for i in range(1, 29)] + ['Amount']

st.title('Credit Card Fraud Detection')
st.write('Input transaction features to predict fraud risk.')

# Load model
if not MODEL_PATH.exists():
    st.error('Model file not found. Please train and save the model first.')
    st.stop()
if not SCALER_PATH.exists():
    st.error('Scaler file not found. Please retrain the model pipeline to save the scaler.')
    st.stop()

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

input_data = {}
for col in FEATURE_NAMES:
    input_data[col] = st.number_input(col, value=0.0)
input_df = pd.DataFrame([input_data])

input_scaled = scaler.transform(input_df)

# Predict
pred = model.predict(input_scaled)[0]
proba = model.predict_proba(input_scaled)[0][1]

st.write(f'Prediction: {"Fraud" if pred == 1 else "Not Fraud"}')
st.write(f'Fraud Probability: {proba:.4f}')
