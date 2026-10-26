"""
preprocess.py
Handles feature/target split, train-test split, scaling, and imbalance handling (SMOTE).
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import logging

def preprocess_data(df, test_size=0.2, random_state=42, use_smote=True):
    """
    Prepares the data for modeling:
    - Splits into features (X) and target (y)
    - Splits into training and test sets (80/20)
    - Scales the features (makes numbers similar scale)
    - Handles imbalance using SMOTE (makes fraud/not fraud more balanced)
    Returns processed data and the scaler.
    """
    if 'Class' not in df.columns:
        raise KeyError("Expected a 'Class' column in the input DataFrame.")

    y = df['Class']  # Target: 1 = fraud, 0 = not fraud
    columns_to_drop = ['Class']
    if 'Time' in df.columns:
        columns_to_drop.append('Time')
    X = df.drop(columns=columns_to_drop)  # Features: drop target and time if present
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()  # Makes all features have mean=0, std=1
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    if use_smote:
        smote = SMOTE(random_state=random_state)
        X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
        logging.info(f"After SMOTE: {dict(pd.Series(y_train_res).value_counts())}")
        return X_train_res, X_test_scaled, y_train_res, y_test, scaler
    else:
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler
