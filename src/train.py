

"""
train.py
Trains Logistic Regression and Random Forest models. Supports class_weight and SMOTE.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib
import logging
from pathlib import Path

def train_logistic_regression(X, y, class_weight=None):
    """
    Trains a Logistic Regression model.
    This is a simple model that tries to draw a line between fraud and not fraud.
    """
    model = LogisticRegression(class_weight=class_weight, max_iter=1000, random_state=42)
    model.fit(X, y)
    logging.info("Trained Logistic Regression model.")
    return model

def train_random_forest(X, y, class_weight=None):
    """
    Trains a Random Forest model.
    This model uses many decision trees and is usually more powerful.
    """
    model = RandomForestClassifier(class_weight=class_weight, n_estimators=100, random_state=42)
    model.fit(X, y)
    logging.info("Trained Random Forest model.")
    return model

def save_model(model, path):
    """
    Saves the trained model to a file so it can be used later (like saving a recipe).
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    logging.info("Model saved to %s", output_path)
