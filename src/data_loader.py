"""
data_loader.py
Loads the credit card fraud dataset and provides basic inspection utilities.
"""
import pandas as pd
import logging

def load_data(path):
    """
    Loads the credit card fraud dataset from a CSV file.
    Think of this as opening an Excel file and reading it into Python.
    Returns a DataFrame (like a table).
    """
    try:
        df = pd.read_csv(path)
        logging.info(f"Loaded data from {path} with shape {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def inspect_data(df):
    """
    Prints basic info about the data:
    - Shape: how many rows and columns
    - Columns: names of the columns (features)
    - Class distribution: how many fraud/not fraud
    """
    print('Shape:', df.shape)
    print('Columns:', df.columns.tolist())
    print('\nClass Distribution:')
    print(df['Class'].value_counts())
    print('\nClass Distribution (%):')
    print(df['Class'].value_counts(normalize=True) * 100)
