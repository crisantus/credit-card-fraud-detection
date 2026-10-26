"""
utils.py
General utility functions (e.g., logging setup, feature importance plotting).
"""
import logging
import matplotlib.pyplot as plt
import numpy as np

def setup_logging():
    """
    Sets up logging so you can see info and errors in the console.
    This helps track what the code is doing.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def plot_feature_importance(model, feature_names, top_n=10):
    """
    Shows which features (columns) are most important for the model's decisions.
    Only works for models like Random Forest.
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        num_features = len(indices)
        plt.figure(figsize=(8,5))
        plt.title('Top Feature Importances')
        plt.bar(range(num_features), importances[indices], align='center')
        plt.xticks(range(num_features), [feature_names[i] for i in indices], rotation=45)
        plt.tight_layout()
        plt.show()
        plt.close()
    else:
        print('Model does not support feature importances.')
