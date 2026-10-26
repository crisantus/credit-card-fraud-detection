"""
example_usage.py
Example script to run the full fraud detection pipeline step by step.
Each step is explained in simple terms for easy understanding.
"""
import joblib
from pathlib import Path

try:
    from .data_loader import load_data, inspect_data
    from .preprocess import preprocess_data
    from .train import train_logistic_regression, train_random_forest, save_model
    from .evaluate import evaluate_model
    from .utils import setup_logging, plot_feature_importance
except ImportError:
    from data_loader import load_data, inspect_data
    from preprocess import preprocess_data
    from train import train_logistic_regression, train_random_forest, save_model
    from evaluate import evaluate_model
    from utils import setup_logging, plot_feature_importance

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / 'data' / 'creditcard.csv'
MODEL_PATH = ROOT_DIR / 'models' / 'model.pkl'
SCALER_PATH = ROOT_DIR / 'models' / 'scaler.pkl'

def main():
    # 1. Set up logging (shows info in the console)
    setup_logging()

    # 2. Load the dataset (reads the CSV file into a table)
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Place creditcard.csv in the project's data directory."
        )
    df = load_data(DATA_PATH)

    # 3. Inspect the data (see what it looks like)
    inspect_data(df)

    # 4. Preprocess the data (split, scale, handle imbalance)
    # This step splits the data into features (X) and target (y),
    # splits into train/test, scales the numbers, and balances the classes using SMOTE.
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

    # 5. Train models (fit the models to the training data)
    # Logistic Regression is a simple model, Random Forest is more powerful.
    lr_model = train_logistic_regression(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)

    # 6. Evaluate models (see how well they work on new data)
    lr_metrics = evaluate_model(lr_model, X_test, y_test, 'Logistic Regression')
    rf_metrics = evaluate_model(rf_model, X_test, y_test, 'Random Forest')

    # 7. Save the best model (the one with the highest recall)
    best_model = rf_model if rf_metrics['recall'] >= lr_metrics['recall'] else lr_model
    save_model(best_model, MODEL_PATH)

    # 8. (Optional) Show which features are most important (for Random Forest)
    feature_columns = [column for column in df.columns if column not in {'Class', 'Time'}]
    plot_feature_importance(rf_model, feature_names=feature_columns)

    # 9. (Optional) Save the scaler for use in the app
    SCALER_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)

if __name__ == '__main__':
    main()
