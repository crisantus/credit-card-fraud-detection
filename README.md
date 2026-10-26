# Credit Card Fraud Detection

A Python machine learning prototype for exploring fraud classification on imbalanced transaction data, with reusable preprocessing, training, evaluation, and prediction interfaces.

## Engineering approach

- Stratified train-test splitting preserves the fraud class distribution.
- `StandardScaler` is fitted on training data and reused for test and prediction inputs.
- SMOTE is applied only to training data to avoid synthetic examples entering the held-out split.
- Logistic regression and random forest training functions support optional class weighting.
- Evaluation reports precision, recall, F1, ROC-AUC, and a confusion matrix rather than relying on accuracy alone.
- The model and scaler are saved together for consistent inference.
- FastAPI exposes `/predict`; Streamlit provides a simple interactive input form.

Recall measures missed fraud, while precision helps assess false alarms. The project makes both visible instead of treating a single accuracy score as sufficient.

## Run

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) separately and place `creditcard.csv` in `data/`. Data and trained models are excluded from Git.

From the repository root:

```sh
python src/example_usage.py
streamlit run app/app.py
```

Or serve the prediction API after training:

```sh
uvicorn app.api:app --reload
```

`POST /predict` accepts a JSON object containing `features`: 29 numbers in the order `V1` through `V28`, followed by `Amount`. It returns a predicted class and the model's fraud probability.

## Scope and evaluation limits

This is a portfolio prototype, with no claimed benchmark results or production deployment. The example selects a model by recall on its held-out split; a rigorous experiment should use a separate validation split for selection and reserve an untouched test set for final reporting. Probability calibration and operating-threshold selection are future evaluation work.

## Structure

- `src/data_loader.py`: CSV loading and class inspection.
- `src/preprocess.py`: splitting, scaling, and optional SMOTE.
- `src/train.py`: estimators and model serialization.
- `src/evaluate.py`: classification metrics and plots.
- `src/example_usage.py`: end-to-end example.
- `app/api.py` and `app/app.py`: prediction interfaces.
