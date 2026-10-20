"""Prediction logic for the Streamlit app (no Streamlit imports, so it can be tested on its own).

It repeats the exact preprocessing from Section B of notebooks/classification.ipynb, using the objects
that were FITTED ON THE TRAINING SET and saved by the last cells of the notebook (models/preprocessing.pkl):
  raw row -> TotalCharges fix -> IQR clipping -> one-hot encoding -> engineered features -> StandardScaler -> model
"""
from pathlib import Path

import joblib
import pandas as pd

MODELS_DIR = Path(__file__).resolve().parent.parent / 'models'


def load_artifacts():
    model = joblib.load(MODELS_DIR / 'best_classifier.pkl')
    prep = joblib.load(MODELS_DIR / 'preprocessing.pkl')
    return model, prep


def build_features(raw, prep):
    """Turn rows with the raw dataset columns (no customerID / Churn) into the model's scaled inputs."""
    X = raw.copy()

    # TotalCharges: blank for brand-new customers (tenure = 0) -> 0, as in the notebook
    X['TotalCharges'] = pd.to_numeric(X['TotalCharges'], errors='coerce')
    X.loc[(X['tenure'] == 0) & X['TotalCharges'].isna(), 'TotalCharges'] = 0

    # Outlier capping with the bounds learned on the training set
    clip_cols = list(prep['clip_lower'].index)
    X[clip_cols] = X[clip_cols].astype(float).clip(prep['clip_lower'], prep['clip_upper'], axis=1)

    # One-hot encoding with the encoder fitted on the training set
    encoder = prep['encoder']
    cat_cols = list(encoder.feature_names_in_)
    encoded = pd.DataFrame(encoder.transform(X[cat_cols]),
                           columns=encoder.get_feature_names_out(cat_cols), index=X.index)
    X = pd.concat([X.drop(columns=cat_cols), encoded], axis=1)

    # Engineered features (Section B3)
    X['Charges_Per_Tenure'] = X['TotalCharges'] / (X['tenure'] + 1)
    X['NumServices'] = X[prep['service_cols']].sum(axis=1)

    # Same column order as training, then the scaler fitted on the training set
    X = X[prep['feature_columns']]
    return pd.DataFrame(prep['scaler'].transform(X), columns=prep['feature_columns'], index=X.index)


def predict_churn(raw, model, prep):
    """Return P(churn) for every row of `raw`."""
    scaled = build_features(raw[prep['raw_columns']], prep)
    churn_col = list(model.classes_).index(1)
    return model.predict_proba(scaled)[:, churn_col]
