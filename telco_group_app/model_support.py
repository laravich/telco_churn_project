from pathlib import Path
import joblib
import numpy as np
import pandas as pd

def engineer_features(X):
    X = X.copy()
    addons = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    X['InternetAddonCount'] = X[addons].eq('Yes').sum(axis=1)
    X['AutomaticPayment'] = X['PaymentMethod'].isin(['Bank transfer (automatic)', 'Credit card (automatic)']).astype(int)
    X['NewMonthToMonth'] = (X['tenure'].le(6) & X['Contract'].eq('Month-to-month')).astype(int)
    return X

def export_bundle(path, model, member, threshold, feature_mode='raw', positive_label=1):
    """Export a fitted pipeline, never refit it. Feature mode: raw or lavanya_v2."""
    if feature_mode not in ('raw', 'lavanya_v2'):
        raise ValueError('Unknown feature mode')
    if not 0 <= float(threshold) <= 1:
        raise ValueError('Threshold must be between 0 and 1')
    if not hasattr(model, 'predict_proba') or positive_label not in list(model.classes_):
        raise ValueError('A fitted classifier with predict_proba and the correct positive label is required')
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    import sklearn
    joblib.dump(dict(model=model, member=member, threshold=float(threshold), feature_mode=feature_mode,
                     positive_label=positive_label, sklearn_version=sklearn.__version__), path)

def predict_bundle(bundle, raw):
    mode = bundle['feature_mode']
    if mode not in ('raw', 'lavanya_v2'):
        raise ValueError(f'Unsupported feature mode: {mode}')
    X = engineer_features(raw) if mode == 'lavanya_v2' else raw.copy()
    model = bundle['model']
    expected = getattr(model, 'feature_names_in_', None)
    if expected is not None:
        missing = set(expected) - set(X.columns)
        if missing:
            raise ValueError(f'Missing model inputs: {sorted(missing)}')
        X = X.loc[:, list(expected)]
    index = list(model.classes_).index(bundle['positive_label'])
    score = float(model.predict_proba(X)[0, index])
    threshold = float(bundle['threshold'])
    if not np.isfinite(score) or not 0 <= score <= 1 or not 0 <= threshold <= 1:
        raise ValueError('Invalid score or threshold')
    return dict(Member=bundle['member'], Churn_score=score, Threshold=threshold,
                Prediction='Churn flagged' if score >= threshold else 'Predicted stay')
