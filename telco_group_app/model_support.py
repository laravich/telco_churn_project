"""Shared export and prediction helpers for fitted model pipelines."""

from pathlib import Path

import joblib
import numpy as np


def export_bundle(path, model, member, threshold, positive_label=1):
    """Save a fitted pipeline that accepts original customer inputs."""
    threshold = float(threshold)

    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1.")

    if not hasattr(model, "predict_proba"):
        raise ValueError("The model must support predict_proba.")

    classes = list(model.classes_)
    if positive_label not in classes:
        raise ValueError(
            f"Churn label {positive_label!r} not found in {classes}."
        )

    import sklearn

    Path(path).parent.mkdir(parents=True, exist_ok=True)

    joblib.dump({
        "model": model,
        "member": member,
        "threshold": threshold,
        "positive_label": positive_label,
        "feature_mode": "raw",
        "sklearn_version": sklearn.__version__,
    }, path)


def predict_bundle(bundle, raw):
    """Predict one customer using the model's own complete pipeline."""
    if len(raw) != 1:
        raise ValueError("Provide exactly one customer.")

    # Older Lavanya bundles need re-exporting with feature preparation
    # included inside their pipeline.
    if bundle.get("feature_mode", "raw") != "raw":
        raise ValueError(
            "Re-export this model with its feature preparation "
            "included inside the pipeline."
        )

    model = bundle["model"]
    X = raw.copy()

    expected = getattr(model, "feature_names_in_", None)
    if expected is not None:
        missing = set(expected) - set(X.columns)
        if missing:
            raise ValueError(f"Missing model inputs: {sorted(missing)}")
        X = X.loc[:, list(expected)]

    index = list(model.classes_).index(bundle["positive_label"])
    score = float(model.predict_proba(X)[0, index])
    threshold = float(bundle["threshold"])

    if not np.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Invalid churn score.")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Invalid threshold.")

    return {
        "Member": bundle["member"],
        "Churn_score": score,
        "Threshold": threshold,
        "Prediction": (
            "Churn flagged" if score >= threshold
            else "Predicted stay"
        ),
    }