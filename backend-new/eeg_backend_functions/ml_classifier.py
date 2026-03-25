"""
ML Classifier: predict familiarity from a 1D feature vector.

The classifier notebook scales features with StandardScaler and then
trains classifiers (SVM/LogReg/RF). That notebook doesn't persist models,
so this function supports either:
- passing in a (scaler, model) pair already loaded in memory, or
- loading them from disk (joblib/pickle) if you have saved them.

Output is a boolean flag:
- True  => is_unfamiliar / is_unrecognized
- False => familiar
"""

from __future__ import annotations

from typing import Optional
import numpy as np


def ml_classifier(
    features: np.ndarray,
    *,
    model: Optional[object] = None,
    scaler: Optional[object] = None,
    model_path: Optional[str] = None,
    scaler_path: Optional[str] = None,
) -> bool:
    """
    Inputs
    ------
    features : np.ndarray
        1D statistical feature vector (same column order as training).

    Outputs
    -------
    is_unfamiliar : bool
        True if predicted unfamiliar/unrecognized, else False.
    """
    x = np.asarray(features, dtype=float).reshape(1, -1)

    if (model is None or scaler is None) and (model_path or scaler_path):
        # Lazy-load if paths provided
        try:
            import joblib  # type: ignore
        except Exception as e:
            raise ImportError("joblib is required to load model/scaler from disk.") from e

        if scaler is None:
            if not scaler_path:
                raise ValueError("scaler_path must be provided if scaler is None.")
            scaler = joblib.load(scaler_path)

        if model is None:
            if not model_path:
                raise ValueError("model_path must be provided if model is None.")
            model = joblib.load(model_path)

    if scaler is None or model is None:
        raise ValueError(
            "ml_classifier() needs a trained `model` and `scaler`, "
            "or `model_path` and `scaler_path` to load them."
        )

    x_scaled = scaler.transform(x)
    y_pred = model.predict(x_scaled)

    # Handle classifiers that return shape (1,) with bool/int
    pred = y_pred[0]
    if isinstance(pred, (np.bool_, bool)):
        return bool(pred)
    if isinstance(pred, (np.integer, int)):
        return bool(int(pred) == 1)
    # If it returned strings/labels:
    return str(pred).strip().lower() in {"1", "true", "unf", "unfamiliar", "unrecognized"}
