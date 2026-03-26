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

Optional CSV logging
--------------------
Pass ``raw_csv_path`` and/or ``scaled_csv_path`` to append a row containing
a UTC timestamp, the prediction label, and each feature value to the
respective file.  The header is written automatically when the file is new
or empty.  Both files are written atomically under a module-level lock so
concurrent pipeline calls don't interleave rows.
"""

from __future__ import annotations

import csv
import logging
import os
import threading
import warnings
from datetime import datetime, timezone
from typing import Optional, Sequence
import numpy as np


logger = logging.getLogger(__name__)
_csv_lock = threading.Lock()


def _append_feature_row(
    csv_path: str,
    features: np.ndarray,
    is_unfamiliar: bool,
    feature_names: "Optional[Sequence[str]]" = None,
) -> None:
    """Append one row to a feature-log CSV; creates the file with a header if needed."""
    features_flat = features.flatten()
    n = len(features_flat)
    names = list(feature_names) if feature_names is not None else [f"feat_{i}" for i in range(n)]
    fieldnames = ["timestamp_utc", "is_unfamiliar"] + names
    row: dict = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "is_unfamiliar": int(is_unfamiliar),
    }
    row.update({name: features_flat[i] for i, name in enumerate(names)})

    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    file_exists = os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0
    with _csv_lock:
        with open(csv_path, "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)


def ml_classifier(
    features: np.ndarray,
    *,
    model: Optional[object] = None,
    scaler: Optional[object] = None,
    model_path: Optional[str] = None,
    scaler_path: Optional[str] = None,
    scale_divisor: Optional[float] = 200,
    raw_csv_path: Optional[str] = None,
    scaled_csv_path: Optional[str] = None,
    feature_names: "Optional[Sequence[str]]" = None,
) -> bool:
    """
    Inputs
    ------
    features : np.ndarray
        1D statistical feature vector (same column order as training).
    scaler : sklearn-style scaler (has ``.transform()``) or any callable, optional
        If a callable without ``.transform``, it is called directly as
        ``scaler(x)`` and must return the scaled array.  Pass e.g.
        ``scaler=lambda x: x / 200`` for a simple divisor.
    scale_divisor : float, optional
        If provided, raw features are divided by this value *before* being
        passed to any scaler.  Useful as a quick brute-force normalisation
        when the fitted scaler leaves values too large (e.g. ``scale_divisor=200``).
    raw_csv_path : str, optional
        If provided, append the pre-scaled feature vector to this CSV file.
    scaled_csv_path : str, optional
        If provided, append the post-scaler feature vector to this CSV file.

    Outputs
    -------
    is_unfamiliar : bool
        True if predicted unfamiliar/unrecognized, else False.
    """
    x = np.asarray(features, dtype=float).reshape(1, -1)

    # Early check for NaN or Inf in input features
    if not np.all(np.isfinite(x)):
        nan_count = np.isnan(x).sum()
        inf_count = np.isinf(x).sum()
        logger.warning(
            "ml_classifier: input features contain %d NaN and %d Inf values (shape=%s) — "
            "this typically means eeg_processing encountered all-bad channels or empty time windows. "
            "Treating as unfamiliar.",
            nan_count, inf_count, x.shape,
        )
        return True  # Default to unfamiliar if input is invalid

    # logger.debug(
    #     "ml_classifier: input features shape=%s, range=[%.3f, %.3f]",
    #     x.shape,
    #     np.nanmin(x),
    #     np.nanmax(x),
    # )

    if (model is None or scaler is None) and (model_path or scaler_path):
        # Lazy-load if paths provided
        try:
            import joblib  # type: ignore
        except Exception as e:
            raise ImportError("joblib is required to load model/scaler from disk.") from e

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Trying to unpickle estimator")
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

    # Optional brute-force pre-scaling divisor applied before the scaler
    if scale_divisor is not None:
        logger.debug("ml_classifier: applying scale_divisor=%.4g", scale_divisor)
        x = x / scale_divisor

    # Support two scaler styles:
    #   1. sklearn-style object with .transform()  (default)
    #   2. any plain callable, called directly as scaler(x)
    logger.debug("ml_classifier: running scaler (input shape=%s)", x.shape)
    if not hasattr(scaler, "transform") and callable(scaler):
        x_scaled = np.asarray(scaler(x), dtype=float)
        logger.debug("ml_classifier: custom callable scaler applied")
    else:
        # If the scaler was fitted with a DataFrame, pass a DataFrame so column names
        # match and sklearn does not emit a feature-names warning.
        x_input: object
        if hasattr(scaler, "feature_names_in_"):
            import pandas as pd
            x_input = pd.DataFrame(x, columns=scaler.feature_names_in_)
            logger.debug("ml_classifier: using DataFrame input with %d named features", len(scaler.feature_names_in_))
        else:
            x_input = x

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names")
            x_scaled = scaler.transform(x_input)
    logger.debug("ml_classifier: scaler complete; running model.predict")
    y_pred = model.predict(x_scaled)
    logger.debug("ml_classifier: model.predict returned %s", y_pred)

    # Log decision scores or probabilities when available
    if hasattr(model, "decision_function"):
        try:
            scores = model.decision_function(x_scaled)
            logger.debug("ml_classifier: decision_function scores=%s", scores)
        except Exception:
            pass
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(x_scaled)
            logger.debug("ml_classifier: predict_proba=%s  classes=%s", proba, getattr(model, 'classes_', '?'))
        except Exception:
            pass

    # Handle classifiers that return shape (1,) with bool/int
    pred = y_pred[0]
    if isinstance(pred, (np.bool_, bool)):
        is_unfamiliar = bool(pred)
    elif isinstance(pred, (np.integer, int)):
        is_unfamiliar = bool(int(pred) == 1)
    else:
        # If it returned strings/labels:
        is_unfamiliar = str(pred).strip().lower() in {"1", "true", "unf", "unfamiliar", "unrecognized"}

    if raw_csv_path:
        _append_feature_row(raw_csv_path, x, is_unfamiliar, feature_names=feature_names)
    if scaled_csv_path:
        _append_feature_row(scaled_csv_path, x_scaled, is_unfamiliar, feature_names=feature_names)

    return is_unfamiliar
