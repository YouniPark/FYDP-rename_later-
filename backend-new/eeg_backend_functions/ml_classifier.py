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
import os
import threading
from datetime import datetime, timezone
from typing import Optional
import numpy as np


_csv_lock = threading.Lock()


def _append_feature_row(csv_path: str, features: np.ndarray, is_unfamiliar: bool) -> None:
    """Append one row to a feature-log CSV; creates the file with a header if needed."""
    features_flat = features.flatten()
    n = len(features_flat)
    fieldnames = ["timestamp_utc", "is_unfamiliar"] + [f"feat_{i}" for i in range(n)]
    row: dict = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "is_unfamiliar": int(is_unfamiliar),
    }
    row.update({f"feat_{i}": features_flat[i] for i in range(n)})

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
    raw_csv_path: Optional[str] = None,
    scaled_csv_path: Optional[str] = None,
) -> bool:
    """
    Inputs
    ------
    features : np.ndarray
        1D statistical feature vector (same column order as training).
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
        is_unfamiliar = bool(pred)
    elif isinstance(pred, (np.integer, int)):
        is_unfamiliar = bool(int(pred) == 1)
    else:
        # If it returned strings/labels:
        is_unfamiliar = str(pred).strip().lower() in {"1", "true", "unf", "unfamiliar", "unrecognized"}

    if raw_csv_path:
        _append_feature_row(raw_csv_path, x, is_unfamiliar)
    if scaled_csv_path:
        _append_feature_row(scaled_csv_path, x_scaled, is_unfamiliar)

    return is_unfamiliar
