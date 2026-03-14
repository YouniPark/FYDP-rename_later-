"""
EEG processing: extract statistical ERP features from an epoch (MNE Epochs).
"""

from __future__ import annotations

from typing import Dict, Tuple, Sequence, Optional
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
import mne


def extract_epoch_features(epochs: mne.Epochs, ch_windows: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
    """
    (aadapted from EEG_Data_Analysis_2.ipynb, cell defining extract_epoch_features)
    """
    sfreq = float(epochs.info.get('sfreq', 250))

    # Select channels present in epochs
    name_map = {ch.upper(): ch for ch in epochs.ch_names}
    selected = {}
    for ch, win in ch_windows.items():
        if ch.upper() in name_map:
            selected[name_map[ch.upper()]] = win
    if not selected:
        raise RuntimeError(f"None of target channels {list(ch_windows.keys())} found in epochs: {epochs.ch_names}")

    X = epochs.get_data()  # (n_epochs, n_channels, n_times)
    ch_idx = {ch: epochs.ch_names.index(ch) for ch in selected.keys()}

    # Epoch condition labels from event IDs
    id_map = {v: k for k, v in epochs.event_id.items()}
    conditions = [id_map.get(code, str(code)) for code in epochs.events[:, 2]]

    # Pre-compute sample index ranges per channel window
    idx_windows = {}
    for ch, (t0, t1) in selected.items():
        i0, i1 = epochs.time_as_index([float(t0), float(t1)])
        if i0 > i1:  # Flip if reversed
            i0, i1 = i1, i0
        idx_windows[ch] = (int(i0), int(i1))

    stat_order = ['mean', 'median', 'max', 'min', 'ptp', 'std', 'skew', 'auc', 'kurtosis']

    # Define N250 and P300 channel groups
    n250_channels = ['P7', 'P8', 'O1', 'O2']
    p300_channels = ['Cz', 'Pz']

    # Identify present channels for each group
    n250_present = [ch for ch in n250_channels if ch in selected.keys()]
    p300_present = [ch for ch in p300_channels if ch in selected.keys()]

    # Identify absent channels for each group
    n250_absent = [ch for ch in n250_channels if ch not in selected.keys()]
    p300_absent = [ch for ch in p300_channels if ch not in selected.keys()]

    rows = []
    for epoch in range(X.shape[0]):
        row = {}
        for ch, (i0, i1) in idx_windows.items():
            x = X[epoch, ch_idx[ch], i0:i1 + 1] * 1e6  # convert to µV
            if x.size == 0:
                vals = {s: float('nan') for s in stat_order}
            else:
                vals = {
                    'mean': float(np.mean(x)),
                    'median': float(np.median(x)),
                    'max': float(np.max(x)),
                    'min': float(np.min(x)),
                    'ptp': float(np.ptp(x)),
                    'std': float(np.std(x, ddof=0)),
                    'skew': float(skew(x, bias=False)) if x.size > 2 else float('nan'),
                    'auc': float(np.trapezoid(x, dx=1.0 / sfreq)),
                    'kurtosis': float(kurtosis(x, fisher=False, bias=False)) if x.size > 3 else float('nan'),
                }
            for s in stat_order:
                row[f"{ch}_{s}"] = vals[s]

        # Compute averaged features for N250
        if n250_present:
            for s in stat_order:
                values = [row[f"{ch}_{s}"] for ch in n250_present if f"{ch}_{s}" in row]
                if values:
                    row[f"N250avg_{s}"] = float(np.nanmean(values))
                    for ch in n250_absent:
                        row[f"{ch}_{s}"] = float(np.nanmean(values))
                else:
                    row[f"N250avg_{s}"] = float('nan')
                    for ch in n250_absent:
                        row[f"{ch}_{s}"] = float('nan')

        # Compute averaged features for P300
        if p300_present:
            for s in stat_order:
                values = [row[f"{ch}_{s}"] for ch in p300_present if f"{ch}_{s}" in row]
                if values:
                    row[f"P300avg_{s}"] = float(np.nanmean(values))
                    for ch in p300_absent:
                        row[f"{ch}_{s}"] = float(np.nanmean(values))
                else:
                    row[f"P300avg_{s}"] = float('nan')
                    for ch in p300_absent:
                        row[f"{ch}_{s}"] = float('nan')

        row['condition'] = conditions[epoch] if epoch < len(conditions) else None
        rows.append(row)

    df = pd.DataFrame(rows)

    preferred_ch_order = ['Cz','Pz', 'P7', 'P8', 'O1','O2']
    ordered_cols = []
    for ch in preferred_ch_order:
        if ch in selected.keys() or ch in n250_absent or ch in p300_absent:
            ordered_cols.extend([f"{ch}_{s}" for s in stat_order])
    if n250_present:
        ordered_cols.extend([f"N250avg_{s}" for s in stat_order])
    if p300_present:
        ordered_cols.extend([f"P300avg_{s}" for s in stat_order])

    cols = ['condition'] + [c for c in ordered_cols if c in df.columns]
    df = df[cols]
    return df


def eeg_processing(
    epoch_eeg_data: mne.Epochs,
    *,
    ch_windows: Optional[Dict[str, Tuple[float, float]]] = None,
) -> np.ndarray:
    """
    Extract a 1D feature vector from a single epoch (Epochs with len==1).

    Inputs
    ------
    epoch_eeg_data : mne.Epochs
        Single-epoch MNE object.

    Outputs
    -------
    features : np.ndarray, shape (n_features,)
        1D statistical feature vector (ordered as in training CSV columns).
    """
    if ch_windows is None:
        win_n250 = (0.200, 0.300)
        win_p300 = (0.250, 0.350)
        ch_windows = {
            'Cz': win_p300,
            'Pz': win_p300,
            'P7': win_n250,
            'P8': win_n250,
            'O1': win_n250,
            'O2': win_n250,
        }

    df = extract_epoch_features(epoch_eeg_data, ch_windows)

    # Drop condition and return as 1D vector for the first epoch
    feature_cols = [c for c in df.columns if c != "condition"]
    feats = df.loc[0, feature_cols].to_numpy(dtype=float, copy=False)

    return feats
