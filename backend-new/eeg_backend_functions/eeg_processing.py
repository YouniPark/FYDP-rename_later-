"""
EEG processing: full single-trial preprocessing pipeline and statistical ERP feature extraction.

Pipeline order (each step is optional / configurable):
  1. Bandpass + notch filter
  2. ICA artefact removal (requires a saved .fif ICA from a calibration session)
  3. REST re-referencing (requires a saved forward solution, or computed on the fly)
  4. Baseline correction
  5. Artifact quality check  →  raises EpochRejectedError if epoch is too noisy
  6. Statistical ERP feature extraction  →  returns 1D np.ndarray
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import mne
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class EpochRejectedError(Exception):
    """Raised when a single epoch fails artifact quality checks.

    The pipeline treats this the same as *is_unfamiliar=True* but logs it
    separately so that artifact-driven responses can be distinguished from
    genuine classifier outputs.
    """

    def __init__(self, message: str, bad_channels: List[str]) -> None:
        super().__init__(message)
        self.bad_channels = bad_channels


# ---------------------------------------------------------------------------
# Preprocessing steps
# ---------------------------------------------------------------------------

def filter_epoch(
    epoch: mne.Epochs,
    l_freq: float = 1.0,
    h_freq: float = 40.0,
    notch_freqs: Optional[List[float]] = None,
) -> mne.Epochs:
    """Apply a bandpass and optional notch filter to epoch data.

    Parameters
    ----------
    epoch : mne.Epochs
        Input epoch (need not be preloaded).
    l_freq, h_freq : float
        Lower and upper passband frequencies in Hz.
    notch_freqs : list of float, optional
        Power-line frequencies to notch out (Hz).  Defaults to [60.0, 120.0].
    """
    if notch_freqs is None:
        notch_freqs = [60.0, 120.0]
    epoch_filt = epoch.copy().load_data()
    epoch_filt.filter(l_freq, h_freq, verbose=False)
    if notch_freqs:
        epoch_filt.notch_filter(freqs=notch_freqs, verbose=False)
    return epoch_filt


def apply_ica_to_epoch(epoch: mne.Epochs, ica_path: str) -> mne.Epochs:
    """Load a saved ICA object and apply it to remove artefacts.

    The ICA file must be a MNE .fif ICA file saved from a calibration session
    with the exclusion list already set.

    Parameters
    ----------
    epoch : mne.Epochs
        Filtered epoch data.
    ica_path : str
        Path to the saved ICA .fif file (e.g. ``"./data/ica/sub01-ica.fif"``).
    """
    ica = mne.preprocessing.read_ica(ica_path, verbose=False)
    logger.info("Loaded ICA from %s; excluding components: %s", ica_path, ica.exclude)
    if not ica.exclude:
        logger.warning("Loaded ICA at '%s' has no components marked for exclusion", ica_path)
    epoch_clean = epoch.copy()
    ica.apply(epoch_clean, verbose=False)
    return epoch_clean


def apply_rest_reference_to_epoch(
    epoch: mne.Epochs,
    forward_path: Optional[str] = None,
) -> mne.Epochs:
    """Re-reference to REST (Reference Electrode Standardization Technique).

    REST is an infinite-reference technique that removes the contribution of the
    physical reference electrode using a forward model.

    Parameters
    ----------
    epoch : mne.Epochs
        Epoch with a standard montage already set.
    forward_path : str, optional
        Path to a pre-computed MNE forward solution (.fif).  If *None*, a
        sphere-model forward is computed automatically (slow; ~30 s).
    """
    if epoch.get_montage() is None:
        raise ValueError(
            "A standard montage must be set on the epoch before applying REST "
            "re-referencing (e.g. epoch.set_montage('standard_1020'))."
        )

    if forward_path is not None:
        forward = mne.read_forward_solution(forward_path, verbose=False)
        logger.info("Loaded forward model for REST from %s", forward_path)
    else:
        logger.info(
            "No forward_path provided; computing sphere-model forward for REST "
            "(consider pre-computing and saving to avoid this overhead)"
        )
        sphere = mne.make_sphere_model("auto", "auto", epoch.info)
        src = mne.setup_volume_source_space(sphere=sphere, exclude=30.0, pos=15.0)
        forward = mne.make_forward_solution(epoch.info, trans=None, src=src, bem=sphere)

    epoch_rest = epoch.copy()
    epoch_rest.set_eeg_reference("REST", forward=forward, verbose=False)
    return epoch_rest


def apply_epoch_baseline_correction(
    epoch: mne.Epochs,
    baseline_window: Optional[Tuple[Optional[float], Optional[float]]] = None,
) -> mne.Epochs:
    """Subtract the mean of the baseline period from each channel.

    Parameters
    ----------
    epoch : mne.Epochs
        Epoch data.
    baseline_window : (tmin, tmax) in seconds, optional
        ``None`` for either endpoint means "epoch start" or "epoch end".
        Defaults to ``(None, 0.0)`` — the entire pre-stimulus period.
    """
    if baseline_window is None:
        baseline_window = (None, 0.0)
    epoch_bl = epoch.copy()
    epoch_bl.apply_baseline(baseline=baseline_window, verbose=False)
    return epoch_bl


def flag_bad_epoch(
    epoch: mne.Epochs,
    amp_thresh: float = 200e-6,
    baseline_window: Tuple[float, float] = (-1.0, -0.7),
    erp_window: Tuple[float, float] = (0.1, 0.65),
    n250_channels: Sequence[str] = ("T7", "T8", "P7", "P8", "O1", "O2"),
    p300_channels: Sequence[str] = ("Cz", "Pz"),
    rejection_threshold: float = 0.5,
) -> Tuple[bool, List[str]]:
    """Check a single epoch for excessive artifacts.

    A channel is *bad* if its peak-to-peak amplitude in the baseline window
    **or** the ERP window exceeds *amp_thresh*.  The epoch is *rejected* if
    more than *rejection_threshold* fraction of either the N250 or P300
    channel group is bad.

    Parameters
    ----------
    epoch : mne.Epochs
        Baseline-corrected, single-epoch object.
    amp_thresh : float
        Peak-to-peak threshold in **volts** (default 200 µV).
    baseline_window, erp_window : (tmin, tmax) in seconds
        Windows used for peak-to-peak calculation.  Automatically clamped to
        the epoch's actual time range.
    n250_channels, p300_channels : sequence of str
        Channel groups whose bad-channel fraction determines rejection.
    rejection_threshold : float
        Fraction of channels in a group that must be bad to trigger rejection.

    Returns
    -------
    (is_rejected, bad_channels)
        is_rejected  : True if the epoch should be treated as unfamiliar/rejected.
        bad_channels : names of channels exceeding the amplitude threshold.
    """
    ch_names = epoch.ch_names
    n250_set = {c.upper() for c in n250_channels}
    p300_set = {c.upper() for c in p300_channels}

    n250_idx = [i for i, ch in enumerate(ch_names) if ch.upper() in n250_set]
    p300_idx = [i for i, ch in enumerate(ch_names) if ch.upper() in p300_set]

    X = epoch.get_data()  # (1, n_channels, n_times) for a single epoch

    # Clamp detection windows to the actual epoch boundaries
    t_min, t_max = float(epoch.tmin), float(epoch.tmax)
    bl = (max(baseline_window[0], t_min), min(baseline_window[1], t_max))
    erp = (max(erp_window[0], t_min), min(erp_window[1], t_max))

    i0b, i1b = epoch.time_as_index([bl[0], bl[1]])
    i0e, i1e = epoch.time_as_index([erp[0], erp[1]])

    bad_channels: List[str] = []
    for ch_i, ch in enumerate(ch_names):
        ptp_bl = float(np.ptp(X[0, ch_i, i0b : i1b + 1]))
        ptp_erp = float(np.ptp(X[0, ch_i, i0e : i1e + 1]))
        if ptp_bl > amp_thresh or ptp_erp > amp_thresh:
            bad_channels.append(ch)

    bad_set = {ch.upper() for ch in bad_channels}

    n250_bad = sum(1 for i in n250_idx if ch_names[i].upper() in bad_set)
    p300_bad = sum(1 for i in p300_idx if ch_names[i].upper() in bad_set)

    n250_ratio = n250_bad / len(n250_idx) if n250_idx else 0.0
    p300_ratio = p300_bad / len(p300_idx) if p300_idx else 0.0

    is_rejected = n250_ratio > rejection_threshold or p300_ratio > rejection_threshold

    if bad_channels:
        logger.info(
            "Epoch quality: bad_channels=%s n250=%d/%d p300=%d/%d rejected=%s",
            bad_channels,
            n250_bad, len(n250_idx),
            p300_bad, len(p300_idx),
            is_rejected,
        )

    return is_rejected, bad_channels


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_epoch_features(
    epochs: mne.Epochs,
    ch_windows: Dict[str, Tuple[float, float]],
    *,
    bad_channels_per_epoch: Optional[Sequence[Sequence[str]]] = None,
) -> pd.DataFrame:
    """Compute per-channel statistical features over specified time windows.

    Adapted from EEG_Data_Analysis_2.ipynb.  Supports one or more epochs.
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

    if bad_channels_per_epoch is None:
        bad_channels_per_epoch = [[] for _ in range(X.shape[0])]
    elif len(bad_channels_per_epoch) != X.shape[0]:
        raise ValueError(
            "bad_channels_per_epoch must have one entry per epoch: "
            f"got {len(bad_channels_per_epoch)} for {X.shape[0]} epoch(s)"
        )

    rows = []
    for epoch in range(X.shape[0]):
        epoch_bad_channels = {channel.upper() for channel in bad_channels_per_epoch[epoch]}
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

        n250_good = [ch for ch in n250_present if ch.upper() not in epoch_bad_channels]
        n250_bad = [ch for ch in n250_present if ch.upper() in epoch_bad_channels]
        p300_good = [ch for ch in p300_present if ch.upper() not in epoch_bad_channels]
        p300_bad = [ch for ch in p300_present if ch.upper() in epoch_bad_channels]

        # Compute averaged features for N250 using only non-bad channels, then
        # impute bad or absent channels from the component average.
        if n250_present or n250_absent:
            for s in stat_order:
                values = [row[f"{ch}_{s}"] for ch in n250_good if f"{ch}_{s}" in row]
                if values:
                    component_mean = float(np.nanmean(values))
                    row[f"N250avg_{s}"] = component_mean
                    for ch in n250_bad + n250_absent:
                        row[f"{ch}_{s}"] = component_mean
                else:
                    row[f"N250avg_{s}"] = float('nan')
                    for ch in n250_bad + n250_absent:
                        row[f"{ch}_{s}"] = float('nan')

        # Compute averaged features for P300 using only non-bad channels, then
        # impute bad or absent channels from the component average.
        if p300_present or p300_absent:
            for s in stat_order:
                values = [row[f"{ch}_{s}"] for ch in p300_good if f"{ch}_{s}" in row]
                if values:
                    component_mean = float(np.nanmean(values))
                    row[f"P300avg_{s}"] = component_mean
                    for ch in p300_bad + p300_absent:
                        row[f"{ch}_{s}"] = component_mean
                else:
                    row[f"P300avg_{s}"] = float('nan')
                    for ch in p300_bad + p300_absent:
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
    # --- filtering ---
    l_freq: float = 1.0,
    h_freq: float = 40.0,
    notch_freqs: Optional[List[float]] = None,
    # --- ICA ---
    ica_path: Optional[str] = None,
    # --- REST re-referencing ---
    apply_rest: bool = False,
    forward_path: Optional[str] = None,
    # --- baseline correction ---
    baseline_window: Optional[Tuple[Optional[float], Optional[float]]] = None,
    # --- artifact rejection ---
    amp_thresh: float = 200e-6,
    # --- feature extraction ---
    ch_windows: Optional[Dict[str, Tuple[float, float]]] = None,
) -> np.ndarray:
    """Full single-trial EEG preprocessing pipeline returning a 1D feature vector.

    Steps
    -----
    1. Bandpass + notch filter
    2. ICA artefact removal (if *ica_path* is provided)
    3. REST re-referencing (if *apply_rest* is True)
    4. Baseline correction
    5. Artifact quality check → raises :class:`EpochRejectedError` if noisy
    6. Statistical ERP feature extraction

    Parameters
    ----------
    epoch_eeg_data : mne.Epochs
        Single-epoch MNE object (len == 1).
    l_freq, h_freq : float
        Bandpass filter bounds in Hz.
    notch_freqs : list of float, optional
        Notch filter frequencies.  Defaults to [60.0, 120.0].
    ica_path : str, optional
        Path to a saved MNE ICA .fif file.  If *None*, ICA is skipped.
    apply_rest : bool
        Whether to apply REST re-referencing.
    forward_path : str, optional
        Path to a pre-computed forward solution for REST.  Ignored when
        *apply_rest* is False.
    baseline_window : (tmin, tmax), optional
        Baseline correction window in seconds.  Defaults to (None, 0.0).
    amp_thresh : float
        Peak-to-peak artifact threshold in **volts** (default 200 µV).
    ch_windows : dict, optional
        Mapping of channel name → (tmin, tmax) for feature extraction.

    Returns
    -------
    features : np.ndarray, shape (n_features,)
        1D statistical feature vector (same column order as training CSV).

    Raises
    ------
    EpochRejectedError
        If the epoch fails artifact quality checks.  The pipeline treats this
        as *is_unfamiliar=True* and logs it separately.
    """
    epoch = epoch_eeg_data

    # 1. Filter
    epoch = filter_epoch(epoch, l_freq=l_freq, h_freq=h_freq, notch_freqs=notch_freqs)

    # 2. ICA artefact removal (optional)
    if ica_path is not None:
        epoch = apply_ica_to_epoch(epoch, ica_path)

    # 3. REST re-referencing (optional)
    if apply_rest:
        epoch = apply_rest_reference_to_epoch(epoch, forward_path=forward_path)

    # 4. Baseline correction
    epoch = apply_epoch_baseline_correction(epoch, baseline_window=baseline_window)

    # 5. Artifact quality check
    is_rejected, bad_channels = flag_bad_epoch(epoch, amp_thresh=amp_thresh)
    if is_rejected:
        raise EpochRejectedError(
            f"Epoch rejected: {len(bad_channels)} bad channel(s) — {bad_channels}",
            bad_channels=bad_channels,
        )

    # 6. Feature extraction
    if ch_windows is None:
        win_n250 = (0.200, 0.300)
        win_p300 = (0.250, 0.350)
        ch_windows = {
            "Cz": win_p300,
            "Pz": win_p300,
            "P7": win_n250,
            "P8": win_n250,
            "O1": win_n250,
            "O2": win_n250,
        }

    df = extract_epoch_features(
        epoch,
        ch_windows,
        bad_channels_per_epoch=[bad_channels],
    )
    feature_cols = [c for c in df.columns if c != "condition"]
    feats = df.loc[0, feature_cols].to_numpy(dtype=float, copy=False)
    return feats
