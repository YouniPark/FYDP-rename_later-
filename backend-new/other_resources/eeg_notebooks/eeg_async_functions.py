"""
Functions from the asynchronous EEG processing notebook. To be adapted to the backend. 
These used multi-trial continuous EEG data, which was epoched after the filter. 
The backend use case is single-trial/single-epoch from the start.
"""
import mne
import numpy as np

def prepare_eeg_data(eeg_df, ch_names):
    """
    Prepare raw EEG data from DataFrame and create MNE Raw object.
    
    Parameters
    ----------
    eeg_df : pd.DataFrame
        DataFrame containing EEG data with columns for each channel and TIMESTAMP, SAMPLING_RATE
    ch_names : list of str
        Desired channel names in order (e.g., ["Cz", "Pz", "F7", "F8", "P7", "P8", "O1", "O2"])
    
    Returns
    -------
    raw : mne.io.Raw
        MNE RawArray object with montage set
    """
    print("="*60)
    print("Preparing EEG Data")
    print("="*60)
    
    # Current EEG columns
    meta_cols = {"TIMESTAMP", "SAMPLING_RATE"}
    eeg_cols_current = [c for c in eeg_df.columns if c not in meta_cols]
    
    if len(eeg_cols_current) != len(ch_names):
        print(
            f"Warning: expected {len(ch_names)} EEG channels, found {len(eeg_cols_current)}: {eeg_cols_current}"
        )
    
    # Prepare numeric data matrix (n_channels x n_times) for MNE RawArray
    # Coerce to numeric and replace NaNs with 0 for stability
    eeg_numeric = eeg_df[eeg_cols_current].apply(pd.to_numeric, errors='coerce')
    data = eeg_numeric.to_numpy().T.astype(float, copy=False)
    np.nan_to_num(data, copy=False)
    
    # Convert from microvolts to volts (MNE expects volts)
    data = data * 1e-6
    
    # Sampling frequency
    sfreq = float(eeg_df["SAMPLING_RATE"].iloc[0]) if "SAMPLING_RATE" in eeg_df.columns else None
    
    # Create MNE Info and RawArray using desired channel labels (truncate in case of mismatch)
    desired_ch_names = ch_names[: data.shape[0]]
    info = mne.create_info(ch_names=desired_ch_names, sfreq=sfreq, ch_types='eeg')
    raw = mne.io.RawArray(data, info, verbose=False)
    
    # Apply a standard montage that includes these channels
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, match_case=False, on_missing='ignore')
    
    print(raw)
    print("Channels:", raw.ch_names)
    print("Sampling rate:", raw.info['sfreq'])
    print("Data range (V):", f"{data.min():.2e} to {data.max():.2e}")
    print()
    
    return raw


def filter_continuous_eeg(raw, l_freq=1.0, h_freq=40.0, notch_freqs=[60.0, 120.0], show_plots=True):
    """
    Apply bandpass and notch filters to continuous EEG data.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data
    l_freq : float
        Low-pass frequency for bandpass filter (Hz)
    h_freq : float
        High-pass frequency for bandpass filter (Hz)
    notch_freqs : list of float
        Frequencies to notch filter (Hz) - typically power line noise
    
    Returns
    -------
    raw_filt : mne.io.Raw
        Filtered raw EEG data
    """
    print("="*60)
    print("Filtering Continuous EEG Data")
    print("="*60)
    
    raw_filt = raw.copy().load_data()
    
    # Bandpass filter
    print(f"Applying bandpass filter: {l_freq}-{h_freq} Hz")
    raw_filt.filter(l_freq, h_freq)
    
    # Notch filter for power line noise
    if notch_freqs:
        print(f"Applying notch filter at: {notch_freqs} Hz")
        raw_filt.notch_filter(freqs=notch_freqs)
    
    print(f"Filtered data: {raw_filt}")
    print()
    
    return raw_filt


# NEED TO UPDATE BACKEND TO LOAD ICA AND EXCLUDE LIST BEFORE CALLING THIS FUNCTION
def apply_ica(epochs, ica):
    """
    Apply ICA artifact removal to epochs.
    
    Parameters
    ----------
    epochs : mne.Epochs
        Epoched EEG data (original, not temporarily filtered)
    ica : mne.preprocessing.ICA
        Fitted ICA object with exclude list set
    
    Returns
    -------
    epochs_clean : mne.Epochs
        Epochs with ICA artifacts removed
    """
    print("="*60)
    print("Applying ICA to Remove Artifacts")
    print("="*60)
    
    print(f"Excluding components: {ica.exclude}")
    if not ica.exclude:
        print("WARNING: No components marked for exclusion!")
    
    # Apply ICA to the original (non-temporarily-filtered) epochs
    print("Applying ICA to remove artifacts from original epochs...")
    epochs_clean = epochs.copy()
    ica.apply(epochs_clean)
    
    print(f"ICA artifact removal complete")
    print()
    
    return epochs_clean


# UPDATE TO LOAD FORWARD MODEL FROM A PREVIOUS DATA COLLECTION SESSION
def apply_rest_reference(epochs, forward=None):
    """
    Apply REST (Reference Electrode Standardization Technique) re-referencing.
    
    REST is an infinite reference technique that estimates the potential at infinity
    using a forward model and spherical spline interpolation.
    
    Parameters
    ----------
    epochs : mne.Epochs
        Epoched EEG data with montage set
    forward : mne.Forward, optional
        Forward solution for REST. If None, will be computed automatically.
    
    Returns
    -------
    epochs_rest : mne.Epochs
        Epochs re-referenced to REST
    forward : mne.Forward
        Forward solution used for REST
    """
    print("="*60)
    print("Re-referencing to REST (Infinite Reference)")
    print("="*60)
    
    # Check if montage is set
    if epochs.get_montage() is None:
        raise ValueError("Montage must be set before applying REST. Use epochs.set_montage()")
    
    print("Applying REST re-referencing...")
    
    # Create forward model if not provided (see mne.set_eeg_reference docs)
    if forward is None:
        print("Creating forward model for REST...")
        sphere = mne.make_sphere_model("auto", "auto", epochs.info)
        src = mne.setup_volume_source_space(sphere=sphere, exclude=30.0, pos=15.0)
        forward = mne.make_forward_solution(epochs.info, trans=None, src=src, bem=sphere)
        print("Forward model created")
    
    # Apply REST reference
    epochs_rest = epochs.copy()
    epochs_rest.set_eeg_reference('REST', forward=forward)
    
    print("REST re-referencing complete")
    print()
    
    return epochs_rest, forward


def apply_baseline_correction(epochs, baseline_window=None):
    """
    Apply baseline correction by subtracting the mean of the baseline period.
    
    Parameters
    ----------
    epochs : mne.Epochs
        Epoched EEG data
    baseline_window : tuple of float, optional
        Baseline time window (tmin, tmax) in seconds.
        If None, uses the entire pre-stimulus period.
    
    Returns
    -------
    epochs_baseline : mne.Epochs
        Baseline-corrected epochs
    """
    print("="*60)
    print("Baseline Correction")
    print("="*60)
    
    if baseline_window is None:
        baseline_window = (None, 0)
        print("Applying baseline correction using entire pre-stimulus period")
    else:
        print(f"Applying baseline correction using window: {baseline_window[0]} to {baseline_window[1]} s")
    
    epochs_baseline = epochs.copy()
    epochs_baseline.apply_baseline(baseline=baseline_window)
    
    print("Baseline correction complete")
    print()
    
    return epochs_baseline


# IN BACKEND, A "REJECTED" TRIAL SHOULD BE CONSIDERED UNFAMILIAR (BUT LOG THE DIFFERENCE)
def reject_bad_trials(epochs, amp_thresh=200e-6, baseline_window=[-1, -0.7], 
                      erp_window=[0.1, 0.65], n250_channels=['T7', 'T8', 'P7', 'P8', 'O1', 'O2'],
                      p300_channels=['Cz', 'Pz'], rejection_threshold=0.5):
    """
    Reject trials based on peak-to-peak amplitude in baseline and ERP windows.
    
    If more than half of the channels in a component (N250 or P300) are bad,
    reject the whole trial. Otherwise, keep the trial and mark bad channels for later
    substitution during feature extraction.
    
    Parameters
    ----------
    epochs : mne.Epochs
        Epoched EEG data
    amp_thresh : float
        Peak-to-peak amplitude threshold in volts
    baseline_window : list of float
        Baseline time window [tmin, tmax] in seconds
    erp_window : list of float
        ERP time window [tmin, tmax] in seconds
    n250_channels : list of str
        Channels in N250 component
    p300_channels : list of str
        Channels in P300 component
    rejection_threshold : float
        Fraction of channels that must be bad to reject trial (default 0.5 = 50%)
    
    Returns
    -------
    epochs_clean : mne.Epochs
        Epochs with bad trials removed
    bad_channels_per_trial : dict
        Dictionary mapping trial index to list of bad channels
    """
    print("="*60)
    print("Rejecting and Substituting Channels per Trial")
    print("="*60)
    
    print(f"Rejection threshold: {amp_thresh*1e6:.0f} µV")
    print(f"Baseline window: {baseline_window}")
    print(f"ERP window: {erp_window}")
    print(f"Trial rejection if >{rejection_threshold*100:.0f}% of component channels are bad")
    print()
    
    # Get epoch labels
    inv_event_id = {v: k for k, v in epochs.event_id.items()}
    labels = [inv_event_id.get(code, str(code)) for code in epochs.events[:, 2]]
    
    X = epochs.get_data()  # (n_epochs, n_channels, n_times)
    ch_names = epochs.ch_names
    
    # Map channel names (case-insensitive)
    upper_map = {ch.upper(): i for i, ch in enumerate(ch_names)}
    
    # Get indices for component channels
    n250_idx = [upper_map[c.upper()] for c in n250_channels if c.upper() in upper_map]
    p300_idx = [upper_map[c.upper()] for c in p300_channels if c.upper() in upper_map]
    
    print(f"N250 channels ({len(n250_idx)}): {[ch_names[i] for i in n250_idx]}")
    print(f"P300 channels ({len(p300_idx)}): {[ch_names[i] for i in p300_idx]}")
    print()
    
    # Get time indices for windows
    i0b, i1b = epochs.time_as_index(baseline_window)
    i0b, i1b = (i0b, i1b) if i0b <= i1b else (i1b, i0b)
    
    i0e, i1e = epochs.time_as_index(erp_window)
    i0e, i1e = (i0e, i1e) if i0e <= i1e else (i1e, i0e)
    
    # Compute peak-to-peak per channel in each window
    ptp_baseline = np.ptp(X[:, :, i0b:i1b + 1], axis=2)  # (n_epochs, n_channels)
    ptp_erp = np.ptp(X[:, :, i0e:i1e + 1], axis=2)
    
    # Identify bad channels per trial
    bad_mask = (ptp_baseline > amp_thresh) | (ptp_erp > amp_thresh)  # (n_epochs, n_channels)
    
    # Determine which trials to reject based on component-wise thresholds
    trials_to_reject = []
    bad_channels_per_trial = {}
    
    for epoch_idx in range(len(epochs)):
        bad_ch_idx = np.flatnonzero(bad_mask[epoch_idx])
        bad_ch_names = [ch_names[i] for i in bad_ch_idx]
        
        if len(bad_ch_idx) > 0:
            bad_channels_per_trial[epoch_idx] = bad_ch_names
        
        # Check N250 component
        n250_bad = sum(1 for i in n250_idx if bad_mask[epoch_idx, i])
        n250_bad_ratio = n250_bad / len(n250_idx) if len(n250_idx) > 0 else 0
        
        # Check P300 component
        p300_bad = sum(1 for i in p300_idx if bad_mask[epoch_idx, i])
        p300_bad_ratio = p300_bad / len(p300_idx) if len(p300_idx) > 0 else 0
        
        # Reject if either component exceeds threshold
        if n250_bad_ratio > rejection_threshold or p300_bad_ratio > rejection_threshold:
            trials_to_reject.append(epoch_idx)
    
    print(f"Trials to reject: {len(trials_to_reject)} of {len(epochs)}")
    print(f"Trials with bad channels (but not rejected): {len(bad_channels_per_trial) - len(trials_to_reject)}")
    
    # Report details of rejected trials
    if trials_to_reject:
        print("\nRejected trials:")
        for idx in trials_to_reject:
            label = labels[idx] if idx < len(labels) else "unknown"
            bad_chs = bad_channels_per_trial.get(idx, [])
            
            n250_bad = sum(1 for ch in bad_chs if ch.upper() in [c.upper() for c in n250_channels])
            p300_bad = sum(1 for ch in bad_chs if ch.upper() in [c.upper() for c in p300_channels])
            
            print(f"  Trial {idx} ({label}): {len(bad_chs)} bad channels | "
                  f"N250: {n250_bad}/{len(n250_idx)} | P300: {p300_bad}/{len(p300_idx)}")
    
    # Count bad channels across all trials
    channel_bad_count = {}
    for trial_idx, bad_chs in bad_channels_per_trial.items():
        for ch in bad_chs:
            channel_bad_count[ch] = channel_bad_count.get(ch, 0) + 1
    
    # Print per-channel bad channel statistics (omitting channels with zero)
    if channel_bad_count:
        print("\nBad channels across trials:")
        for ch in sorted(channel_bad_count.keys()):
            count = channel_bad_count[ch]
            if count > 0:
                print(f"  {ch}: {count} trials")
    
    # Drop bad epochs
    epochs_clean = epochs.copy()
    epochs_clean.drop(trials_to_reject)
    
    print(f"\nFinal: {len(epochs_clean)} good trials remaining")
    print()
    
    return epochs_clean, bad_channels_per_trial



## Epoch feature extraction function 
def extract_epoch_features(epochs, ch_windows):
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
                    'auc': float(np.trapezoid(x, dx=1.0 / sfreq)),  # integrate using trapezoidal rule
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
                        row[f"{ch}_{s}"] = float(np.nanmean(values)) # impute missing N250 channels with average
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
                        row[f"{ch}_{s}"] = float(np.nanmean(values)) # impute missing P300 channels with average
                else:
                    row[f"P300avg_{s}"] = float('nan')
                    for ch in p300_absent:
                        row[f"{ch}_{s}"] = float('nan')
        
        row['condition'] = conditions[epoch] if epoch < len(conditions) else None
        rows.append(row)

    df = pd.DataFrame(rows)

    # Order columns by channel then stat, condition first
    preferred_ch_order = ['Cz','Pz', 'P7', 'P8', 'O1','O2']
    # Build ordered columns following preferred channel order
    ordered_cols = []
    
    # Add individual channel columns in preferred order
    for ch in preferred_ch_order:
        if ch in selected.keys() or ch in n250_absent or ch in p300_absent:
            ordered_cols.extend([f"{ch}_{s}" for s in stat_order])
    
    # Add averaged N250 and P300 columns at the end
    if n250_present:
        ordered_cols.extend([f"N250avg_{s}" for s in stat_order])
    if p300_present:
        ordered_cols.extend([f"P300avg_{s}" for s in stat_order])

    cols = ['condition'] + [c for c in ordered_cols if c in df.columns]
    df = df[cols]
    return df