import pyxdf
import numpy as np
import pandas as pd


def infer_shape(ts):
    """Safely infer shape for list/array time series.
    Returns a tuple like (n_samples,) or (n_samples, n_channels)."""
    try:
        arr = np.asarray(ts, dtype=object)
        if arr.ndim == 2 and arr.dtype != object:
            return arr.shape
        # list-of-lists or ragged object array
        if len(arr) == 0:
            return (0,)
        first = arr[0]
        if isinstance(first, (list, tuple, np.ndarray)):
            try:
                return (len(arr), len(first))
            except Exception:
                return (len(arr),)
        return (len(arr),)
    except Exception:
        try:
            return (len(ts),)
        except Exception:
            return tuple()


def to_2d_numeric(ts):
    """Convert time series to a 2D numpy array when possible.
    - If 1D -> (N,1)
    - If list-of-lists -> (N,M)
    - Falls back to object dtype if needed.
    """
    try:
        arr = np.asarray(ts)
        if arr.ndim == 1:
            return arr.reshape(-1, 1)
        if arr.ndim >= 2:
            return arr
    except Exception:
        pass

    # Fallback: try building from rows
    try:
        rows = [np.asarray(row) for row in ts]
        max_len = max((r.size for r in rows), default=0)
        norm = np.array([np.pad(r, (0, max_len - r.size), mode='constant', constant_values=np.nan) for r in rows])
        return norm
    except Exception:
        # Last resort: object array reshaped to 2D
        return np.asarray(ts, dtype=object).reshape(-1, 1)


# Load the XDF file
data, header = pyxdf.load_xdf('sub-P001_ses-S003_task-Default_run-001_eeg.xdf')

# Iterate through all streams in the file
for i, stream in enumerate(data):
    print(f"\n{'='*60}")
    print(f"Stream {i + 1}:")
    print(f"{'='*60}")

    # Print stream info
    info = stream['info']
    print(f"Name: {info['name'][0]}")
    print(f"Type: {info['type'][0]}")
    print(f"Channel Count: {info['channel_count'][0]}")
    print(f"Nominal Sample Rate: {info['nominal_srate'][0]} Hz")
    print(f"Channel Format: {info['channel_format'][0]}")

    # Print channel names if available
    desc = info.get('desc', [])
    if desc and desc[0]:
        desc0 = desc[0]
        channels_container = desc0.get('channels', [])
        if channels_container and channels_container[0]:
            print("\nChannels:")
            for ch in channels_container[0].get('channel', []):
                if not ch:
                    continue
                label_val = ch.get('label', ['N/A'])
                unit_val = ch.get('unit', ['N/A'])
                type_val = ch.get('type', ['N/A'])
                label = label_val[0] if isinstance(label_val, list) and label_val else 'N/A'
                unit = unit_val[0] if isinstance(unit_val, list) and unit_val else 'N/A'
                ch_type = type_val[0] if isinstance(type_val, list) and type_val else 'N/A'
                print(f"  - {label} (Type: {ch_type}, Unit: {unit})")

    # Print data statistics
    time_series = stream.get('time_series', [])
    time_stamps = stream.get('time_stamps', [])

    shape = infer_shape(time_series)
    print(f"\nData shape: {shape}")

    # Duration if timestamps available
    if len(time_stamps) >= 2:
        try:
            duration = float(time_stamps[-1]) - float(time_stamps[0])
            print(f"Duration: {duration:.2f} seconds")
        except Exception:
            print("Duration: N/A")
    else:
        print("Duration: N/A")
    print(f"Number of samples: {len(time_stamps)}")

    # Print first few samples
    print(f"\nFirst 3 samples:")
    try:
        print(np.asarray(time_series, dtype=object)[:3])
    except Exception:
        print(time_series[:3] if isinstance(time_series, list) else 'N/A')
    print(f"\nFirst 3 timestamps:")
    try:
        print(np.asarray(time_stamps)[:3])
    except Exception:
        print(time_stamps[:3] if isinstance(time_stamps, list) else 'N/A')

    # Save Gaze2d stream to CSV
    if info['name'][0] == 'Gaze2d':
        arr = to_2d_numeric(time_series)
        ts = np.asarray(time_stamps)
        # Align lengths
        n = min(arr.shape[0], ts.shape[0])
        arr, ts = arr[:n], ts[:n]
        # Create columns
        n_cols = arr.shape[1] if arr.ndim >= 2 else 1
        columns = [f'Channel_{j}' for j in range(n_cols)]
        df = pd.DataFrame(arr, columns=columns)
        df.insert(0, 'Timestamp', ts)
        df.to_csv('Gaze2d.csv', index=False)
        print(f"\nSaved {info['name'][0]} to Gaze2d.csv")

    # Print all unique markers for Explore_84A1_Marker stream
    if info['name'][0] == 'Explore_84A1_Marker':
        try:
            markers = []
            for row in time_series:
                if isinstance(row, (list, tuple, np.ndarray)):
                    for elem in row:
                        markers.append(str(elem))
                else:
                    markers.append(str(row))
            unique_markers = sorted(set(markers))
            print("\nUnique markers (Explore_84A1_Marker):")
            for m in unique_markers:
                print(f"  {m}")
        except Exception as e:
            print(f"Could not extract unique markers: {e}")