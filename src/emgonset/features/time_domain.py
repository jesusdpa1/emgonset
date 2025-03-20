"""
/src/emgonset/features/time_domain.py
Time-domain feature extraction functions for EMG signals
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from ..detection.onset import OnsetOffsetPoint, OnsetOffsetResult
from ..utils.internals import public_api


@public_api
def extract_segment_features(
    signal: Union[np.ndarray, torch.Tensor],
    onset: OnsetOffsetPoint,
    offset: OnsetOffsetPoint,
    fs: float,
    feature_set: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Extract time-domain features from a single EMG segment between onset and offset

    Args:
        signal: EMG signal (1D array/tensor)
        onset: Onset point indicating start of segment
        offset: Offset point indicating end of segment
        fs: Sampling frequency in Hz
        feature_set: List of features to extract. If None, extracts all features.
                    Options: 'mav', 'rms', 'wl', 'zc', 'ssc', 'iemg', 'var', 'duration'

    Returns:
        Dictionary of extracted features
    """
    # Convert to numpy if tensor
    if isinstance(signal, torch.Tensor):
        signal_np = signal.detach().cpu().numpy()
    else:
        signal_np = signal

    # Ensure 1D array
    if signal_np.ndim > 1:
        raise ValueError("Expected 1D signal for feature extraction")

    # Extract segment
    start_idx = onset.sample_index
    end_idx = offset.sample_index
    segment = signal_np[start_idx : end_idx + 1]

    # Default feature set if none provided
    if feature_set is None:
        feature_set = ["mav", "rms", "wl", "zc", "ssc", "iemg", "var", "duration"]

    features = {}

    # Calculate segment duration in seconds
    duration = (end_idx - start_idx) / fs
    if "duration" in feature_set:
        features["duration"] = duration

    # Mean Absolute Value (MAV)
    if "mav" in feature_set:
        features["mav"] = np.mean(np.abs(segment))

    # Root Mean Square (RMS)
    if "rms" in feature_set:
        features["rms"] = np.sqrt(np.mean(np.square(segment)))

    # Waveform Length (WL)
    if "wl" in feature_set:
        features["wl"] = np.sum(np.abs(np.diff(segment)))

    # Zero Crossings (ZC) with threshold to reduce noise sensitivity
    if "zc" in feature_set:
        # Calculate threshold as 10% of RMS
        threshold = 0.1 * np.sqrt(np.mean(np.square(segment)))

        # Count zero crossings with threshold
        zc_count = 0
        for i in range(1, len(segment)):
            if (
                (segment[i] > 0 and segment[i - 1] < 0)
                or (segment[i] < 0 and segment[i - 1] > 0)
            ) and (abs(segment[i] - segment[i - 1]) >= threshold):
                zc_count += 1

        features["zc"] = zc_count

        # Normalize by segment duration for rate
        features["zc_rate"] = zc_count / duration if duration > 0 else 0

    # Slope Sign Changes (SSC) with threshold
    if "ssc" in feature_set:
        # Calculate threshold as 10% of RMS
        threshold = 0.1 * np.sqrt(np.mean(np.square(segment)))

        # Count slope sign changes with threshold
        ssc_count = 0
        for i in range(1, len(segment) - 1):
            if (
                (segment[i] > segment[i - 1] and segment[i] > segment[i + 1])
                or (segment[i] < segment[i - 1] and segment[i] < segment[i + 1])
            ) and (
                abs(segment[i] - segment[i - 1]) >= threshold
                or abs(segment[i] - segment[i + 1]) >= threshold
            ):
                ssc_count += 1

        features["ssc"] = ssc_count

        # Normalize by segment duration for rate
        features["ssc_rate"] = ssc_count / duration if duration > 0 else 0

    # Integrated EMG (IEMG)
    if "iemg" in feature_set:
        features["iemg"] = np.sum(np.abs(segment))

    # Variance (VAR)
    if "var" in feature_set:
        features["var"] = np.var(segment)

    # Maximum amplitude
    if "max_amp" in feature_set:
        features["max_amp"] = np.max(np.abs(segment))

    # Mean frequency (approximate using zero crossings)
    if "mean_freq" in feature_set:
        # Use zero crossing rate as approximation
        zc_count = 0
        for i in range(1, len(segment)):
            if (segment[i] > 0 and segment[i - 1] < 0) or (
                segment[i] < 0 and segment[i - 1] > 0
            ):
                zc_count += 1

        # Each zero crossing represents half a cycle
        features["mean_freq"] = (zc_count / 2) / duration if duration > 0 else 0

    return features


@public_api
def extract_all_segments_features(
    signal: Union[np.ndarray, torch.Tensor],
    detection_result: OnsetOffsetResult,
    fs: float,
    feature_set: Optional[List[str]] = None,
) -> List[Dict[str, float]]:
    """
    Extract features from all detected segments in a signal

    Args:
        signal: EMG signal (1D array/tensor)
        detection_result: Onset/offset detection result
        fs: Sampling frequency in Hz
        feature_set: List of features to extract

    Returns:
        List of feature dictionaries, one for each segment
    """
    # Ensure we have matching onset-offset pairs
    if len(detection_result.onsets) != len(detection_result.offsets):
        raise ValueError("Number of onsets and offsets must match")

    # Extract features for each segment
    all_features = []

    for onset, offset in zip(detection_result.onsets, detection_result.offsets):
        segment_features = extract_segment_features(
            signal=signal, onset=onset, offset=offset, fs=fs, feature_set=feature_set
        )

        # Add onset/offset time information
        segment_features["onset_time"] = onset.time_sec
        segment_features["offset_time"] = offset.time_sec

        all_features.append(segment_features)

    return all_features


@public_api
def extract_windowed_features(
    signal: Union[np.ndarray, torch.Tensor],
    fs: float,
    window_size_ms: float = 200.0,
    window_step_ms: float = 100.0,
    feature_set: Optional[List[str]] = None,
) -> Tuple[List[Dict[str, float]], np.ndarray]:
    """
    Extract features using a sliding window approach

    Args:
        signal: EMG signal (1D array/tensor)
        fs: Sampling frequency in Hz
        window_size_ms: Window size in milliseconds
        window_step_ms: Window step size in milliseconds
        feature_set: List of features to extract

    Returns:
        Tuple of (list of feature dictionaries, array of window centers in seconds)
    """
    # Convert to numpy if tensor
    if isinstance(signal, torch.Tensor):
        signal_np = signal.detach().cpu().numpy()
    else:
        signal_np = signal

    # Ensure 1D array
    if signal_np.ndim > 1:
        raise ValueError("Expected 1D signal for feature extraction")

    # Calculate window parameters in samples
    window_size = int(window_size_ms * fs / 1000)
    window_step = int(window_step_ms * fs / 1000)

    # Create sliding windows
    n_windows = max(1, (len(signal_np) - window_size) // window_step + 1)

    features_list = []
    window_centers = np.zeros(n_windows)

    for i in range(n_windows):
        start_idx = i * window_step
        end_idx = start_idx + window_size

        # Create artificial onset/offset points
        onset = OnsetOffsetPoint(
            sample_index=start_idx,
            time_sec=start_idx / fs,
            amplitude=signal_np[start_idx],
        )

        offset = OnsetOffsetPoint(
            sample_index=end_idx - 1,
            time_sec=(end_idx - 1) / fs,
            amplitude=signal_np[min(end_idx - 1, len(signal_np) - 1)],
        )

        # Extract features
        window_features = extract_segment_features(
            signal=signal_np, onset=onset, offset=offset, fs=fs, feature_set=feature_set
        )

        # Record window center time
        window_center = (start_idx + end_idx / 2) / fs
        window_centers[i] = window_center
        window_features["center_time"] = window_center

        features_list.append(window_features)

    return features_list, window_centers


@public_api
def features_to_dataframe(features_list: List[Dict[str, float]]):
    """
    Convert list of feature dictionaries to a pandas DataFrame

    Args:
        features_list: List of feature dictionaries

    Returns:
        pandas DataFrame containing all features
    """
    try:
        import pandas as pd

        return pd.DataFrame(features_list)
    except ImportError:
        raise ImportError("pandas is required for this function but not installed")


@public_api
def extract_multi_channel_features(
    signals: Union[np.ndarray, torch.Tensor],
    detection_results: List[OnsetOffsetResult],
    fs: float,
    channel_names: Optional[List[str]] = None,
    feature_set: Optional[List[str]] = None,
) -> Dict[str, List[Dict[str, float]]]:
    """
    Extract features from multiple channels

    Args:
        signals: Multi-channel EMG signals [channels, samples]
        detection_results: List of onset/offset detection results for each channel
        fs: Sampling frequency in Hz
        channel_names: Optional list of channel names
        feature_set: List of features to extract

    Returns:
        Dictionary mapping channel names to lists of feature dictionaries
    """
    # Convert to numpy if tensor
    if isinstance(signals, torch.Tensor):
        signals_np = signals.detach().cpu().numpy()
    else:
        signals_np = signals

    # Get number of channels
    n_channels = signals_np.shape[0]

    # Create default channel names if not provided
    if channel_names is None:
        channel_names = [f"channel_{i + 1}" for i in range(n_channels)]

    # Ensure we have detection results for each channel
    if len(detection_results) != n_channels:
        raise ValueError(
            f"Number of detection results ({len(detection_results)}) does not match number of channels ({n_channels})"
        )

    # Extract features for each channel
    channel_features = {}

    for i in range(n_channels):
        channel_name = channel_names[i]

        features = extract_all_segments_features(
            signal=signals_np[i],
            detection_result=detection_results[i],
            fs=fs,
            feature_set=feature_set,
        )

        channel_features[channel_name] = features

    return channel_features


@public_api
def aggregate_features_across_channels(
    multi_channel_features: Dict[str, List[Dict[str, float]]],
) -> List[Dict[str, float]]:
    """
    Aggregate features across channels for each aligned segment

    NOTE: This assumes segments are aligned or already matched across channels

    Args:
        multi_channel_features: Dictionary mapping channel names to feature lists

    Returns:
        List of aggregated feature dictionaries
    """
    # Get channel names
    channel_names = list(multi_channel_features.keys())

    if not channel_names:
        return []

    # Get number of segments in first channel
    n_segments = len(multi_channel_features[channel_names[0]])

    # Ensure all channels have the same number of segments
    for channel in channel_names[1:]:
        if len(multi_channel_features[channel]) != n_segments:
            raise ValueError(
                f"Channels have different numbers of segments. Channel alignment may be required."
            )

    # Aggregate features across channels
    aggregated_features = []

    for i in range(n_segments):
        # Start with timing information from first channel
        segment_dict = {
            "onset_time": multi_channel_features[channel_names[0]][i].get(
                "onset_time", 0
            ),
            "offset_time": multi_channel_features[channel_names[0]][i].get(
                "offset_time", 0
            ),
            "duration": multi_channel_features[channel_names[0]][i].get("duration", 0),
        }

        # Aggregate each feature across channels
        for channel in channel_names:
            channel_dict = multi_channel_features[channel][i]

            for feature, value in channel_dict.items():
                # Skip timing information which we already added
                if feature in ("onset_time", "offset_time", "duration"):
                    continue

                # Create feature name with channel prefix
                channel_feature = f"{channel}_{feature}"
                segment_dict[channel_feature] = value

        aggregated_features.append(segment_dict)

    return aggregated_features
