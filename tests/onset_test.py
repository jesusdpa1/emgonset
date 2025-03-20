# %%
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from scipy import signal
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from emgonset.detection.onset import (
    create_adaptive_threshold_detector,
    create_fixed_threshold_detector,
    create_std_threshold_detector,
)
from emgonset.processing.envelope import create_lowpass_envelope, create_tkeo_envelope
from emgonset.processing.filters import (
    create_bandpass_filter,
    create_emg_filter,
    create_notch_filter,
)
from emgonset.processing.normalization import create_zscore_normalizer
from emgonset.processing.rectifiers import create_abs_rectifier
from emgonset.processing.tkeo import create_tkeo2
from emgonset.processing.transforms import EMGTransformCompose
from emgonset.utils.io import create_emg_dataloader  # Corrected import
from emgonset.visualization.detection_plots import (
    plot_multi_channel_onsets,
    plot_onset_detection,
)
from emgonset.visualization.general_plots import plot_time_series


# Function to run detection on a batch and visualize results
def process_and_visualize_batch(batch_idx, batch_data, detector_name, detector):
    """Process a single batch with the detector and visualize results"""
    # Get the raw processed signals
    processed_signals = batch_data  # Already processed by transform

    # For multi-channel data, visualize each channel
    n_channels = processed_signals.shape[1]

    # Create a figure for showing all channels with their detections
    detection_results = []

    # Process each channel
    for channel_idx in range(n_channels):
        # Get the signal for this channel
        signal_ = processed_signals[0, channel_idx]  # First dimension is batch

        # Apply detection
        detection_result = detector(signal_)
        detection_results.append(detection_result)

        # Print detection statistics
        print(f"Batch {batch_idx}, Channel {channel_idx + 1}:")
        print(f"  - Detected {len(detection_result.onsets)} onset/offset pairs")
        print(f"  - Threshold value: {detection_result.threshold_value:.4f}")
        if detection_result.baseline_value is not None:
            print(f"  - Baseline value: {detection_result.baseline_value:.4f}")

        # Visualize each detection individually
        if len(detection_result.onsets) > 0:
            fig = plot_onset_detection(
                signal=signal_,
                detection_result=detection_result,
                fs=fs,
                raw_signal=None,  # No raw signal available in this example
                channel_idx=0,  # Dimensionality already handled above
                title=f"{detector_name} Detection - Batch {batch_idx}, Channel {channel_idx + 1}",
                show_threshold=True,
                show_baseline=True,
                dark_mode=True,
            )

    # Visualize all channels together if multiple channels
    if n_channels > 1:
        fig_multi = plot_multi_channel_onsets(
            signals=processed_signals[0],  # First dimension is batch
            detection_results=detection_results,
            fs=fs,
            channel_names=[f"Channel {i + 1}" for i in range(n_channels)],
            title=f"{detector_name} Detection - Batch {batch_idx}",
            show_threshold=True,
            dark_mode=True,
        )

    return detection_results


# %%
base_dir = Path(r"E:/jpenalozaa")
data_dir = base_dir.joinpath(r"onset_detection\00_")

# Create filters with default padding (100ms)
emg_filter = create_emg_filter(
    [
        create_notch_filter(notch_freq=60),
        create_bandpass_filter(low_cutoff=20, high_cutoff=2000),
    ]
)

# Create envelope detector with default padding
envelope = create_tkeo_envelope(
    tkeo_type="tkeo2", rectification="abs", lowpass_cutoff=20
)
zscore_norm = create_zscore_normalizer()
# Combine them in a pipeline
transform = EMGTransformCompose([emg_filter, envelope, zscore_norm])

# Create data loader
dataloader, fs = create_emg_dataloader(
    data_dir=data_dir,
    window_size_sec=5.0,
    window_stride_sec=0.5,
    batch_size=1,  # Use 1 for easier processing with detection
    channels=[0, 1],
    transform=transform,
    num_workers=0,
    shuffle=False,
)

# After creating the dataloader, the filter_pipeline is now initialized with the correct fs
print(f"Dataloader created with sampling frequency: {fs} Hz")
# %%
# Start timing
start_time = time.time()

# Create an iterator from the dataloader
dataloader_iter = iter(dataloader)

# Get the first batch
batch = next(dataloader_iter)

# End timing
end_time = time.time()
elapsed = end_time - start_time

print(f"Time to load one batch: {elapsed:.4f} seconds")
print(f"Batch shape: {batch.shape}")
# %%
import matplotlib.pyplot as plt
import numpy as np

# Get a single segment from the batch (first one in the batch)
# The shape is [batch_size, channels, samples]
segment = batch[0]  # Shape should be [2, 24414]

a = plot_time_series(
    data=batch[0],  # First sample from batch
    fs=fs,
    title="EMG Recording",
    dark_mode=True,
    grid=True,
)

# %%
# Create detectors with built-in normalization
# Using the new norm parameter in the detector constructor

# 1. Fixed threshold detector with min-max normalization
fixed_detector_minmax = create_fixed_threshold_detector(
    threshold=0,  # Higher threshold since we're normalizing to [0,1]
    min_duration_ms=100.0,  # Minimum activation duration
    hysteresis=0.05,  # Hysteresis to prevent rapid on/off switching
    norm=None,  # Apply min-max normalization before detection
)
fixed_detector_minmax.initialize(fs)


# %%
# Get a single batch
batch_data = next(iter(dataloader))
batch_idx = 0

# Process the same batch with the different detectors
print(
    "\n=== Processing Single Batch with Different Detectors and Normalization Methods ==="
)

# 1. Fixed threshold with min-max normalization
print("\n--- Fixed Threshold with Min-Max Normalization ---")
minmax_results = process_and_visualize_batch(
    batch_idx, batch_data, "Fixed Threshold (MinMax Norm)", fixed_detector_minmax
)
# %%
