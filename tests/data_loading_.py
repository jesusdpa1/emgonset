# %%
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from scipy import signal
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from emgonset.processing.filters import (
    create_bandpass_filter,
    create_emg_filter,
    create_notch_filter,
)
from emgonset.processing.rectifiers import create_abs_rectifier
from emgonset.processing.tkeo import create_tkeo2
from emgonset.processing.transforms import EMGTransformCompose
from emgonset.utils.io import create_emg_dataloader  # Corrected import

# %%
base_dir = Path(r"E:/jpenalozaa")
data_dir = base_dir.joinpath(r"onset_detection\00_")

# Create the filter pipeline without initializing yet
# These filter functions will be initialized when the dataset is created
emg_filter = create_emg_filter(
    [
        create_notch_filter(notch_freq=60),
        create_bandpass_filter(low_cutoff=20, high_cutoff=2000),
    ]
)

# Create a rectifier
rectifier = create_abs_rectifier()

# TKEO2 (4-sample)
transform = EMGTransformCompose(
    [
        emg_filter,
        create_abs_rectifier(),
        create_tkeo2(),  # 4-sample TKEO
    ]
)


# The dataset will initialize all filters with the correct fs
dataloader, fs = create_emg_dataloader(
    data_dir=data_dir,
    window_size_sec=5.0,
    window_stride_sec=0.5,
    batch_size=32,
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

# Create time vector in seconds
time_ = np.linspace(0, segment.shape[1] / fs, segment.shape[1])

# Create the plot
plt.figure(figsize=(12, 6))

# Plot both channels
for i in range(segment.shape[0]):
    plt.subplot(2, 1, i + 1)
    plt.plot(time_, segment[i].numpy())
    plt.title(f"EMG Channel {i}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)

plt.tight_layout()
plt.show()

# If you want to zoom in on a specific region (e.g., first 0.2 seconds)
plt.figure(figsize=(12, 6))

# %%
