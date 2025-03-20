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


class EMGFilter:
    """Transform for filtering EMG data with improved numerical stability"""

    def __init__(
        self,
        bandpass: Tuple[float, float] = (20, 450),
        notch: float = 60,
        notch_quality: float = 30.0,
    ):
        """Initialize filter parameters (without computing coefficients yet)"""
        self.bandpass = bandpass
        self.notch = notch
        self.notch_quality = notch_quality  # Q factor for notch filter
        self.fs: Optional[float] = None
        self.is_initialized: bool = False
        # These will be set when initialized
        self.sos_bandpass: Optional[np.ndarray] = None
        self.sos_notch: Optional[np.ndarray] = None

    def initialize(self, fs: float) -> None:
        """Initialize filters with sampling frequency"""
        self.fs = fs
        nyquist = 0.5 * self.fs

        # Bandpass filter using SOS representation for better numerical stability
        low, high = self.bandpass[0] / nyquist, self.bandpass[1] / nyquist
        self.sos_bandpass = signal.butter(4, [low, high], btype="band", output="sos")

        # Proper IIR notch filter with Q factor
        # Q factor controls the width of the notch (higher Q = narrower notch)
        b_notch, a_notch = signal.iirnotch(
            w0=self.notch / nyquist, Q=self.notch_quality
        )
        self.sos_notch = signal.tf2sos(b_notch, a_notch)

        self.is_initialized = True

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply filtering to EMG data"""
        if not self.is_initialized:
            raise RuntimeError("EMGFilter not initialized. Call initialize(fs) first.")

        # Convert to numpy for filtering
        data_np = tensor.numpy()
        filtered_data = np.zeros_like(data_np)

        for i in range(data_np.shape[0]):
            # Apply bandpass filter using sosfiltfilt for zero-phase filtering
            bandpassed = signal.sosfiltfilt(self.sos_bandpass, data_np[i])  # type: ignore

            # Apply notch filter
            filtered = signal.sosfiltfilt(self.sos_notch, bandpassed)  # type: ignore

            filtered_data[i] = filtered

        return torch.tensor(filtered_data, dtype=tensor.dtype)


class EMGData:
    """Class for loading and accessing EMG data from parquet files"""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.parquet_path = None
        self.metadata = None
        self.data = None

        # Locate parquet file and metadata
        self._locate_files()

        # Load metadata
        self._load_metadata()

    def _locate_files(self):
        """Locate the parquet file and metadata.json"""
        # Find parquet files
        parquet_files = list(self.data_dir.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {self.data_dir}")

        # For simplicity, use the first parquet file
        self.parquet_path = parquet_files[0]

        # Check for metadata.json
        self.metadata_path = self.data_dir / "metadata.json"
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

    def _load_metadata(self):
        """Load metadata from JSON file"""
        with open(self.metadata_path, "r") as f:
            self.metadata = json.load(f)

        # Extract useful metadata
        self.fs = self.metadata["base"]["fs"]
        self.channel_names = self.metadata["base"]["channel_names"]
        self.total_samples = self.metadata["base"]["number_of_samples"]
        self.n_channels = self.metadata["base"]["channel_numbers"]

    def load_data(self, force_reload: bool = False):
        """Load data using PyArrow memory mapping"""
        if self.data is not None and not force_reload:
            return self.data

        try:
            with pa.memory_map(str(self.parquet_path), "r") as mmap:
                table = pq.read_table(mmap)
                self.data = table.to_pandas().values
        except Exception as e:
            raise RuntimeError(f"Failed to load data: {e}") from e

        return self.data


class EMGDataset(Dataset):
    """PyTorch Dataset for EMG data with windowing"""

    def __init__(
        self,
        data_dir: str,
        window_size_sec: float = 1.0,
        window_stride_sec: float = 0.5,
        channels: Optional[List[int]] = None,
        preload: bool = False,
        transform=None,
        verbose: bool = True,
    ):
        """
        Initialize the EMG dataset

        Args:
            data_dir: Directory containing parquet files and metadata.json
            window_size_sec: Window size in seconds
            window_stride_sec: Window stride in seconds
            channels: List of channel indices to load (None for all channels)
            preload: Whether to preload all data into memory
            transform: Optional transform to apply to the data
            verbose: Whether to display progress bars
        """
        self.verbose = verbose

        # Initialize EMG data loader
        self.emg_data = EMGData(Path(data_dir))

        # Initialize transform with fs if it has an initialize method
        if transform is not None and hasattr(transform, "initialize"):
            transform.initialize(self.emg_data.fs)
        self.transform = transform

        # Calculate window sizes in samples
        self.window_size = int(window_size_sec * self.emg_data.fs)
        self.window_stride = int(window_stride_sec * self.emg_data.fs)

        # Set channels to use
        if channels is None:
            self.channels = list(range(self.emg_data.n_channels))
        else:
            self.channels = channels

        # Preload data if requested
        if preload:
            if verbose:
                print("Preloading all data into memory...")
            self.data = self.emg_data.load_data()
        else:
            self.data = None

        # Calculate window indices
        self._calculate_window_indices()

    def _calculate_window_indices(self):
        """Calculate indices for all possible windows based on stride"""
        if self.verbose:
            print(
                f"Calculating window indices for {self.emg_data.total_samples} samples..."
            )

        self.window_indices = []

        # Calculate how many windows we can extract
        num_windows = max(
            0,
            (self.emg_data.total_samples - self.window_size) // self.window_stride + 1,
        )

        # Use tqdm for progress if verbose and we have many windows
        iterator = range(num_windows)
        if self.verbose and num_windows > 1000:
            iterator = tqdm(iterator, desc="Calculating windows")

        for i in iterator:
            start_idx = i * self.window_stride
            end_idx = start_idx + self.window_size
            self.window_indices.append((start_idx, end_idx))

        if self.verbose:
            print(f"Created {len(self.window_indices)} windows")

    def __len__(self) -> int:
        """Return the number of windows in the dataset"""
        return len(self.window_indices)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get a specific window of EMG data"""
        # Get window indices
        start_idx, end_idx = self.window_indices[idx]

        # Get data
        if self.data is None:
            # Load data on-the-fly
            with pa.memory_map(str(self.emg_data.parquet_path), "r") as mmap:
                table = pq.read_table(mmap)
                data = table.to_pandas().values
                window_data = data[start_idx:end_idx, self.channels]
        else:
            # Use preloaded data
            window_data = self.data[start_idx:end_idx, self.channels]

        # Convert to tensor (transpose to get [channels, samples])
        data_tensor = torch.tensor(window_data, dtype=torch.float32).T

        # Apply transform if provided
        if self.transform is not None:
            data_tensor = self.transform(data_tensor)

        return data_tensor


def create_emg_dataloader(
    data_dir: str,
    window_size_sec: float = 1.0,
    window_stride_sec: float = 0.5,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    channels: Optional[List[int]] = None,
    preload: bool = False,
    transform=None,
    verbose: bool = True,
) -> Tuple[DataLoader, float]:
    """
    Create a DataLoader for EMG data

    Args:
        data_dir: Directory containing parquet files and metadata.json
        window_size_sec: Window size in seconds
        window_stride_sec: Window stride in seconds
        batch_size: Number of samples in each batch
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for loading data
        channels: List of channel indices to load (None for all channels)
        preload: Whether to preload all data into memory
        transform: Optional transform to apply to the data
        verbose: Whether to display progress bars

    Returns:
        dataloader: PyTorch DataLoader
        fs: Sampling frequency from metadata
    """
    dataset = EMGDataset(
        data_dir=data_dir,
        window_size_sec=window_size_sec,
        window_stride_sec=window_stride_sec,
        channels=channels,
        preload=preload,
        transform=transform,
        verbose=verbose,
    )

    # Use standard DataLoader instead of custom ProgressDataLoader
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return dataloader, dataset.emg_data.fs


# %%
base_dir = Path(r"E:/jpenalozaa")
data_dir = base_dir.joinpath(r"onset_detection\00_")

transform_pipeline = EMGFilter(bandpass=(20, 2000), notch=60)


# Use the pipeline in your dataloader
dataloader, fs = create_emg_dataloader(
    data_dir=data_dir,
    window_size_sec=5.0,
    window_stride_sec=0.5,
    batch_size=32,
    channels=[0, 1],
    transform=transform_pipeline,  # Pass the composed transforms
    num_workers=0,
)
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
