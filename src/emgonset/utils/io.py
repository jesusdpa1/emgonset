"""
/src/emgonset/utils/io.py
"""

import json
from pathlib import Path
from typing import List, Optional, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .internals import public_api


class EMGTransformCompose:
    """
    Compose multiple EMG transforms together while properly handling initialization.
    Similar to torchvision.transforms.Compose but with initialize() support.
    """

    def __init__(self, transforms):
        """
        Args:
            transforms: List of transforms to apply sequentially
        """
        self.transforms = transforms

    def initialize(self, fs: float) -> None:
        """Initialize all contained transforms with sampling frequency"""
        for transform in self.transforms:
            if hasattr(transform, "initialize"):
                transform.initialize(fs)

    def __call__(self, tensor):
        """Apply all transforms sequentially"""
        for transform in self.transforms:
            tensor = transform(tensor)
        return tensor


@public_api
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


@public_api
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


@public_api
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
