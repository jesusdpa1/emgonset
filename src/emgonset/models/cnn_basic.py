"""
Author: jpenalozaa
Data: March 2025
"""

# %% Import required libraries
import logging
import os
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numba import jit
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm

# Import EMG-specific libraries
# Import EMG processing libraries
from emgonset.detection.onset import (
    OnsetOffsetResult,
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
from emgonset.processing.stft import create_stft
from emgonset.processing.transforms import EMGTransformCompose
from emgonset.utils.io import create_emg_dataloader
from emgonset.visualization.detection_plots import plot_onset_detection

from .cnn_utils import calculate_gaussian_activations

mpl.rcParams["font.family"] = "sans-serif"  # Use a common font family
mpl.rcParams["font.sans-serif"] = [
    "DejaVu Sans",
    "Arial",
    "Helvetica",
]  # Specify fallbacks


def generate_onset_labels(tsx, fs, envelope_detectors, threshold_detectors):
    """
    Generate onset/offset labels more efficiently

    Args:
        tsx: Filtered EMG tensor [batch_size, channels, time_steps]
        fs: Sampling frequency in Hz
        envelope_detectors: Pre-initialized envelope detectors
        threshold_detectors: Pre-initialized threshold detectors

    Returns:
        Tuple of (onset_labels, offset_labels, binary_activations) as tensors
    """
    batch_size, channels, time_steps = tsx.shape
    device = tsx.device

    # Create label tensors
    binary_activations = torch.zeros((batch_size, time_steps), device=device)
    onset_labels = torch.zeros((batch_size, time_steps), device=device)
    offset_labels = torch.zeros((batch_size, time_steps), device=device)

    # Process each batch sample
    for b in range(batch_size):
        # Process just first channel for simplicity and speed
        tsx_ch = tsx[b, 0].cpu()  # Use only first channel

        # Process with first envelope detector (most reliable)
        env = envelope_detectors[0]
        env_result = env(tsx_ch.unsqueeze(0))  # Add channel dimension

        # Get envelope as numpy array
        env_signal = env_result[0].cpu().numpy()

        # Apply threshold detection
        detector = threshold_detectors[0]
        detection_result = detector(env_signal)

        # Extract onset/offset indices
        onset_indices = [int(onset.sample_index) for onset in detection_result.onsets]
        offset_indices = [
            int(offset.sample_index) for offset in detection_result.offsets
        ]

        # Create NumPy arrays for Numba function
        onset_np = onset_labels[b].cpu().numpy()
        offset_np = offset_labels[b].cpu().numpy()

        # Call Numba-optimized Gaussian activations
        onset_np, offset_np = calculate_gaussian_activations(
            onset_np, offset_np, onset_indices, offset_indices, time_steps, fs
        )

        # Update tensors
        onset_labels[b] = torch.tensor(onset_np, device=device)
        offset_labels[b] = torch.tensor(offset_np, device=device)

        # Set binary activations
        for onset_idx, offset_idx in zip(onset_indices, offset_indices):
            if onset_idx < offset_idx:  # Ensure valid activation period
                onset_idx = max(0, min(onset_idx, time_steps - 1))
                offset_idx = max(0, min(offset_idx, time_steps - 1))
                binary_activations[b, onset_idx : offset_idx + 1] = 1

    return onset_labels, offset_labels, binary_activations


class STFTOnsetDetectionModel(nn.Module):
    """
    Neural network model for EMG onset detection using STFT features
    Optimized with efficient batch processing and temporal modeling
    """

    def __init__(
        self,
        input_channels=1,
        n_fft=128,
        hop_length=64,
        hidden_channels=64,
        num_layers=3,
        dropout_rate=0.3,
        stft_normalized: bool = True,
        use_temporal_layer: bool = True,
        bidirectional: bool = False,
    ):
        """
        Initialize STFT-based EMG onset detection model with optimizations

        Args:
            input_channels: Number of EMG channels (default: 1)
            n_fft: Size of FFT (default: 256)
            hop_length: Hop length for STFT (default: 64)
            hidden_channels: Size of hidden layers (default: 64)
            num_layers: Number of convolutional layers (default: 3)
            dropout_rate: Dropout probability (default: 0.3)
            stft_normalized: Whether STFT windows should be normalized (default: True)
            use_temporal_layer: Whether to use GRU for temporal modeling
            bidirectional: Whether the GRU should be bidirectional
        """
        super(STFTOnsetDetectionModel, self).__init__()

        self.input_channels = input_channels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.use_temporal_layer = use_temporal_layer
        self.bidirectional = bidirectional
        self.stft_normalized = stft_normalized

        # STFT transform (will be initialized during forward pass)
        self.stft_transform = None
        self.fs = None

        # Calculate model dimensions
        freq_bins = n_fft // 2 + 1  # For real signals

        # CNN layers for spectral feature extraction with residual connections
        self.conv_layers = nn.ModuleList()

        # First convolution takes raw spectral features
        self.conv_layers.append(
            nn.Sequential(
                nn.Conv2d(
                    input_channels, hidden_channels, kernel_size=(3, 3), padding=(1, 1)
                ),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 1)),  # Pooling in frequency dimension only
                nn.Dropout(dropout_rate),
            )
        )

        # Additional convolutional layers with residual connections
        for _ in range(num_layers - 1):
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        hidden_channels,
                        hidden_channels,
                        kernel_size=(3, 3),
                        padding=(1, 1),
                    ),
                    nn.BatchNorm2d(hidden_channels),
                    nn.ReLU(),
                    nn.MaxPool2d(
                        kernel_size=(2, 1)
                    ),  # Pooling in frequency dimension only
                    nn.Dropout(dropout_rate),
                )
            )

        # Calculate final frequency dimension after pooling
        final_freq_dim = freq_bins // (2**num_layers)
        if final_freq_dim < 1:
            final_freq_dim = 1

        # Fully connected layers for classification
        self.fc_layer = nn.Sequential(
            nn.Linear(hidden_channels * final_freq_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        # Temporal modeling layer for better time series processing
        if use_temporal_layer:
            gru_output_size = hidden_channels * (2 if bidirectional else 1)
            self.temporal_layer = nn.GRU(
                input_size=hidden_channels,
                hidden_size=hidden_channels,
                batch_first=True,
                bidirectional=bidirectional,
            )
            self.temporal_projection = nn.Linear(gru_output_size, hidden_channels)

        # Output layers for onset and offset detection
        self.onset_layer = nn.Linear(hidden_channels, 1)
        self.offset_layer = nn.Linear(hidden_channels, 1)

        # Initialize weights for better convergence
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)

    def initialize_stft(self, fs):
        """
        Initialize STFT transform with sampling frequency using torchaudio directly

        Args:
            fs: Sampling frequency in Hz
        """
        if self.stft_transform is None or self.fs != fs:
            self.fs = fs

            # Get the model's device
            device = next(self.parameters()).device

            # Use torchaudio's Spectrogram transform directly
            import torchaudio.transforms as T

            self.stft_transform = T.Spectrogram(
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.n_fft,
                power=2.0,  # Power spectrogram
                normalized=self.stft_normalized,
                center=True,
                pad_mode="reflect",
                window_fn=torch.hann_window,
            ).to(device)

            # Log initialization
            print(
                f"STFT initialized using torchaudio on device: {device} with fs={fs} Hz"
            )

    def compute_stft(self, tsx):
        """
        Compute STFT features with optimized batch processing using torchaudio

        Args:
            tsx: Input tensor [batch_size, channels, time_steps]

        Returns:
            STFT features [batch_size, channels, freq_bins, time_frames]
        """
        batch_size, channels, time_steps = tsx.shape
        device = tsx.device
        # Process efficiently with batched operations
        stft_features = []
        self.stft_transform = self.stft_transform.to(device)
        for b in range(batch_size):
            # Process all channels for this batch
            ch_features = []

            for c in range(channels):
                # Get the time series for this channel
                tsx_bc = tsx[b, c]

                # Apply STFT directly using torchaudio
                # Send both transform and data to the same device
                stft = self.stft_transform(tsx_bc)
                ch_features.append(stft)

            # Stack channels
            stft_features.append(torch.stack(ch_features))

        # Stack batches
        return torch.stack(stft_features)

    def forward(self, tsx, fs=None):
        """
        Forward pass of the model with optimized processing

        Args:
            tsx: Input tensor [batch_size, channels, time_steps]
            fs: Sampling frequency in Hz (needed for STFT)

        Returns:
            Activity prediction probabilities
        """
        batch_size, channels, time_steps = tsx.shape

        # Initialize STFT transform if needed
        if fs is not None:
            self.initialize_stft(fs)

        if self.stft_transform is None:
            raise ValueError(
                "STFT transform not initialized. Please provide sampling frequency."
            )

        # Compute STFT features - batched operation
        # Shape: [batch_size, channels, freq_bins, time_frames]
        stft_features = self.compute_stft(tsx)

        # Apply convolutional layers with residual connections
        features = stft_features
        residual = None

        for i, conv_layer in enumerate(self.conv_layers):
            if i > 0 and residual is not None:
                # Apply residual connection if shapes match
                if features.shape[2:] == residual.shape[2:]:
                    features = features + residual

            # Store for next residual connection
            residual = features.clone()
            # Apply convolution
            features = conv_layer(features)

        # Reshape for fully connected layers and temporal processing
        # [batch_size, channels, freq_bins, time_frames] -> [batch_size, time_frames, channels*freq_bins]
        batch_size, channels, freq_bins, time_frames = features.shape
        features = features.permute(0, 3, 1, 2).contiguous()
        features = features.view(batch_size, time_frames, channels * freq_bins)

        # Apply fully connected layer
        features = self.fc_layer(features)

        # Apply temporal modeling if enabled
        if self.use_temporal_layer:
            # GRU for temporal dependency modeling
            features, _ = self.temporal_layer(features)

            # If bidirectional, project back to hidden_channels size
            if self.bidirectional:
                features = self.temporal_projection(features)

        # Generate onset and offset predictions
        onset_logits = self.onset_layer(features).squeeze(-1)
        offset_logits = self.offset_layer(features).squeeze(-1)

        # Apply sigmoid to get probabilities
        onset_probs = torch.sigmoid(onset_logits)
        offset_probs = torch.sigmoid(offset_logits)

        # Average the probabilities to get activity prediction
        activity_probs = (onset_probs + offset_probs) / 2.0

        # Interpolate back to original time resolution if needed
        if time_frames != time_steps:
            # Upsample to match original tsx length
            activity_probs = F.interpolate(
                activity_probs.unsqueeze(1),
                size=time_steps,
                mode="linear",
                align_corners=False,
            ).squeeze(1)

        # Return activity prediction
        return activity_probs


def activity_detection_loss(activity_pred, binary_activations, pos_weight=5.0):
    """
    Weighted binary cross-entropy loss for direct activity detection

    Args:
        activity_pred: Predicted activity probabilities
        binary_activations: Ground truth binary activations (active/inactive regions)
        pos_weight: Weight for positive examples (active regions)

    Returns:
        Loss value
    """
    # Prevent numerical instability
    eps = 1e-7
    activity_pred = torch.clamp(activity_pred, eps, 1.0 - eps)

    # Weighted BCE loss
    pos_term = pos_weight * binary_activations * torch.log(activity_pred)
    neg_term = (1 - binary_activations) * torch.log(1 - activity_pred)

    return -torch.mean(pos_term + neg_term)


# %%
checkpoint_dir: str = "checkpoints"
debug_mode: bool = True  # Set to True for detailed logging

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

log_file: str = (
    f"{checkpoint_dir}/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)

logging.basicConfig(
    level=logging.DEBUG if debug_mode else logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# %% Data loading and preprocessing
# Data directory
base_dir = Path(r"E:/jpenalozaa")
data_dir = base_dir.joinpath(r"onset_detection\00_")

# Create preprocessing pipeline - ONLY FILTERING
emg_filter = create_emg_filter(
    [
        create_notch_filter(notch_freq=60),  # Remove power line noise
        create_bandpass_filter(
            low_cutoff=20, high_cutoff=2000
        ),  # Keep relevant EMG frequencies
    ]
)

# Apply filtering in the dataloader
transform = EMGTransformCompose([emg_filter])

# Create dataloader with single channel
# No shuffling for time series data to preserve temporal dependencies
channel_idx: int = 0  # Use first channel

logger.info("Creating EMG dataloader...")
dataloader, fs = create_emg_dataloader(
    data_dir=str(data_dir),
    window_size_sec=5.0,
    window_stride_sec=1.0,
    batch_size=16,
    channels=[channel_idx],  # Single channel
    transform=transform,
    num_workers=0,  # No multiprocessing - could be a bottleneck
    shuffle=False,  # No shuffling for time series data
)

logger.info(f"Data loaded with sampling frequency: {fs} Hz")

# %% Sample batch check and data splitting
# Get sample batch to check dimensions
sample_batch = next(iter(dataloader))
batch_size, channels, time_steps = sample_batch.shape
logger.info(f"Sample batch shape: {sample_batch.shape}")

# Split data chronologically for training and validation
# For time series, we DON'T use random splitting
data_size: int = len(dataloader.dataset)
train_size: int = int(0.8 * data_size)  # Use 80% for training
val_size: int = data_size - train_size

# Create chronological train/val split (NOT random)
train_dataset = Subset(cast(Dataset, dataloader.dataset), range(0, train_size))
val_dataset = Subset(cast(Dataset, dataloader.dataset), range(train_size, data_size))

# Create new dataloaders with the splits
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=False,  # No shuffling for time series
    num_workers=0,  # Could increase this to speed up data loading
    pin_memory=True,  # Speed up CPU->GPU transfers
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,  # No shuffling for time series
    num_workers=0,
    pin_memory=True,  # Speed up CPU->GPU transfers
)

logger.info(f"Training on {train_size} samples, validating on {val_size} samples")

# %% Model initialization
# Create model with temporal layers for better time series handling
model = STFTOnsetDetectionModel(
    input_channels=channels,
    n_fft=128,
    hop_length=64,
    hidden_channels=64,
    num_layers=3,
    use_temporal_layer=True,  # Add temporal modeling (GRU)
    bidirectional=True,  # Use bidirectional GRU for better temporal context
)

# %% Configure device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Log device information
if device.type == "cuda":
    logger.info(f"Training on GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"CUDA Version: {torch.version.cuda}")
    logger.info(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    logger.info(f"Memory cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
else:
    logger.info("Training on CPU")

logger.info(f"Model architecture:\n{model}")

# Move model to device
model.to(device)

# %% Set training parameters and optimizer
num_epochs: int = 30
learning_rate: float = 1e-3
weight_decay: float = 1e-5
checkpoint_frequency: int = 5
early_stopping_patience: int = 10

# Initialize optimizer and scheduler
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=weight_decay
)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=5, verbose=True
)

# Save initial model state
torch.save(
    {
        "epoch": 0,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    },
    f"{checkpoint_dir}/initial_model.pth",
)


# %% Add profiling to track performance
class Profiler:
    def __init__(self) -> None:
        self.steps: Dict[str, List[float]] = {}

    def record(self, step_name: str, duration: float) -> None:
        if step_name not in self.steps:
            self.steps[step_name] = []
        self.steps[step_name].append(duration)

    def get_avg(self, step_name: str) -> float:
        if step_name not in self.steps:
            return 0.0
        return sum(self.steps[step_name]) / max(1, len(self.steps[step_name]))

    def report(self) -> None:
        print("\n--- Profiler Report ---")
        for step_name, times in self.steps.items():
            print(
                f"{step_name}: {sum(times) / max(1, len(times)):.4f}s avg, {sum(times):.2f}s total"
            )


profiler = Profiler()

# Initialize tracking variables
train_losses: List[float] = []
val_losses: List[float] = []
best_val_loss: float = float("inf")
epochs_no_improve: int = 0


# %% Define a data prefetcher to speed up data loading
class DataPrefetcher:
    def __init__(self, loader: DataLoader, device: torch.device) -> None:
        self.loader = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream() if device.type == "cuda" else None
        self.next_tsx: Optional[torch.Tensor] = None
        self.preload()

    def preload(self) -> None:
        try:
            self.next_tsx = next(self.loader)
        except StopIteration:
            self.next_tsx = None
            return

        if self.stream is not None:
            with torch.cuda.stream(self.stream):
                self.next_tsx = self.next_tsx.to(self.device, non_blocking=True)

    def next(self) -> Optional[torch.Tensor]:
        if self.stream is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
        tsx = self.next_tsx
        self.preload()
        return tsx


# %%
## initialize for onset_detection

# Initialize these once at the start of training, not inside the function
envelope_detectors = [
    create_tkeo_envelope(
        tkeo_type="tkeo2", rectification="abs", lowpass_cutoff=40.0, normalize=True
    ),
    create_tkeo_envelope(
        tkeo_type="mtkeo",
        rectification="abs",
        lowpass_cutoff=40.0,
        normalize=True,
        tkeo_params={"k1": 1.0, "k2": 0.5, "k3": 0.5},
    ),
]
threshold_detectors = [
    create_fixed_threshold_detector(
        threshold=0.05, min_duration_ms=50.0, hysteresis=0.01, norm="minmax"
    )
]

# Initialize them once
for envelope in envelope_detectors:
    envelope.initialize(fs)
for detector in threshold_detectors:
    detector.initialize(fs)

# Now update the training loop
# %% Run the first epoch to identify bottlenecks
epoch = 0
epoch_start = time.time()

# --- Training phase ---
model.train()
running_loss = 0.0
batch_count = 0

# Progress bar for batches
train_pbar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}", position=0)

# Use prefetcher if on CUDA
use_prefetcher = device.type == "cuda"
if use_prefetcher:
    prefetcher = DataPrefetcher(train_loader, device)
    batch_data = prefetcher.next()
    batch_idx = 0
else:
    # Regular iteration if not using prefetcher
    train_iter = iter(train_pbar)

# Record batch processing times
batch_times: List[float] = []

while True:
    batch_start = time.time()

    if use_prefetcher:
        if batch_data is None:
            break
    else:
        try:
            batch_data = next(train_iter)
            batch_idx = len(batch_times)
        except StopIteration:
            break

    # Test first batch (good for debugging)
    if batch_idx == 0:
        logger.debug(f"First batch shape: {batch_data.shape}")
        logger.debug(f"Batch data type: {batch_data.dtype}")
        logger.debug(
            f"Batch range: min={batch_data.min().item():.4f}, max={batch_data.max().item():.4f}"
        )

        # Test model forward pass
        try:
            logger.debug("Testing forward pass on first batch...")
            with torch.no_grad():
                # Updated to handle single activity output
                test_activity = model(batch_data.to(device), fs=fs)
            logger.debug(
                f"Forward pass successful: activity_pred shape={test_activity.shape}"
            )
        except Exception as e:
            logger.error(f"Forward pass test failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    try:
        # Move data to device if not using prefetcher
        t0 = time.time()
        if not use_prefetcher:
            batch_data = batch_data.to(device)
        profiler.record("data_to_device", time.time() - t0)

        # Memory logging (every 10 batches)
        if debug_mode and batch_idx % 10 == 0 and device.type == "cuda":
            logger.debug(
                f"CUDA memory: allocated {torch.cuda.memory_allocated(0) / 1e9:.2f} GB"
            )

        # Zero gradients
        t0 = time.time()
        optimizer.zero_grad()
        profiler.record("zero_grad", time.time() - t0)

        # Generate labels - THIS MIGHT BE A BOTTLENECK
        t0 = time.time()
        # Keep the original label generation but we'll only use binary_activations
        _, _, binary_activations = generate_onset_labels(
            batch_data,
            fs,
            envelope_detectors=envelope_detectors,
            threshold_detectors=threshold_detectors,
        )
        label_time = time.time() - t0
        profiler.record("label_generation", label_time)

        if label_time > 0.5:  # If label generation is slow
            logger.warning(
                f"Label generation taking {label_time:.2f}s - potential bottleneck"
            )

        # Forward pass
        t0 = time.time()
        # Get single activity prediction from model
        activity_pred = model(batch_data, fs=fs)
        forward_time = time.time() - t0
        profiler.record("forward", forward_time)

        if forward_time > 0.5:  # If forward pass is slow
            logger.warning(
                f"Forward pass taking {forward_time:.2f}s - potential bottleneck"
            )

        # Calculate loss - using the new loss function
        t0 = time.time()
        loss = activity_detection_loss(
            activity_pred, binary_activations, pos_weight=5.0
        )
        profiler.record("loss_calculation", time.time() - t0)

        # Catch NaN losses
        if torch.isnan(loss):
            logger.error(f"NaN loss detected in batch {batch_idx}. Skipping batch.")
            if use_prefetcher:
                batch_data = prefetcher.next()
            continue

        # Backward pass
        t0 = time.time()
        loss.backward()
        backward_time = time.time() - t0
        profiler.record("backward", backward_time)

        if backward_time > 0.5:  # If backward pass is slow
            logger.warning(
                f"Backward pass taking {backward_time:.2f}s - potential bottleneck"
            )

        # Check for exploding gradients
        t0 = time.time()
        grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2).item()
                grad_norm += param_norm**2
        grad_norm = grad_norm**0.5

        if grad_norm > 10.0:  # Threshold for exploding gradients
            logger.warning(f"Exploding gradient: {grad_norm:.2f}. Clipping gradients.")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        profiler.record("gradient_check", time.time() - t0)

        # Optimizer step
        t0 = time.time()
        optimizer.step()
        profiler.record("optimizer_step", time.time() - t0)

        # Track loss
        running_loss += loss.item()
        batch_count += 1

        # Update progress bar
        train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        train_pbar.update(1)

        # Track total batch time
        batch_time = time.time() - batch_start
        batch_times.append(batch_time)
        profiler.record("full_batch", batch_time)

        # Get next batch if using prefetcher
        if use_prefetcher:
            batch_data = prefetcher.next()
            batch_idx += 1

    except Exception as e:
        logger.error(f"Error in training batch {batch_idx}: {str(e)}")
        logger.error(traceback.format_exc())
        # Debug information for the failed batch
        try:
            if "batch_data" in locals():
                logger.error(f"Failed batch shape: {batch_data.shape}")
                logger.error(
                    f"Batch min: {batch_data.min().item():.4f}, max: {batch_data.max().item():.4f}"
                )
        except:
            logger.error("Could not log batch details")

        # Get next batch if using prefetcher
        if use_prefetcher:
            batch_data = prefetcher.next()
            batch_idx += 1
        continue

# Calculate epoch average loss
epoch_loss = running_loss / max(1, batch_count)
train_losses.append(epoch_loss)

# Print profiler report
logger.info("First epoch completed - Profiler Report:")
profiler.report()

# Print detailed bottleneck analysis
print("\n=== BOTTLENECK ANALYSIS ===")
step_times = [(step, profiler.get_avg(step)) for step in profiler.steps]
step_times.sort(key=lambda x: x[1], reverse=True)

print("Top 3 bottlenecks:")
for i, (step, avg_time) in enumerate(step_times[:3]):
    print(
        f"{i + 1}. {step}: {avg_time:.4f}s average - {avg_time / sum(batch_times) * 100:.1f}% of batch time"
    )

# %%
# Load a sample from the validation set
sample_batch = next(iter(val_loader))
sample_tsx = sample_batch.to(device)

# Get ground truth labels for comparison
_, _, binary_gt = generate_onset_labels(
    sample_tsx,
    fs,
    envelope_detectors=envelope_detectors,
    threshold_detectors=threshold_detectors,
)

# Set model to evaluation mode
model.eval()

# Forward pass
with torch.no_grad():
    activity_pred = model(sample_tsx, fs=fs)

# Select the first sample for visualization
idx = 0  # Choose first sample
emg_signal = sample_tsx[idx, 0].cpu().numpy()
pred = activity_pred[idx].cpu().numpy()
gt = binary_gt[idx].cpu().numpy()

# Get some statistics on the predictions
threshold = 0.5
binary_pred = (activity_pred > threshold).float()

# Calculate simple metrics
correct = (binary_pred == binary_gt).float().mean().item()
true_positives = ((binary_pred == 1) & (binary_gt == 1)).float().sum().item()
false_positives = ((binary_pred == 1) & (binary_gt == 0)).float().sum().item()
false_negatives = ((binary_pred == 0) & (binary_gt == 1)).float().sum().item()
true_negatives = ((binary_pred == 0) & (binary_gt == 0)).float().sum().item()

# Calculate accuracy, precision, recall, F1
accuracy = correct
precision = true_positives / (true_positives + false_positives + 1e-7)
recall = true_positives / (true_positives + false_negatives + 1e-7)
f1 = 2 * precision * recall / (precision + recall + 1e-7)

# Class balance
active_ratio = binary_gt.mean().item() * 100

# Print results
print(f"Overall accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 score: {f1:.4f}")
print(f"Active time percentage: {active_ratio:.2f}%")

# Print prediction analysis
print("\nSample prediction analysis:")
print(f"Sample shape: {sample_tsx.shape}")
for i in range(min(3, sample_tsx.shape[0])):  # First 3 samples
    sample_pred = activity_pred[i].cpu().numpy()
    sample_gt = binary_gt[i].cpu().numpy()

    # Get continuous activation regions
    pred_activations = []
    gt_activations = []

    # Find predicted activation regions
    in_activation = False
    start_idx = 0
    for j in range(len(sample_pred)):
        if sample_pred[j] > threshold and not in_activation:
            in_activation = True
            start_idx = j
        elif sample_pred[j] <= threshold and in_activation:
            in_activation = False
            pred_activations.append((start_idx, j))

    # Handle case where activation continues to the end
    if in_activation:
        pred_activations.append((start_idx, len(sample_pred) - 1))

    # Find ground truth activation regions
    in_activation = False
    start_idx = 0
    for j in range(len(sample_gt)):
        if sample_gt[j] > 0.5 and not in_activation:
            in_activation = True
            start_idx = j
        elif sample_gt[j] <= 0.5 and in_activation:
            in_activation = False
            gt_activations.append((start_idx, j))

    # Handle case where activation continues to the end
    if in_activation:
        gt_activations.append((start_idx, len(sample_gt) - 1))

    print(f"\nSample {i + 1}:")
    print(f"  Ground truth activations: {len(gt_activations)}")
    print(f"  Predicted activations: {len(pred_activations)}")

    # Print first few activations
    if gt_activations:
        print(
            f"  First GT activation: {gt_activations[0][0] / fs:.2f}s - {gt_activations[0][1] / fs:.2f}s (duration: {(gt_activations[0][1] - gt_activations[0][0]) / fs:.2f}s)"
        )
    if pred_activations:
        print(
            f"  First predicted activation: {pred_activations[0][0] / fs:.2f}s - {pred_activations[0][1] / fs:.2f}s (duration: {(pred_activations[0][1] - pred_activations[0][0]) / fs:.2f}s)"
        )

# Create time vector
time = np.arange(len(emg_signal)) / fs

# Create the plot
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

# Plot raw EMG signal
axes[0].plot(time, emg_signal, "b-")
axes[0].set_title("Raw EMG Signal")
axes[0].set_ylabel("Amplitude")
axes[0].grid(True)

# Plot ground truth activity
axes[1].plot(time, gt, "g-")
axes[1].set_title("Ground Truth Activity")
axes[1].set_ylabel("Activity")
axes[1].grid(True)

# Plot predicted activity
axes[2].plot(time, pred, "r-")
axes[2].axhline(y=0.5, color="k", linestyle="--", alpha=0.5, label="Threshold")
axes[2].set_title("Predicted Activity")
axes[2].set_xlabel("Time (s)")
axes[2].set_ylabel("Probability")
axes[2].grid(True)
axes[2].legend()

plt.tight_layout()
plt.show()

# %%
