"""
Author: jpenalozaa
Data: March 2025
"""

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numba import jit
from torch.utils.data import DataLoader, Dataset

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
from emgonset.processing.normalization import (
    BaseNormalizer,
    create_max_amplitude_normalizer,
    create_minmax_normalizer,
    create_robust_normalizer,
    create_zscore_normalizer,
)
from emgonset.processing.stft import create_stft
from emgonset.processing.transforms import EMGTransformCompose
from emgonset.utils.io import create_emg_dataloader
from emgonset.visualization.detection_plots import plot_onset_detection


@jit(nopython=True)
def calculate_gaussian_activations(
    onset_labels, offset_labels, detection_onsets, detection_offsets, time_steps, fs
):
    """
    Numba-optimized function to calculate Gaussian activations around onset/offset points

    Args:
        onset_labels: NumPy array to store onset activations
        offset_labels: NumPy array to store offset activations
        detection_onsets: List of onset indices
        detection_offsets: List of offset indices
        time_steps: Total number of time steps
        fs: Sampling frequency
    """
    # Create binary activations and Gaussian smoothed labels
    for i in range(len(detection_onsets)):
        onset_idx = min(max(detection_onsets[i], 0), time_steps - 1)
        offset_idx = min(max(detection_offsets[i], 0), time_steps - 1)

        # Create Gaussian activations around onset/offset points
        sigma = int(0.01 * fs)  # 10ms spread
        window_size = 5 * sigma  # Only calculate within 5 standard deviations

        # Onset
        start_t = max(0, onset_idx - window_size)
        end_t = min(time_steps, onset_idx + window_size + 1)
        for t in range(start_t, end_t):
            onset_val = np.exp(-0.5 * ((t - onset_idx) / sigma) ** 2)
            if onset_val > onset_labels[t]:
                onset_labels[t] = onset_val

        # Offset
        start_t = max(0, offset_idx - window_size)
        end_t = min(time_steps, offset_idx + window_size + 1)
        for t in range(start_t, end_t):
            offset_val = np.exp(-0.5 * ((t - offset_idx) / sigma) ** 2)
            if offset_val > offset_labels[t]:
                offset_labels[t] = offset_val

    return onset_labels, offset_labels


def generate_onset_labels(signal, fs):
    """
    Generate onset/offset labels using traditional detection methods
    with smooth Gaussian activations - Numba-optimized version

    Args:
        signal: Filtered EMG tensor [batch_size, channels, time_steps]
        fs: Sampling frequency in Hz

    Returns:
        Tuple of (onset_labels, offset_labels, binary_activations) as tensors
    """
    batch_size, channels, time_steps = signal.shape
    device = signal.device  # Get the original device

    # Create label tensors
    binary_activations = torch.zeros((batch_size, time_steps), device=device)
    onset_labels = torch.zeros((batch_size, time_steps), device=device)
    offset_labels = torch.zeros((batch_size, time_steps), device=device)

    # Create envelope detectors WITH normalization (simplified approach)
    envelope_detectors = [
        # TKEO2 with normalization
        create_tkeo_envelope(
            tkeo_type="tkeo2",
            rectification="abs",
            lowpass_cutoff=40.0,
            normalize=True,
        ),
        # Modified TKEO with normalization
        create_tkeo_envelope(
            tkeo_type="mtkeo",
            rectification="abs",
            lowpass_cutoff=40.0,
            normalize=True,
            tkeo_params={"k1": 1.0, "k2": 0.5, "k3": 0.5},
        ),
    ]

    # Create only fixed threshold detector for consistency
    threshold_detectors = [
        # Fixed threshold
        create_fixed_threshold_detector(
            threshold=0.05,
            min_duration_ms=50.0,
            hysteresis=0.01,
            norm="minmax",  # Using internal normalization
        ),
    ]

    # Initialize all processors
    for envelope in envelope_detectors:
        envelope.initialize(fs)

    # Initialize threshold detectors
    for detector in threshold_detectors:
        detector.initialize(fs)

    # Process each batch sample
    for b in range(batch_size):
        for ch in range(channels):
            # Get filtered signal - ensure on CPU for envelope processing
            filtered_signal = signal[b, ch].detach().clone().cpu()

            # Process each envelope method
            for envelope in envelope_detectors:
                # Prepare tensor for envelope processing
                signal_input = filtered_signal.unsqueeze(0)  # Add channel dim

                try:
                    # Apply envelope detection (with built-in normalization)
                    env_result = envelope(signal_input)

                    # Extract the envelope signal
                    if env_result.ndim > 2:
                        # Handle multi-channel output if needed
                        env_signal = env_result[0].cpu().numpy()  # Take first channel
                    else:
                        # Extract single channel
                        env_signal = env_result.squeeze(0).cpu().numpy()

                    # Ensure correct dimensions
                    if env_signal.ndim > 1 and env_signal.shape[0] != time_steps:
                        # Try to transpose if dimensions don't match
                        if env_signal.shape[1] == time_steps:
                            env_signal = env_signal.T
                        else:
                            # If still not matching, find a dimension that matches
                            for i in range(env_signal.shape[0]):
                                if env_signal[i].shape[0] == time_steps:
                                    env_signal = env_signal[i]
                                    break

                    # Apply threshold detection to each envelope
                    for detector in threshold_detectors:
                        detection_result = detector(env_signal)

                        # Prepare data for Numba optimization
                        onset_indices = [
                            int(onset.sample_index) for onset in detection_result.onsets
                        ]
                        offset_indices = [
                            int(offset.sample_index)
                            for offset in detection_result.offsets
                        ]

                        # Create NumPy arrays for Numba function
                        onset_np = onset_labels[b].cpu().numpy()
                        offset_np = offset_labels[b].cpu().numpy()

                        # Call Numba-optimized function for Gaussian activations
                        onset_np, offset_np = calculate_gaussian_activations(
                            onset_np,
                            offset_np,
                            onset_indices,
                            offset_indices,
                            time_steps,
                            fs,
                        )

                        # Update PyTorch tensors with results
                        onset_labels[b] = torch.tensor(onset_np, device=device)
                        offset_labels[b] = torch.tensor(offset_np, device=device)

                        # Set binary activation periods
                        for onset_idx, offset_idx in zip(onset_indices, offset_indices):
                            if onset_idx < offset_idx:  # Ensure valid activation period
                                onset_idx = max(0, min(onset_idx, time_steps - 1))
                                offset_idx = max(0, min(offset_idx, time_steps - 1))
                                binary_activations[b, onset_idx : offset_idx + 1] = 1
                except Exception as e:
                    print(f"Error processing envelope: {e}")
                    continue

    return onset_labels, offset_labels, binary_activations


class STFTOnsetDetectionModel(nn.Module):
    """
    Neural network model for EMG onset detection using STFT features

    Note: This model is designed for time series data and preserves
    temporal dependencies through its architecture.
    """

    def __init__(
        self,
        input_channels=1,
        n_fft=256,
        hop_length=64,
        hidden_channels=64,
        num_layers=3,
        dropout_rate=0.3,
        normalization: str = None,
        norm_params: dict = None,
    ):
        """
        Initialize STFT-based EMG onset detection model

        Args:
            input_channels: Number of EMG channels (default: 1)
            n_fft: Size of FFT (default: 256)
            hop_length: Hop length for STFT (default: 64)
            hidden_channels: Size of hidden layers (default: 64)
            num_layers: Number of convolutional layers (default: 3)
            dropout_rate: Dropout probability (default: 0.3)
            normalization: Type of normalization to apply
                           (None, 'minmax', 'zscore', 'robust', 'max_amplitude')
            norm_params: Additional parameters for normalization
        """
        super(STFTOnsetDetectionModel, self).__init__()

        self.input_channels = input_channels
        self.n_fft = n_fft
        self.hop_length = hop_length

        # Normalization setup
        self.normalization = None
        if normalization is not None:
            norm_params = norm_params or {}

            # Create normalizer based on the specified type
            if normalization == "minmax":
                self.normalization = create_minmax_normalizer(**norm_params)
            elif normalization == "zscore":
                self.normalization = create_zscore_normalizer(**norm_params)
            elif normalization == "robust":
                self.normalization = create_robust_normalizer(**norm_params)
            elif normalization == "max_amplitude":
                self.normalization = create_max_amplitude_normalizer(**norm_params)
            else:
                raise ValueError(f"Unsupported normalization type: {normalization}")

        # STFT transform (will be initialized during forward pass)
        self.stft_transform = None
        self.fs = None

        # Calculate model dimensions
        freq_bins = n_fft // 2 + 1  # For real signals

        # CNN layers for spectral feature extraction
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

        # Additional convolutional layers
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

        # Output layers for onset and offset detection
        self.onset_layer = nn.Linear(hidden_channels, 1)
        self.offset_layer = nn.Linear(hidden_channels, 1)

    def initialize_stft(self, fs, normalized: bool = False):
        """
        Initialize STFT transform with sampling frequency

        Args:
            fs: Sampling frequency in Hz
            normalized: Whether to normalize the window function to preserve signal energy
        """
        if self.stft_transform is None or self.fs != fs:
            self.fs = fs
            device = next(self.parameters()).device  # Get the device of the model
            self.stft_transform = create_stft(
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window_fn=lambda size: torch.hann_window(size).to(
                    device
                ),  # Move window to the same device
                power=2.0,  # Use power spectrogram
                normalized=normalized,
            )
            self.stft_transform.initialize(fs)

    def compute_stft(self, x):
        """Compute STFT features for the input signal"""
        batch_size, channels, time_steps = x.shape

        # Optional pre-processing normalization
        if self.normalization is not None:
            if isinstance(self.normalization, BaseNormalizer):
                x = self.normalization(x)

        # Process each batch and channel
        stft_features = []

        for b in range(batch_size):
            channel_features = []

            for ch in range(channels):
                # Get signal for this channel
                signal = x[b, ch].detach().clone()

                # Ensure signal is a tensor
                if not isinstance(signal, torch.Tensor):
                    signal = torch.tensor(signal, dtype=torch.float32)

                # Apply STFT
                stft = self.stft_transform(signal.unsqueeze(0))
                channel_features.append(stft.squeeze(0))

            # Stack channels
            stft_features.append(torch.stack(channel_features))

        # Stack batches
        return torch.stack(stft_features)

    def forward(self, x, fs=None, stft_normalized: bool = False):
        """
        Forward pass of the model

        Args:
            x: Input tensor [batch_size, channels, time_steps]
            fs: Sampling frequency in Hz (needed for STFT)
            stft_normalized: Whether to normalize STFT window function

        Returns:
            Tuple of (onset_probabilities, offset_probabilities)
        """
        batch_size, channels, time_steps = x.shape

        # Initialize STFT transform if needed
        if fs is not None:
            self.initialize_stft(fs, normalized=stft_normalized)

        if self.stft_transform is None:
            raise ValueError(
                "STFT transform not initialized. Please provide sampling frequency."
            )

        # Compute STFT features
        # Shape: [batch_size, channels, freq_bins, time_frames]
        stft_features = self.compute_stft(x)

        # Apply convolutional layers
        features = stft_features
        for conv_layer in self.conv_layers:
            features = conv_layer(features)

        # Reshape for fully connected layers
        # [batch_size, channels, freq_bins, time_frames] -> [batch_size, time_frames, channels*freq_bins]
        batch_size, channels, freq_bins, time_frames = features.shape
        features = features.permute(0, 3, 1, 2).contiguous()
        features = features.view(batch_size, time_frames, channels * freq_bins)

        # Apply fully connected layer
        features = self.fc_layer(features)

        # Generate onset and offset predictions
        onset_logits = self.onset_layer(features).squeeze(-1)
        offset_logits = self.offset_layer(features).squeeze(-1)

        # Apply sigmoid to get probabilities
        onset_probs = torch.sigmoid(onset_logits)
        offset_probs = torch.sigmoid(offset_logits)

        # Interpolate back to original time resolution if needed
        if time_frames != time_steps:
            # Upsample to match original signal length
            onset_probs = F.interpolate(
                onset_probs.unsqueeze(1),
                size=time_steps,
                mode="linear",
                align_corners=False,
            ).squeeze(1)

            offset_probs = F.interpolate(
                offset_probs.unsqueeze(1),
                size=time_steps,
                mode="linear",
                align_corners=False,
            ).squeeze(1)

        return onset_probs, offset_probs


# ---------------------------------------------------------------
# Loss Function
# ---------------------------------------------------------------


def weighted_bce_loss(pred, target, pos_weight=2.0):
    """
    Weighted binary cross-entropy loss

    Args:
        pred: Predicted probabilities
        target: Target labels
        pos_weight: Weight for positive examples

    Returns:
        Loss value
    """
    # Prevent numerical instability
    eps = 1e-7
    pred = torch.clamp(pred, eps, 1.0 - eps)

    # Weighted BCE loss
    pos_term = pos_weight * target * torch.log(pred)
    neg_term = (1 - target) * torch.log(1 - pred)

    return -torch.mean(pos_term + neg_term)


def onset_detection_loss(
    onset_pred, offset_pred, onset_gt, offset_gt, onset_weight=2.0, offset_weight=2.0
):
    """
    Combined loss function for onset/offset detection

    Args:
        onset_pred: Predicted onset probabilities
        offset_pred: Predicted offset probabilities
        onset_gt: Ground truth onset labels
        offset_gt: Ground truth offset labels
        onset_weight: Weight for positive onset examples
        offset_weight: Weight for positive offset examples

    Returns:
        Combined loss value
    """
    # Calculate weighted BCE for onsets and offsets
    onset_loss = weighted_bce_loss(onset_pred, onset_gt, pos_weight=onset_weight)
    offset_loss = weighted_bce_loss(offset_pred, offset_gt, pos_weight=offset_weight)

    # Combine losses (equal weighting)
    return onset_loss + offset_loss


# ---------------------------------------------------------------
# Training Functions
# ---------------------------------------------------------------
def train_model(
    model,
    dataloader,
    fs,
    num_epochs=30,
    learning_rate=1e-3,
    weight_decay=1e-5,
    device=None,
    validation_dataloader=None,
    checkpoint_dir="checkpoints",
    checkpoint_frequency=5,
    debug_mode=False,
    early_stopping_patience=10,
):
    """
    Train the onset detection model for time series data with enhanced debugging

    Args:
        model: The neural network model to train
        dataloader: Training data loader
        fs: Sampling frequency in Hz
        num_epochs: Number of training epochs
        learning_rate: Initial learning rate
        weight_decay: L2 regularization factor
        device: Torch device (cuda or cpu)
        validation_dataloader: Optional validation data loader
        checkpoint_dir: Directory to save checkpoints
        checkpoint_frequency: How often to save checkpoints (epochs)
        debug_mode: If True, enable verbose debugging output
        early_stopping_patience: Number of epochs with no improvement before stopping

    Returns:
        Trained model
    """
    # Create logger for debugging
    import logging
    import os
    import traceback
    from datetime import datetime

    import matplotlib.pyplot as plt
    from tqdm.auto import tqdm

    log_file = (
        f"{checkpoint_dir}/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )

    # Configure logging
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    logging.basicConfig(
        level=logging.DEBUG if debug_mode else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)

    logger.info(f"Starting training with {num_epochs} epochs")
    logger.info(f"Model architecture:\n{model}")

    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Log device information
    if device.type == "cuda":
        logger.info(f"Training on GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Version: {torch.version.cuda}")
        logger.info(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        logger.info(f"Memory cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    else:
        logger.info("Training on CPU")

    # Move model to device
    model.to(device)

    # Create optimizer with weight decay
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    # History tracking
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    epochs_no_improve = 0

    # Save initial model state
    logger.info("Saving initial model state")
    torch.save(
        {
            "epoch": 0,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        },
        f"{checkpoint_dir}/initial_model.pth",
    )

    # Training loop with progress bar for epochs
    try:
        for epoch in tqdm(range(num_epochs), desc="Epochs", position=0):
            # --- Training phase ---
            model.train()
            running_loss = 0.0
            batch_count = 0

            # Progress bar for batches
            train_pbar = tqdm(
                dataloader,
                desc=f"Training Epoch {epoch + 1}/{num_epochs}",
                leave=False,
                position=1,
            )

            # Debug: log batch information for first epoch
            log_first_batch = debug_mode and epoch == 0

            for batch_idx, batch_data in enumerate(train_pbar):
                # Debug first batch of first epoch in detail
                if log_first_batch and batch_idx == 0:
                    logger.debug(f"First batch shape: {batch_data.shape}")
                    logger.debug(f"Batch data type: {batch_data.dtype}")
                    logger.debug(
                        f"Batch range: min={batch_data.min().item():.4f}, max={batch_data.max().item():.4f}"
                    )

                    # Test model forward pass before full training cycle
                    try:
                        logger.debug("Testing forward pass on first batch...")
                        with torch.no_grad():
                            test_onset, test_offset = model(
                                batch_data.to(device), fs=fs
                            )
                        logger.debug(
                            f"Forward pass successful: onset shape={test_onset.shape}, offset shape={test_offset.shape}"
                        )
                    except Exception as e:
                        logger.error(f"Forward pass test failed: {str(e)}")
                        logger.error(traceback.format_exc())
                        raise

                try:
                    # Move data to device
                    batch_data = batch_data.to(device)

                    # Debug memory usage periodically
                    if debug_mode and batch_idx % 10 == 0 and device.type == "cuda":
                        logger.debug(
                            f"CUDA memory before batch {batch_idx}: allocated {torch.cuda.memory_allocated(0) / 1e9:.2f} GB"
                        )

                    # Zero gradients
                    optimizer.zero_grad()

                    # Generate labels with timing for debugging
                    start_time = datetime.now()
                    onset_gt, offset_gt, binary_activations = generate_onset_labels(
                        batch_data, fs
                    )
                    label_gen_time = (datetime.now() - start_time).total_seconds()

                    if debug_mode and batch_idx % 10 == 0:
                        logger.debug(
                            f"Batch {batch_idx} label generation took {label_gen_time:.2f} seconds"
                        )
                        logger.debug(
                            f"Generated {sum([1 for x in binary_activations.sum(dim=1) if x > 0])} samples with activations"
                        )

                    # Forward pass with timing
                    start_time = datetime.now()
                    onset_pred, offset_pred = model(batch_data, fs=fs)
                    forward_time = (datetime.now() - start_time).total_seconds()

                    if debug_mode and batch_idx % 10 == 0:
                        logger.debug(
                            f"Batch {batch_idx} forward pass took {forward_time:.2f} seconds"
                        )

                    # Calculate loss
                    loss = onset_detection_loss(
                        onset_pred, offset_pred, onset_gt, offset_gt
                    )

                    # Catch NaN losses
                    if torch.isnan(loss):
                        logger.error(
                            f"NaN loss detected in batch {batch_idx}. Skipping batch."
                        )
                        logger.error(
                            f"onset_pred range: {onset_pred.min().item():.4f} to {onset_pred.max().item():.4f}"
                        )
                        logger.error(
                            f"offset_pred range: {offset_pred.min().item():.4f} to {offset_pred.max().item():.4f}"
                        )
                        continue

                    # Backward pass and optimize with timing
                    start_time = datetime.now()
                    loss.backward()

                    # Check for exploding gradients
                    grad_norm = 0.0
                    for p in model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2).item()
                            grad_norm += param_norm**2
                    grad_norm = grad_norm**0.5

                    if grad_norm > 10.0:  # Threshold for exploding gradients
                        logger.warning(
                            f"Exploding gradient detected in batch {batch_idx}: {grad_norm:.2f}. Clipping gradients."
                        )
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)

                    optimizer.step()
                    backward_time = (datetime.now() - start_time).total_seconds()

                    if debug_mode and batch_idx % 10 == 0:
                        logger.debug(
                            f"Batch {batch_idx} backward pass took {backward_time:.2f} seconds"
                        )
                        logger.debug(f"Gradient norm: {grad_norm:.4f}")

                    # Track loss
                    running_loss += loss.item()
                    batch_count += 1

                    # Update progress bar with current loss
                    train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

                except Exception as e:
                    logger.error(f"Error in training batch {batch_idx}: {str(e)}")
                    logger.error(traceback.format_exc())

                    # Debug information for the failed batch
                    logger.error(
                        f"Failed batch shape: {batch_data.shape if 'batch_data' in locals() else 'unknown'}"
                    )
                    try:
                        logger.error(
                            f"Batch min: {batch_data.min().item():.4f}, max: {batch_data.max().item():.4f}"
                        )
                    except:
                        logger.error("Could not log batch min/max values")
                    continue

            # Calculate epoch average loss
            epoch_loss = running_loss / max(1, batch_count)
            train_losses.append(epoch_loss)
            logger.info(
                f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {epoch_loss:.4f}"
            )

            # --- Validation phase ---
            if validation_dataloader is not None:
                model.eval()
                val_loss = 0.0
                val_batch_count = 0

                # Progress bar for validation
                val_pbar = tqdm(
                    validation_dataloader,
                    desc=f"Validating Epoch {epoch + 1}/{num_epochs}",
                    leave=False,
                    position=1,
                )

                with torch.no_grad():
                    for val_batch_idx, val_batch in enumerate(val_pbar):
                        try:
                            # Move data to device
                            val_batch = val_batch.to(device)

                            # Generate labels
                            val_onset_gt, val_offset_gt, _ = generate_onset_labels(
                                val_batch, fs
                            )
                            val_onset_gt = val_onset_gt.to(device)
                            val_offset_gt = val_offset_gt.to(device)

                            # Forward pass
                            val_onset_pred, val_offset_pred = model(val_batch, fs=fs)

                            # Calculate loss
                            batch_loss = onset_detection_loss(
                                val_onset_pred,
                                val_offset_pred,
                                val_onset_gt,
                                val_offset_gt,
                            )

                            # Skip NaN losses
                            if torch.isnan(batch_loss):
                                logger.warning(
                                    f"NaN loss detected in validation batch {val_batch_idx}. Skipping."
                                )
                                continue

                            val_loss += batch_loss.item()
                            val_batch_count += 1

                            # Update validation progress bar
                            val_pbar.set_postfix(
                                {"val_loss": f"{batch_loss.item():.4f}"}
                            )
                        except Exception as e:
                            logger.error(
                                f"Error in validation batch {val_batch_idx}: {str(e)}"
                            )
                            logger.error(traceback.format_exc())
                            continue

                # Calculate validation loss
                avg_val_loss = val_loss / max(1, val_batch_count)
                val_losses.append(avg_val_loss)

                # Print epoch results
                logger.info(
                    f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {epoch_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
                )

                # Update learning rate based on validation loss
                scheduler.step(avg_val_loss)

                # Save best model
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    torch.save(model.state_dict(), f"{checkpoint_dir}/best_model.pth")
                    logger.info(
                        f"Saved new best model with validation loss: {best_val_loss:.4f}"
                    )
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    logger.info(
                        f"No improvement for {epochs_no_improve} epochs (best: {best_val_loss:.4f})"
                    )

                # Early stopping check
                if (
                    early_stopping_patience > 0
                    and epochs_no_improve >= early_stopping_patience
                ):
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break
            else:
                # Update learning rate based on training loss
                scheduler.step(epoch_loss)

            # Log model gradients and weights periodically
            if debug_mode and (epoch + 1) % 5 == 0:
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        logger.debug(f"Layer: {name}")
                        if param.grad is not None:
                            logger.debug(
                                f"  Gradient - Mean: {param.grad.mean().item():.6f}, Std: {param.grad.std().item():.6f}"
                            )
                        logger.debug(
                            f"  Weights - Mean: {param.data.mean().item():.6f}, Std: {param.data.std().item():.6f}"
                        )

            # Save checkpoint periodically
            if (epoch + 1) % checkpoint_frequency == 0:
                checkpoint_path = f"{checkpoint_dir}/checkpoint_epoch_{epoch + 1}.pth"
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "train_losses": train_losses,
                        "val_losses": val_losses if validation_dataloader else None,
                    },
                    checkpoint_path,
                )
                logger.info(
                    f"Saved checkpoint at epoch {epoch + 1} to {checkpoint_path}"
                )

        # Plot training history
        plt.figure(figsize=(10, 6))
        plt.plot(
            range(1, len(train_losses) + 1), train_losses, "b-", label="Training Loss"
        )
        if validation_dataloader is not None and len(val_losses) > 0:
            plt.plot(
                range(1, len(val_losses) + 1), val_losses, "r-", label="Validation Loss"
            )
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(True)
        history_plot_path = f"{checkpoint_dir}/training_history.png"
        plt.savefig(history_plot_path)
        plt.close()
        logger.info(f"Training history plot saved to {history_plot_path}")

        # Save final model
        final_model_path = f"{checkpoint_dir}/final_model.pth"
        torch.save(model.state_dict(), final_model_path)
        logger.info(f"Final model saved to {final_model_path}")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        # Save current model on keyboard interrupt
        interrupted_model_path = (
            f"{checkpoint_dir}/interrupted_model_epoch_{epoch + 1}.pth"
        )
        torch.save(model.state_dict(), interrupted_model_path)
        logger.info(f"Saved model at interruption to {interrupted_model_path}")

    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        logger.error(traceback.format_exc())
        # Save model on error
        error_model_path = f"{checkpoint_dir}/error_model_epoch_{epoch + 1}.pth"
        torch.save(model.state_dict(), error_model_path)
        logger.error(f"Saved model at error to {error_model_path}")
        raise

    # Final memory cleanup
    if device.type == "cuda":
        torch.cuda.empty_cache()
        logger.info("CUDA cache cleared")

    return model


# %%
"""
Main function to run EMG onset detection model

Note: This implementation properly handles time series data by:
1. NOT shuffling the data to preserve temporal order
2. Using appropriate sliding window approach for segmentation
3. Maintaining sequential dependencies in the STFT and neural network
"""
# Data directory
base_dir = Path(r"E:/jpenalozaa")
data_dir = base_dir.joinpath(r"onset_detection\00_")

# Create preprocessing pipeline
# ONLY FILTERING - no envelope detection or normalization
emg_filter = create_emg_filter(
    [
        create_notch_filter(notch_freq=60),  # Remove power line noise
        create_bandpass_filter(
            low_cutoff=20, high_cutoff=2000
        ),  # Keep relevant EMG frequencies
    ]
)

# Only apply filtering in the dataloader
transform = EMGTransformCompose([emg_filter])

# Create dataloader with single channel
# IMPORTANT: No shuffling for time series data to preserve temporal dependencies
channel_idx = 0  # Use first channel
dataloader, fs = create_emg_dataloader(
    data_dir=str(data_dir),
    window_size_sec=5.0,
    window_stride_sec=1.0,
    batch_size=16,
    channels=[channel_idx],  # Single channel
    transform=transform,
    num_workers=0,
    shuffle=False,  # No shuffling for time series data
)

# Print data information
print(f"Data loaded with sampling frequency: {fs} Hz")
# %%
# Get sample batch to check dimensions
sample_batch = next(iter(dataloader))
batch_size, channels, time_steps = sample_batch.shape
print(f"Sample batch shape: {sample_batch.shape}")

# Split data chronologically for training and validation
# For time series, we DON'T use random splitting
data_size = len(dataloader.dataset)
train_size = int(0.8 * data_size)  # Use 80% for training
val_size = data_size - train_size

# Create chronological train/val split (NOT random)
train_dataset = torch.utils.data.Subset(dataloader.dataset, range(0, train_size))
val_dataset = torch.utils.data.Subset(dataloader.dataset, range(train_size, data_size))

# Create new dataloaders with the splits
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=False,  # No shuffling for time series
    num_workers=0,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,  # No shuffling for time series
    num_workers=0,
)

print(f"Training on {train_size} samples, validating on {val_size} samples")
# %%
# Create model with recurrent layers to better handle temporal dependencies
model = STFTOnsetDetectionModel(
    input_channels=channels,
    n_fft=256,
    hop_length=64,
    hidden_channels=64,
    num_layers=3,
)
# %%
# Train model with proper time series handling
trained_model = train_model(
    model=model,
    dataloader=train_loader,  # Use training split
    fs=fs,
    num_epochs=30,
    learning_rate=1e-3,
    weight_decay=1e-5,
)
# %%
sample_batch = next(iter(dataloader))

# Select first segment in the batch
segment = sample_batch[0]

# Add batch dimension if needed
if segment.dim() == 2:
    segment = segment.unsqueeze(0)


# Ensure model is in evaluation mode
model.eval()

# Forward pass with sampling frequency
with torch.no_grad():
    onset_probs, offset_probs = model(segment, fs=fs)

print("Onset probabilities shape:", onset_probs.shape)
print("Offset probabilities shape:", offset_probs.shape)


# %%
