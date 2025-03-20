from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from emgonset.detection.onset import (
    create_adaptive_threshold_detector,
    create_fixed_threshold_detector,
    create_std_threshold_detector,
)
from emgonset.processing.envelope import create_tkeo_envelope
from emgonset.processing.filters import (
    create_bandpass_filter,
    create_emg_filter,
    create_notch_filter,
)
from emgonset.processing.normalization import create_zscore_normalizer
from emgonset.processing.stft import create_stft
from emgonset.processing.transforms import EMGTransformCompose
from emgonset.utils.io import create_emg_dataloader


class OnsetDetectionModel(nn.Module):
    def __init__(
        self,
        input_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 3,
        dropout_rate: float = 0.3,
    ):
        """
        Simple neural network for onset/offset detection from processed EMG signal

        Args:
            input_channels: Number of input channels
            hidden_channels: Number of hidden channels in convolutional layers
            num_layers: Number of convolutional layers
            dropout_rate: Dropout probability for regularization
        """
        super(OnsetDetectionModel, self).__init__()

        # Convolutional layers for feature extraction
        self.conv_layers = nn.ModuleList()

        # First conv layer
        self.conv_layers.append(
            nn.Sequential(
                nn.Conv1d(input_channels, hidden_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            )
        )

        # Additional conv layers
        for _ in range(num_layers - 1):
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        hidden_channels, hidden_channels, kernel_size=3, padding=1
                    ),
                    nn.BatchNorm1d(hidden_channels),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                )
            )

        # Final prediction layers
        self.onset_layer = nn.Conv1d(hidden_channels, 1, kernel_size=1)
        self.offset_layer = nn.Conv1d(hidden_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass of the model

        Args:
            x: Input tensor of shape [batch_size, channels, time_steps]

        Returns:
            Tuple of (onset_probabilities, offset_probabilities)
        """
        # Feature extraction through convolutional layers
        features = x
        for conv_layer in self.conv_layers:
            features = conv_layer(features)

        # Predict onset and offset probabilities
        onset_probs = torch.sigmoid(self.onset_layer(features)).squeeze(1)
        offset_probs = torch.sigmoid(self.offset_layer(features)).squeeze(1)

        return onset_probs, offset_probs


def loss_function(
    onset_pred: torch.Tensor,
    offset_pred: torch.Tensor,
    onset_gt: torch.Tensor,
    offset_gt: torch.Tensor,
) -> torch.Tensor:
    """
    Custom loss function for onset/offset detection

    Args:
        onset_pred: Predicted onset probabilities
        offset_pred: Predicted offset probabilities
        onset_gt: Ground truth onset labels
        offset_gt: Ground truth offset labels

    Returns:
        Combined loss tensor
    """
    # Binary cross-entropy loss for onset and offset detection
    onset_loss = F.binary_cross_entropy(onset_pred, onset_gt.float())
    offset_loss = F.binary_cross_entropy(offset_pred, offset_gt.float())

    # Combine losses
    return onset_loss + offset_loss


def create_pseudo_labels(
    signal: torch.Tensor, fs: float, threshold_type: str = "std"
) -> tuple:
    """
    Create pseudo ground truth labels for onset/offset detection using different detection methods

    Args:
        signal: Input signal tensor [batch_size, channels, time_steps]
        fs: Sampling frequency
        threshold_type: Type of threshold detection method ('fixed', 'std', 'adaptive')

    Returns:
        Tuple of (onset_labels, offset_labels)
    """
    batch_size, channels, time_steps = signal.shape

    # Prepare label tensors
    onset_labels = torch.zeros_like(signal[:, 0, :])
    offset_labels = torch.zeros_like(signal[:, 0, :])

    # Choose detection method
    if threshold_type == "fixed":
        detector = create_fixed_threshold_detector(
            threshold=0.5,  # Adjust as needed
            min_duration_ms=100.0,
            hysteresis=0.1,
            norm="minmax",
        )
    elif threshold_type == "std":
        detector = create_std_threshold_detector(
            std_dev_factor=3.0, baseline_window_ms=500.0, min_duration_ms=100.0
        )
    elif threshold_type == "adaptive":
        detector = create_adaptive_threshold_detector(
            background_window_ms=1000.0,
            detection_window_ms=50.0,
            std_dev_factor=3.0,
            min_duration_ms=100.0,
        )
    else:
        raise ValueError(f"Unknown threshold type: {threshold_type}")

    # Process each sample in the batch
    for b in range(batch_size):
        for ch in range(channels):
            # Initialize detector
            detector.initialize(fs)

            # Detect onsets/offsets for this channel
            detection_result = detector(signal[b, ch].numpy())

            # Convert detection results to labels
            for onset in detection_result.onsets:
                onset_idx = max(0, min(int(onset.sample_index), time_steps - 1))
                onset_labels[b, onset_idx] = 1.0

            for offset in detection_result.offsets:
                offset_idx = max(0, min(int(offset.sample_index), time_steps - 1))
                offset_labels[b, offset_idx] = 1.0

    return onset_labels, offset_labels


def train_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    fs: float,
    num_epochs: int = 100,
    learning_rate: float = 1e-3,
    threshold_type: str = "std",
):
    """
    Training loop for the onset detection model

    Args:
        model: Onset detection model
        dataloader: PyTorch DataLoader with EMG data
        fs: Sampling frequency
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        threshold_type: Type of threshold detection method
    """
    # Move model to appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        epoch_loss = 0.0

        for batch_idx, batch_data in enumerate(dataloader):
            # Move data to device
            batch_data = batch_data.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Create pseudo ground truth labels using real detection methods
            onset_gt, offset_gt = create_pseudo_labels(
                batch_data, fs=fs, threshold_type=threshold_type
            )

            # Move ground truth to device
            onset_gt = onset_gt.to(device)
            offset_gt = offset_gt.to(device)

            # Forward pass
            onset_pred, offset_pred = model(batch_data)

            # Compute loss
            loss = loss_function(onset_pred, offset_pred, onset_gt, offset_gt)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Accumulate loss
            epoch_loss += loss.item()

        # Print average epoch loss
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Avg Loss: {epoch_loss / len(dataloader):.4f}"
        )

    return model


def main():
    # Set up data loading similar to your original script
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
        batch_size=32,  # Increased batch size for training
        channels=[0, 1],
        transform=transform,
        num_workers=0,
        shuffle=True,
    )

    # Print some info about the data
    print(f"Dataloader created with sampling frequency: {fs} Hz")

    # Get sample batch to determine input dimensions
    sample_batch = next(iter(dataloader))
    channels, time_steps = sample_batch.shape[1], sample_batch.shape[2]

    # Initialize model
    model = OnsetDetectionModel(
        input_channels=channels, hidden_channels=64, num_layers=3
    )

    # Train model
    trained_model = train_model(
        model, dataloader, fs, num_epochs=50, learning_rate=1e-3, threshold_type="std"
    )


if __name__ == "__main__":
    main()
