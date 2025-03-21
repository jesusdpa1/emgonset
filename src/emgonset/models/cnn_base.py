"""
cnn_base.py

Defines the CNN model architecture and loss functions for EMG onset detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T


class STFTLayer(nn.Module):
    """
    STFT layer that transforms time-domain signal to time-frequency representation
    """

    def __init__(self, n_fft=256, hop_length=None, win_length=None, normalized=True):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length if hop_length is not None else n_fft // 4
        self.win_length = win_length if win_length is not None else n_fft
        self.normalized = normalized

        # Initialize STFT transform
        self.stft = T.Spectrogram(
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            center=True,
            pad_mode="reflect",
            power=2.0,  # Power spectrogram (magnitude squared)
            normalized=self.normalized,
        )

    def forward(self, x):
        """
        Apply STFT to input

        Args:
            x: Input tensor of shape [batch_size, channels, time]

        Returns:
            Spectrogram of shape [batch_size, channels, freq_bins, time_frames]
        """
        batch_size, channels, time = x.size()
        output = torch.zeros(
            batch_size,
            channels,
            self.n_fft // 2 + 1,
            (time - 1) // self.hop_length + 1,
            device=x.device,
        )

        # Apply STFT to each channel
        for b in range(batch_size):
            for c in range(channels):
                output[b, c] = self.stft(x[b, c])

        return output


class EMGOnsetCNN(nn.Module):
    """
    CNN model for EMG onset detection
    """

    def __init__(self, n_fft=256, hop_length=64, dropout_rate=0.2):
        super().__init__()

        # STFT parameters
        self.n_fft = n_fft
        self.hop_length = hop_length

        # STFT layer
        self.stft_layer = STFTLayer(n_fft=n_fft, hop_length=hop_length)

        # Calculate frequency bins
        n_freq_bins = n_fft // 2 + 1

        # CNN layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(16)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)

        # Convolutional decoder to maintain temporal dimension
        self.conv_decoder = nn.Conv2d(16, 1, kernel_size=3, padding=1)

        # Upsampling to original time resolution
        self.upsample = nn.Upsample(scale_factor=(1, 1), mode="nearest")

        # Final 1D convolution for time-domain output
        self.final_conv = nn.Conv1d(n_freq_bins, 1, kernel_size=3, padding=1)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor of shape [batch_size, channels, time]

        Returns:
            Output tensor of shape [batch_size, 1, time]
        """
        batch_size, channels, time = x.size()

        # Apply STFT to get time-frequency representation
        x = self.stft_layer(x)

        # Reshape if multi-channel
        if channels > 1:
            x = x.view(batch_size * channels, 1, x.size(2), x.size(3))

        # CNN feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.conv3(x)))

        # Decode to single-channel time series
        x = self.conv_decoder(x)

        # Reshape back if multi-channel
        if channels > 1:
            x = x.view(batch_size, channels, x.size(2), x.size(3))

        # Prepare for conversion back to time domain
        x = x.squeeze(1)  # Remove channel dimension

        # Apply 1D convolution across frequency bins to get time domain output
        x = x.permute(0, 2, 1)  # [batch_size, time_frames, freq_bins]
        x = self.final_conv(x)  # [batch_size, 1, time_frames]

        # Interpolate to original time dimension
        target_length = time
        x = F.interpolate(x, size=target_length, mode="linear", align_corners=False)

        # Apply sigmoid to get binary probabilities
        x = torch.sigmoid(x)

        return x


class DiceLoss(nn.Module):
    """
    Dice loss for binary segmentation
    """

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        """
        Calculate Dice loss

        Args:
            predictions: Predicted probabilities
            targets: Target binary mask

        Returns:
            Dice loss value
        """
        # Ensure same shape
        if predictions.shape != targets.shape:
            targets = targets.view(predictions.shape)

        # Flatten
        predictions = predictions.view(-1)
        targets = targets.view(-1)

        # Calculate Dice coefficient
        intersection = (predictions * targets).sum()
        union = predictions.sum() + targets.sum()

        # Calculate Dice loss
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        return 1.0 - dice


class FocalLoss(nn.Module):
    """
    Focal loss for imbalanced binary classification
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, inputs, targets):
        """
        Calculate Focal loss

        Args:
            inputs: Model outputs (logits)
            targets: Target binary mask

        Returns:
            Focal loss value
        """
        # Apply sigmoid explicitly for focal loss calculation
        inputs_sigmoid = torch.sigmoid(inputs)

        # Calculate BCE loss
        bce_loss = self.bce(inputs, targets)

        # Calculate focal weights
        pt = targets * inputs_sigmoid + (1 - targets) * (1 - inputs_sigmoid)
        focal_weight = (1 - pt) ** self.gamma

        # Apply alpha weighting
        if self.alpha is not None:
            alpha_weight = targets * self.alpha + (1 - targets) * (1 - self.alpha)
            focal_weight = alpha_weight * focal_weight

        # Apply weights to BCE loss
        loss = focal_weight * bce_loss

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


def create_emg_onset_model(n_fft=256, hop_length=64, dropout_rate=0.2):
    """
    Factory function to create EMG onset detection model

    Args:
        n_fft: FFT size for STFT
        hop_length: Hop length for STFT
        dropout_rate: Dropout probability for regularization

    Returns:
        Initialized EMGOnsetCNN model
    """
    model = EMGOnsetCNN(n_fft=n_fft, hop_length=hop_length, dropout_rate=dropout_rate)

    # Initialize weights using He initialization
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    return model
