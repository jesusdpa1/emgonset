"""
Padding utilities for signal processing to reduce edge effects.
"""

import numpy as np
import torch

from ..utils.internals import public_api


@public_api
def mirror_pad_numpy(signal: np.ndarray, pad_length: int) -> np.ndarray:
    """
    Apply mirror padding to a numpy signal

    Args:
        signal: Input signal array
        pad_length: Number of samples to pad on each end

    Returns:
        Padded signal with mirrored edges
    """
    # Handle case where pad_length is larger than signal length
    effective_pad = min(pad_length, len(signal))

    if effective_pad <= 0:
        return (
            signal.copy()
        )  # Return a copy of the original signal if no padding needed

    # Left padding (reversed first pad_length samples)
    left_pad = np.flip(signal[:effective_pad])
    # Right padding (reversed last pad_length samples)
    right_pad = np.flip(signal[-effective_pad:])
    # Return padded signal
    return np.concatenate([left_pad, signal, right_pad])


@public_api
def unpad_numpy(signal: np.ndarray, pad_length: int) -> np.ndarray:
    """
    Remove padding from a padded signal

    Args:
        signal: Padded signal array
        pad_length: Number of samples that were padded on each end

    Returns:
        Original signal with padding removed
    """
    if pad_length <= 0 or len(signal) <= 2 * pad_length:
        return signal.copy()  # Return a copy if no unpadding needed or signal too short

    return signal[pad_length:-pad_length].copy()


@public_api
def mirror_pad_torch(tensor: torch.Tensor, pad_length: int) -> torch.Tensor:
    """
    Apply mirror padding to a PyTorch tensor

    Args:
        tensor: Input tensor (1D or 2D with shape [channels, samples])
        pad_length: Number of samples to pad on each end

    Returns:
        Padded tensor with mirrored edges
    """
    if pad_length <= 0:
        return tensor.clone()  # Return a clone if no padding needed

    # Handle 1D tensor case
    if tensor.dim() == 1:
        # Use PyTorch's functional pad with 'reflect' mode
        return torch.nn.functional.pad(
            tensor.unsqueeze(0), (pad_length, pad_length), mode="reflect"
        ).squeeze(0)

    # Handle 2D tensor case [channels, samples]
    elif tensor.dim() == 2:
        return torch.nn.functional.pad(tensor, (pad_length, pad_length), mode="reflect")

    else:
        raise ValueError(
            f"Unsupported tensor dimension: {tensor.dim()}, expected 1D or 2D tensor"
        )


@public_api
def unpad_torch(tensor: torch.Tensor, pad_length: int) -> torch.Tensor:
    """
    Remove padding from a padded tensor

    Args:
        tensor: Padded tensor (1D or 2D with shape [channels, samples])
        pad_length: Number of samples that were padded on each end

    Returns:
        Original tensor with padding removed
    """
    if pad_length <= 0:
        return tensor.clone()  # Return a clone if no unpadding needed

    # Handle 1D tensor case
    if tensor.dim() == 1:
        if len(tensor) <= 2 * pad_length:
            return tensor.clone()  # Return a clone if tensor too short
        return tensor[pad_length:-pad_length].clone()

    # Handle 2D tensor case [channels, samples]
    elif tensor.dim() == 2:
        if tensor.shape[1] <= 2 * pad_length:
            return tensor.clone()  # Return a clone if tensor too short
        return tensor[:, pad_length:-pad_length].clone()

    else:
        raise ValueError(
            f"Unsupported tensor dimension: {tensor.dim()}, expected 1D or 2D tensor"
        )


@public_api
def calculate_pad_length(fs: float, pad_time_ms: float = 100.0) -> int:
    """
    Calculate the padding length in samples based on sampling frequency and desired time

    Args:
        fs: Sampling frequency in Hz
        pad_time_ms: Padding time in milliseconds (default: 100ms)

    Returns:
        Padding length in samples
    """
    return int(fs * pad_time_ms / 1000)
