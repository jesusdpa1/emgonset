"""
/src/emgonset/processing/normalization.py
Normalization methods for EMG signals
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import numpy as np
import torch
from numba import njit, prange
from scipy import signal

from ..utils.internals import public_api


# Numba-optimized normalization functions
@njit
def _minmax_normalize(data, target_min, target_max):
    """Numba-accelerated min-max normalization"""
    result = np.zeros_like(data)

    min_val = np.min(data)
    max_val = np.max(data)

    if max_val > min_val:
        # Scale to [0, 1] first
        normalized = (data - min_val) / (max_val - min_val)
        # Then scale to target range
        result = normalized * (target_max - target_min) + target_min
    else:
        # If all values are the same, set to target_min
        result.fill(target_min)

    return result


@njit
def _zscore_normalize(data, eps):
    """Numba-accelerated z-score normalization"""
    result = np.zeros_like(data)

    mean_val = np.mean(data)
    std_val = np.std(data)

    if std_val > 0:
        result = (data - mean_val) / (std_val + eps)
    else:
        # If std is zero, just subtract mean
        result = data - mean_val

    return result


@njit
def _unit_length_normalize(data, eps):
    """Numba-accelerated unit length normalization"""
    result = np.zeros_like(data)

    # Calculate L2 norm
    norm = np.sqrt(np.sum(data**2))

    if norm > 0:
        result = data / (norm + eps)
    else:
        result = data.copy()

    return result


@njit
def _max_amplitude_normalize(data, scale, eps):
    """Numba-accelerated max amplitude normalization"""
    result = np.zeros_like(data)

    # Calculate max absolute value
    max_abs = np.max(np.abs(data))

    if max_abs > 0:
        result = (data / (max_abs + eps)) * scale
    else:
        result = data.copy()

    return result


# For robust scale normalization, we can't use pure Numba because of percentile
def _robust_scale_normalize(data, eps):
    """Hybrid implementation of robust scale normalization"""
    median = np.median(data)
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1

    # Then use numba for the actual normalization
    return _apply_robust_norm(data, median, iqr, eps)


@njit
def _apply_robust_norm(data, median, iqr, eps):
    """Numba-accelerated part of robust normalization"""
    result = np.zeros_like(data)

    if iqr > 0:
        result = (data - median) / (iqr + eps)
    else:
        result = data - median

    return result


class BaseNormalizer(ABC):
    """Base class for all EMG normalization methods"""

    def __init__(self):
        self.fs: Optional[float] = None
        self.is_initialized: bool = False

    def initialize(self, fs: float) -> None:
        """Initialize normalizer with sampling frequency if needed"""
        self.fs = fs
        self.is_initialized = True

    @abstractmethod
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply normalization to signal tensor"""
        pass


@public_api
class MinMaxNormalizer(BaseNormalizer):
    """
    Min-max normalization scales the signal to a range of [0, 1] or custom range.

    This normalizer scales each channel independently based on its min and max values.
    """

    def __init__(self, target_min: float = 0.0, target_max: float = 1.0):
        """
        Initialize min-max normalizer

        Args:
            target_min: Target minimum value after normalization
            target_max: Target maximum value after normalization
        """
        super().__init__()
        self.target_min = target_min
        self.target_max = target_max

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply min-max normalization to the signal

        Args:
            tensor: PyTorch tensor of shape [channels, samples]

        Returns:
            Normalized tensor with same shape as input
        """
        # Process each channel independently
        result = torch.zeros_like(tensor)

        for ch in range(tensor.shape[0]):
            # Get channel data as numpy array
            channel_data = tensor[ch].detach().cpu().numpy()

            # Apply Numba-accelerated normalization
            norm_data = _minmax_normalize(
                channel_data, self.target_min, self.target_max
            )

            # Convert back to tensor
            result[ch] = torch.tensor(
                norm_data, dtype=tensor.dtype, device=tensor.device
            )

        return result


@public_api
class ZScoreNormalizer(BaseNormalizer):
    """
    Z-score normalization centers the signal around zero with unit variance.

    This normalizer transforms each channel independently based on its mean and standard deviation.
    """

    def __init__(self, eps: float = 1e-10):
        """
        Initialize z-score normalizer

        Args:
            eps: Small value to avoid division by zero
        """
        super().__init__()
        self.eps = eps

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply z-score normalization to the signal

        Args:
            tensor: PyTorch tensor of shape [channels, samples]

        Returns:
            Normalized tensor with same shape as input
        """
        # Process each channel independently
        result = torch.zeros_like(tensor)

        for ch in range(tensor.shape[0]):
            # Get channel data as numpy array
            channel_data = tensor[ch].detach().cpu().numpy()

            # Apply Numba-accelerated normalization
            norm_data = _zscore_normalize(channel_data, self.eps)

            # Convert back to tensor
            result[ch] = torch.tensor(
                norm_data, dtype=tensor.dtype, device=tensor.device
            )

        return result


@public_api
class RobustScaleNormalizer(BaseNormalizer):
    """
    Robust scaling normalizes based on median and interquartile range.

    This is less sensitive to outliers than min-max or z-score normalization.
    """

    def __init__(self, eps: float = 1e-10):
        """
        Initialize robust scale normalizer

        Args:
            eps: Small value to avoid division by zero
        """
        super().__init__()
        self.eps = eps

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply robust scaling to the signal

        Args:
            tensor: PyTorch tensor of shape [channels, samples]

        Returns:
            Normalized tensor with same shape as input
        """
        # Process each channel independently
        result = torch.zeros_like(tensor)

        for ch in range(tensor.shape[0]):
            # Get channel data as numpy array
            channel_data = tensor[ch].detach().cpu().numpy()

            # Apply hybrid normalization (part NumPy, part Numba)
            norm_data = _robust_scale_normalize(channel_data, self.eps)

            # Convert back to tensor
            result[ch] = torch.tensor(
                norm_data, dtype=tensor.dtype, device=tensor.device
            )

        return result


@public_api
class UnitLengthNormalizer(BaseNormalizer):
    """
    Unit length normalization scales the signal to have unit (L2) norm.

    This normalizer scales each channel independently to have a Euclidean length of 1.
    """

    def __init__(self, eps: float = 1e-10):
        """
        Initialize unit length normalizer

        Args:
            eps: Small value to avoid division by zero
        """
        super().__init__()
        self.eps = eps

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply unit length normalization to the signal

        Args:
            tensor: PyTorch tensor of shape [channels, samples]

        Returns:
            Normalized tensor with same shape as input
        """
        # Process each channel independently
        result = torch.zeros_like(tensor)

        for ch in range(tensor.shape[0]):
            # Get channel data as numpy array
            channel_data = tensor[ch].detach().cpu().numpy()

            # Apply Numba-accelerated normalization
            norm_data = _unit_length_normalize(channel_data, self.eps)

            # Convert back to tensor
            result[ch] = torch.tensor(
                norm_data, dtype=tensor.dtype, device=tensor.device
            )

        return result


@public_api
class MaxAmplitudeNormalizer(BaseNormalizer):
    """
    Max amplitude normalization scales the signal by its maximum absolute value.

    This normalizer scales each channel independently to have a max absolute value of 1.
    """

    def __init__(self, scale: float = 1.0, eps: float = 1e-10):
        """
        Initialize max amplitude normalizer

        Args:
            scale: Scale factor to apply after normalization
            eps: Small value to avoid division by zero
        """
        super().__init__()
        self.scale = scale
        self.eps = eps

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply max amplitude normalization to the signal

        Args:
            tensor: PyTorch tensor of shape [channels, samples]

        Returns:
            Normalized tensor with same shape as input
        """
        # Process each channel independently
        result = torch.zeros_like(tensor)

        for ch in range(tensor.shape[0]):
            # Get channel data as numpy array
            channel_data = tensor[ch].detach().cpu().numpy()

            # Apply Numba-accelerated normalization
            norm_data = _max_amplitude_normalize(channel_data, self.scale, self.eps)

            # Convert back to tensor
            result[ch] = torch.tensor(
                norm_data, dtype=tensor.dtype, device=tensor.device
            )

        return result


@public_api
class ReferenceNormalizer(BaseNormalizer):
    """
    Normalizes EMG signal based on a reference value or signal.

    This is useful for normalizing by maximum voluntary contraction (MVC)
    or other reference values.
    """

    def __init__(
        self,
        reference_values: Optional[Union[float, torch.Tensor, np.ndarray]] = None,
        per_channel: bool = True,
        eps: float = 1e-10,
    ):
        """
        Initialize reference normalizer

        Args:
            reference_values: Reference value(s) for normalization. Can be:
                             - Single float: Same value used for all channels
                             - Tensor/array of values: One value per channel
                             If None, reference must be set later via set_reference
            per_channel: Whether to apply normalization per channel
            eps: Small value to avoid division by zero
        """
        super().__init__()
        self.per_channel = per_channel
        self.eps = eps
        self.reference_values = None

        if reference_values is not None:
            self.set_reference(reference_values)

    def set_reference(self, reference_values: Union[float, torch.Tensor, np.ndarray]):
        """
        Set reference values for normalization

        Args:
            reference_values: Reference value(s) for normalization
        """
        if isinstance(reference_values, (int, float)):
            self.reference_values = float(reference_values)
        elif isinstance(reference_values, np.ndarray):
            self.reference_values = torch.tensor(reference_values, dtype=torch.float32)
        elif isinstance(reference_values, torch.Tensor):
            self.reference_values = reference_values.clone().detach().float()
        else:
            raise TypeError(
                "Reference values must be a float, numpy array, or torch tensor"
            )

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply reference-based normalization to the signal

        Args:
            tensor: PyTorch tensor of shape [channels, samples]

        Returns:
            Normalized tensor with same shape as input
        """
        if self.reference_values is None:
            raise ValueError("Reference values not set. Call set_reference() first.")

        # For ReferenceNormalizer, the operations are simple enough that
        # the overhead of converting to NumPy might outweigh benefits
        # But we could optimize the per-channel case with Numba

        # Handle single reference value - keep in PyTorch for simplicity
        if isinstance(self.reference_values, float):
            return tensor / (self.reference_values + self.eps)

        # Handle per-channel normalization
        if self.per_channel:
            if self.reference_values.numel() != tensor.shape[0]:
                raise ValueError(
                    f"Number of reference values ({self.reference_values.numel()}) "
                    f"must match number of channels ({tensor.shape[0]})"
                )

            result = torch.zeros_like(tensor)
            for ch in range(tensor.shape[0]):
                result[ch] = tensor[ch] / (self.reference_values[ch] + self.eps)
            return result
        else:
            # Apply same reference values to all channels
            return tensor / (self.reference_values + self.eps)


# Factory functions - unchanged
@public_api
def create_minmax_normalizer(
    target_min: float = 0.0, target_max: float = 1.0
) -> MinMaxNormalizer:
    """
    Create a min-max normalizer

    Args:
        target_min: Target minimum value after normalization
        target_max: Target maximum value after normalization

    Returns:
        Configured MinMaxNormalizer
    """
    return MinMaxNormalizer(target_min=target_min, target_max=target_max)


@public_api
def create_zscore_normalizer(eps: float = 1e-10) -> ZScoreNormalizer:
    """
    Create a z-score normalizer

    Args:
        eps: Small value to avoid division by zero

    Returns:
        Configured ZScoreNormalizer
    """
    return ZScoreNormalizer(eps=eps)


@public_api
def create_robust_normalizer(eps: float = 1e-10) -> RobustScaleNormalizer:
    """
    Create a robust scale normalizer

    Args:
        eps: Small value to avoid division by zero

    Returns:
        Configured RobustScaleNormalizer
    """
    return RobustScaleNormalizer(eps=eps)


@public_api
def create_unit_length_normalizer(eps: float = 1e-10) -> UnitLengthNormalizer:
    """
    Create a unit length normalizer

    Args:
        eps: Small value to avoid division by zero

    Returns:
        Configured UnitLengthNormalizer
    """
    return UnitLengthNormalizer(eps=eps)


@public_api
def create_max_amplitude_normalizer(
    scale: float = 1.0, eps: float = 1e-10
) -> MaxAmplitudeNormalizer:
    """
    Create a max amplitude normalizer

    Args:
        scale: Scale factor to apply after normalization
        eps: Small value to avoid division by zero

    Returns:
        Configured MaxAmplitudeNormalizer
    """
    return MaxAmplitudeNormalizer(scale=scale, eps=eps)


@public_api
def create_reference_normalizer(
    reference_values: Optional[Union[float, torch.Tensor, np.ndarray]] = None,
    per_channel: bool = True,
    eps: float = 1e-10,
) -> ReferenceNormalizer:
    """
    Create a reference-based normalizer

    Args:
        reference_values: Reference value(s) for normalization
        per_channel: Whether to apply normalization per channel
        eps: Small value to avoid division by zero

    Returns:
        Configured ReferenceNormalizer
    """
    return ReferenceNormalizer(
        reference_values=reference_values, per_channel=per_channel, eps=eps
    )
