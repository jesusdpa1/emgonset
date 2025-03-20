"""
/src/emgonset/processing/normalization.py
Normalization methods for EMG signals
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import numpy as np
import torch
from scipy import signal

from ..utils.internals import public_api


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
            channel_data = tensor[ch]

            # Calculate min and max
            min_val = torch.min(channel_data)
            max_val = torch.max(channel_data)

            # Avoid division by zero
            if max_val > min_val:
                # Scale to [0, 1] first
                normalized = (channel_data - min_val) / (max_val - min_val)

                # Then scale to target range
                normalized = (
                    normalized * (self.target_max - self.target_min) + self.target_min
                )

                result[ch] = normalized
            else:
                # If all values are the same, set to target_min
                result[ch] = torch.ones_like(channel_data) * self.target_min

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
            channel_data = tensor[ch]

            # Calculate mean and standard deviation
            mean_val = torch.mean(channel_data)
            std_val = torch.std(channel_data)

            # Avoid division by zero
            if std_val > 0:
                result[ch] = (channel_data - mean_val) / (std_val + self.eps)
            else:
                # If std is zero, just subtract mean
                result[ch] = channel_data - mean_val

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
            channel_data = tensor[ch]

            # Convert to numpy for quantile calculation
            np_data = channel_data.detach().cpu().numpy()

            # Calculate median and IQR
            median = np.median(np_data)
            q1 = np.percentile(np_data, 25)
            q3 = np.percentile(np_data, 75)
            iqr = q3 - q1

            # Avoid division by zero
            if iqr > 0:
                # Apply normalization
                result[ch] = (channel_data - median) / (iqr + self.eps)
            else:
                # If IQR is zero, just subtract median
                result[ch] = channel_data - median

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
            channel_data = tensor[ch]

            # Calculate L2 norm
            norm = torch.sqrt(torch.sum(channel_data**2))

            # Avoid division by zero
            if norm > 0:
                result[ch] = channel_data / (norm + self.eps)
            else:
                result[ch] = channel_data

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
            channel_data = tensor[ch]

            # Calculate max absolute value
            max_abs = torch.max(torch.abs(channel_data))

            # Avoid division by zero
            if max_abs > 0:
                result[ch] = (channel_data / (max_abs + self.eps)) * self.scale
            else:
                result[ch] = channel_data

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

        # Handle single reference value
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


# Factory functions


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
