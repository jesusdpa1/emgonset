from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch
from numba import njit

from ..utils.internals import public_api


class BaseTKEO(ABC):
    """Base class for Teager-Kaiser Energy Operator implementations"""

    def __init__(self):
        self.fs: Optional[float] = None
        self.is_initialized: bool = False

    def initialize(self, fs: float) -> None:
        """Initialize TKEO with sampling frequency if needed"""
        self.fs = fs
        self.is_initialized = True

    @abstractmethod
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply TKEO to signal tensor"""
        pass


# Define Numba accelerated functions for each TKEO variant
@njit
def _tkeo_operation(signal):
    """Numba accelerated implementation of classic TKEO"""
    result = np.zeros_like(signal)
    if len(signal) > 2:
        # Calculate central part: x[n]² - x[n-1] * x[n+1]
        inner_result = signal[1:-1] ** 2 - signal[:-2] * signal[2:]

        # Pad result to maintain original length
        result[1:-1] = inner_result
        if len(inner_result) > 0:
            result[0] = inner_result[0]
            result[-1] = inner_result[-1]

    return result


@njit
def _tkeo2_operation(signal):
    """Numba accelerated implementation of TKEO2"""
    result = np.zeros_like(signal)
    if len(signal) > 3:
        # Parameters for indexing
        l, p, q, s = 1, 2, 0, 3

        # Calculate x[n+1] * x[n+2] - x[n] * x[n+3]
        inner_result = signal[l:-p] * signal[p:-l] - signal[q:-s] * signal[s:]

        # Pad result to maintain original length
        result[l:-p] = inner_result
        if len(inner_result) > 0:
            result[:l] = inner_result[0]
            result[-p:] = inner_result[-1]

    return result


@njit
def _mtkeo_operation(signal, k1, k2, k3):
    """Numba accelerated implementation of MTKEO"""
    result = np.zeros_like(signal)
    if len(signal) > 5:
        # Standard TKEO: x[n]^2 - x[n-1]*x[n+1]
        tkeo = signal[2:-2] ** 2 - signal[1:-3] * signal[3:-1]

        # TKEO with delay 2: x[n]^2 - x[n-2]*x[n+2]
        tkeo1 = signal[2:-2] ** 2 - signal[0:-4] * signal[4:]

        # TKEO with delay 3: x[n-1]*x[n+1] - x[n-2]*x[n+2]
        tkeo2 = signal[1:-3] * signal[3:-1] - signal[0:-4] * signal[4:]

        # Combine with weights
        inner_result = k1 * tkeo + k2 * tkeo1 + k3 * tkeo2

        # Pad result to maintain original length
        result[2:-2] = inner_result
        if len(inner_result) > 0:
            result[:2] = inner_result[0]
            result[-2:] = inner_result[-1]

    return result


@public_api
class TKEO(BaseTKEO):
    """
    Classic Teager-Kaiser Energy Operator using 3 samples.

    Implementation follows Li et al., 2007:
    TKEO[x[n]] = x[n]² - x[n-1] * x[n+1]
    """

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply classic TKEO to the signal

        Args:
            tensor: PyTorch tensor of shape [channels, samples]

        Returns:
            TKEO result tensor with same shape as input
        """
        # Process each channel
        result = torch.zeros_like(tensor)

        for ch in range(tensor.shape[0]):
            # Get signal for this channel
            signal = tensor[ch].detach().cpu().numpy()

            # Apply Numba-accelerated TKEO
            channel_result = _tkeo_operation(signal)

            # Convert back to tensor
            result[ch] = torch.tensor(
                channel_result, dtype=tensor.dtype, device=tensor.device
            )

        return result


@public_api
class TKEO2(BaseTKEO):
    """
    Alternative Teager-Kaiser Energy Operator using 4 samples.

    Implementation follows Deburchgrave et al., 2008:
    TKEO2[x[n]] = x[n+1] * x[n+2] - x[n] * x[n+3]
    """

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply TKEO2 to the signal

        Args:
            tensor: PyTorch tensor of shape [channels, samples]

        Returns:
            TKEO2 result tensor with same shape as input
        """
        # Process each channel
        result = torch.zeros_like(tensor)

        for ch in range(tensor.shape[0]):
            # Get signal for this channel
            signal = tensor[ch].detach().cpu().numpy()

            # Apply Numba-accelerated TKEO2
            channel_result = _tkeo2_operation(signal)

            # Convert back to tensor
            result[ch] = torch.tensor(
                channel_result, dtype=tensor.dtype, device=tensor.device
            )

        return result


@public_api
class MTKEO(BaseTKEO):
    """
    Modified Teager-Kaiser Energy Operator (MTKEO).

    This extends the classic TKEO by combining multiple time-delay versions
    to better handle noise and frequency changes.

    MTKEO = k1*TKEO(x) + k2*TKEO1(x) + k3*TKEO2(x)

    Where TKEO1 and TKEO2 use different time delays, and k1, k2, k3
    are weighting coefficients.
    """

    def __init__(self, k1: float = 1.0, k2: float = 1.0, k3: float = 1.0):
        """
        Initialize MTKEO with weighting coefficients

        Args:
            k1: Weight for standard TKEO (default 1.0)
            k2: Weight for TKEO with delay 2 (default 1.0)
            k3: Weight for TKEO with delay 3 (default 1.0)
        """
        super().__init__()
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply MTKEO to the signal

        Args:
            tensor: PyTorch tensor of shape [channels, samples]

        Returns:
            MTKEO result tensor with same shape as input
        """
        # Process each channel
        result = torch.zeros_like(tensor)

        for ch in range(tensor.shape[0]):
            # Get signal for this channel
            signal = tensor[ch].detach().cpu().numpy()

            # Apply Numba-accelerated MTKEO
            channel_result = _mtkeo_operation(signal, self.k1, self.k2, self.k3)

            # Convert back to tensor
            result[ch] = torch.tensor(
                channel_result, dtype=tensor.dtype, device=tensor.device
            )

        return result


# Factory functions remain the same
@public_api
def create_tkeo() -> TKEO:
    """Create a classic TKEO transform"""
    return TKEO()


@public_api
def create_tkeo2() -> TKEO2:
    """Create a 4-sample TKEO transform (TKEO2)"""
    return TKEO2()


@public_api
def create_mtkeo(k1: float = 1.0, k2: float = 1.0, k3: float = 1.0) -> MTKEO:
    """
    Create a Modified TKEO (MTKEO) transform

    Args:
        k1: Weight for standard TKEO
        k2: Weight for TKEO with delay 2
        k3: Weight for TKEO with delay 3
    """
    return MTKEO(k1=k1, k2=k2, k3=k3)
