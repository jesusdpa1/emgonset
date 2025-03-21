from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from numba import njit
from scipy import signal


# Numba-optimized functions
@njit(cache=True)
def _abs_rectify(x):
    """Numba-accelerated absolute value rectification"""
    return np.abs(x)


@njit(cache=True)
def _square_rectify(x):
    """Numba-accelerated square rectification"""
    return np.square(x)


# Hilbert transform cannot be directly njit-ed because it uses scipy.signal,
# but we can optimize the envelope calculation part
def _hilbert_transform(x):
    """Compute Hilbert transform (not njit-able due to scipy dependency)"""
    analytic_signal = signal.hilbert(x)
    return analytic_signal


@njit(cache=True)
def _compute_envelope(analytic_signal_real, analytic_signal_imag):
    """Numba-accelerated envelope calculation from analytic signal components"""
    # Manually compute absolute value of complex numbers
    return np.sqrt(analytic_signal_real**2 + analytic_signal_imag**2)


class BaseRectifier(ABC):
    """Base class for all EMG rectification methods"""

    def __init__(self):
        self.fs: Optional[float] = None
        self.is_initialized: bool = False

    def initialize(self, fs: float) -> None:
        """Initialize rectifier with sampling frequency if needed"""
        self.fs = fs
        self.is_initialized = True

    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply rectification to signal"""
        pass


class AbsoluteValueRectifier(BaseRectifier):
    """Full-wave rectification using absolute value"""

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply absolute value rectification"""
        return _abs_rectify(x)


class SquareRectifier(BaseRectifier):
    """Rectification by squaring the signal"""

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply square rectification"""
        return _square_rectify(x)


class HilbertRectifier(BaseRectifier):
    """Rectification using Hilbert transform envelope"""

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Apply Hilbert transform to get signal envelope

        This computes the analytic signal using the Hilbert transform
        and returns the envelope (magnitude of the analytic signal)
        """
        # Get analytic signal (signal + i * hilbert(signal))
        analytic_signal = _hilbert_transform(x)

        # Split into real and imaginary parts for Numba
        real_part = analytic_signal.real
        imag_part = analytic_signal.imag

        # Get amplitude envelope using optimized function
        amplitude_envelope = _compute_envelope(real_part, imag_part)

        return amplitude_envelope


# Factory functions (unchanged)
def create_abs_rectifier() -> AbsoluteValueRectifier:
    """Create a full-wave rectifier using absolute value"""
    return AbsoluteValueRectifier()


def create_square_rectifier() -> SquareRectifier:
    """Create a rectifier that squares the signal"""
    return SquareRectifier()


def create_hilbert_rectifier() -> HilbertRectifier:
    """Create a rectifier using Hilbert transform envelope detection"""
    return HilbertRectifier()
