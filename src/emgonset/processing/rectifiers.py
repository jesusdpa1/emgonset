from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from scipy import signal


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
        return np.abs(x)


class SquareRectifier(BaseRectifier):
    """Rectification by squaring the signal"""

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply square rectification"""
        return np.square(x)


class HilbertRectifier(BaseRectifier):
    """Rectification using Hilbert transform envelope"""

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Apply Hilbert transform to get signal envelope

        This computes the analytic signal using the Hilbert transform
        and returns the envelope (magnitude of the analytic signal)
        """
        # Get analytic signal (signal + i * hilbert(signal))
        analytic_signal = signal.hilbert(x)

        # Get amplitude envelope
        amplitude_envelope = np.abs(analytic_signal)

        return amplitude_envelope


# Factory functions
def create_abs_rectifier() -> AbsoluteValueRectifier:
    """Create a full-wave rectifier using absolute value"""
    return AbsoluteValueRectifier()


def create_square_rectifier() -> SquareRectifier:
    """Create a rectifier that squares the signal"""
    return SquareRectifier()


def create_hilbert_rectifier() -> HilbertRectifier:
    """Create a rectifier using Hilbert transform envelope detection"""
    return HilbertRectifier()
