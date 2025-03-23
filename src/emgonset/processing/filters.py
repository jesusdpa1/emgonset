from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
import torch
from scipy import signal

from ..utils.internals import public_api


class BaseFilter(ABC):
    """Base class for all EMG filters that can be initialized later"""

    def __init__(self):
        self.fs: Optional[float] = None
        self.is_initialized: bool = False
        self.min_signal_length: int = 32  # Minimum signal length to apply filtering

    @abstractmethod
    def initialize(self, fs: float) -> None:
        """Initialize filter with sampling frequency"""
        pass

    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply filter to signal"""
        pass

    def _check_signal_length(self, x: np.ndarray) -> bool:
        """Check if signal is long enough for filtering"""
        return len(x) >= self.min_signal_length


@public_api
class LowpassFilter(BaseFilter):
    """Lowpass filter for EMG signals"""

    def __init__(self, cutoff: float, order: int = 4):
        """
        Initialize lowpass filter

        Args:
            cutoff: Cutoff frequency in Hz
            order: Filter order
        """
        super().__init__()
        self.cutoff = cutoff
        self.order = order
        self.sos = None

    def initialize(self, fs: float) -> None:
        self.fs = fs
        nyquist = 0.5 * fs
        normalized_cutoff = self.cutoff / nyquist
        self.sos = signal.butter(
            self.order, normalized_cutoff, btype="low", output="sos"
        )
        # Set minimum signal length based on filter order
        # Higher order filters need longer signals
        self.min_signal_length = max(32, 4 * self.order)
        self.is_initialized = True

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if not self.is_initialized:
            raise RuntimeError("Filter not initialized. Call initialize(fs) first.")

        # Return original signal if too short for filtering
        if not self._check_signal_length(x):
            return x.copy()  # Return a copy to maintain consistency

        try:
            # Try to use second-order sections for stability
            filtered = signal.sosfiltfilt(self.sos, x, padtype="constant")
            return filtered
        except ValueError as e:
            # If that fails, try to fall back to standard filter with minimal padding
            try:
                # Fallback to a regular filtfilt with minimal padding
                b, a = signal.sos2tf(self.sos)
                padlen = min(
                    3 * self.order, len(x) - 1
                )  # Ensure padlen < signal length
                filtered = signal.filtfilt(b, a, x, padlen=padlen)
                return filtered
            except Exception:
                # If all else fails, return the original signal
                return x.copy()


@public_api
class HighpassFilter(BaseFilter):
    """Highpass filter for EMG signals"""

    def __init__(self, cutoff: float, order: int = 4):
        """
        Initialize highpass filter

        Args:
            cutoff: Cutoff frequency in Hz
            order: Filter order
        """
        super().__init__()
        self.cutoff = cutoff
        self.order = order
        self.sos = None

    def initialize(self, fs: float) -> None:
        self.fs = fs
        nyquist = 0.5 * fs
        normalized_cutoff = self.cutoff / nyquist
        self.sos = signal.butter(
            self.order, normalized_cutoff, btype="high", output="sos"
        )
        # Set minimum signal length based on filter order
        self.min_signal_length = max(32, 4 * self.order)
        self.is_initialized = True

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if not self.is_initialized:
            raise RuntimeError("Filter not initialized. Call initialize(fs) first.")

        # Return original signal if too short for filtering
        if not self._check_signal_length(x):
            return x.copy()

        try:
            # Try to use second-order sections for stability
            filtered = signal.sosfiltfilt(self.sos, x, padtype="constant")
            return filtered
        except ValueError as e:
            # If that fails, try to fall back to standard filter with minimal padding
            try:
                # Fallback to a regular filtfilt with minimal padding
                b, a = signal.sos2tf(self.sos)
                padlen = min(
                    3 * self.order, len(x) - 1
                )  # Ensure padlen < signal length
                filtered = signal.filtfilt(b, a, x, padlen=padlen)
                return filtered
            except Exception:
                # If all else fails, return the original signal
                return x.copy()


@public_api
class BandpassFilter(BaseFilter):
    """Bandpass filter for EMG signals"""

    def __init__(
        self,
        low_cutoff: float,
        high_cutoff: float,
        order: int = 4,
    ):
        """
        Initialize bandpass filter

        Args:
            low_cutoff: Lower cutoff frequency in Hz
            high_cutoff: Upper cutoff frequency in Hz
            order: Filter order
        """
        super().__init__()
        self.low_cutoff = low_cutoff
        self.high_cutoff = high_cutoff
        self.order = order
        self.sos = None

    def initialize(self, fs: float) -> None:
        self.fs = fs
        nyquist = 0.5 * fs
        low = self.low_cutoff / nyquist
        high = self.high_cutoff / nyquist
        self.sos = signal.butter(self.order, [low, high], btype="band", output="sos")
        # Bandpass needs more samples due to combined filters
        self.min_signal_length = max(64, 8 * self.order)
        self.is_initialized = True

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if not self.is_initialized:
            raise RuntimeError("Filter not initialized. Call initialize(fs) first.")

        # Return original signal if too short for filtering
        if not self._check_signal_length(x):
            return x.copy()

        try:
            # Try to use second-order sections for stability
            filtered = signal.sosfiltfilt(self.sos, x, padtype="constant")
            return filtered
        except ValueError as e:
            # If that fails, try to fall back to standard filter with minimal padding
            try:
                # Fallback to a regular filtfilt with minimal padding
                b, a = signal.sos2tf(self.sos)
                padlen = min(
                    3 * self.order, len(x) - 1
                )  # Ensure padlen < signal length
                filtered = signal.filtfilt(b, a, x, padlen=padlen)
                return filtered
            except Exception:
                # If all else fails, return the original signal
                return x.copy()


@public_api
class NotchFilter(BaseFilter):
    """Notch filter for removing power line interference"""

    def __init__(
        self,
        notch_freq: float,
        quality_factor: float = 30.0,
    ):
        """
        Initialize notch filter

        Args:
            notch_freq: Notch frequency in Hz (e.g. 50 or 60 for power line)
            quality_factor: Quality factor controlling notch width
        """
        super().__init__()
        self.notch_freq = notch_freq
        self.quality_factor = quality_factor
        self.sos = None

    def initialize(self, fs: float) -> None:
        self.fs = fs
        nyquist = 0.5 * fs
        w0 = self.notch_freq / nyquist
        b, a = signal.iirnotch(w0, self.quality_factor)
        self.sos = signal.tf2sos(b, a)
        # Notch filters typically need less padding
        self.min_signal_length = 32
        self.is_initialized = True

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if not self.is_initialized:
            raise RuntimeError("Filter not initialized. Call initialize(fs) first.")

        # Return original signal if too short for filtering
        if not self._check_signal_length(x):
            return x.copy()

        try:
            # Try to use second-order sections for stability
            filtered = signal.sosfiltfilt(self.sos, x, padtype="constant")
            return filtered
        except ValueError as e:
            # If that fails, try to fall back to standard filter with minimal padding
            try:
                # Fallback to a regular filtfilt with minimal padding
                b, a = signal.sos2tf(self.sos)
                padlen = min(3, len(x) - 1)  # Notch filters need less padding
                filtered = signal.filtfilt(b, a, x, padlen=padlen)
                return filtered
            except Exception:
                # If all else fails, return the original signal
                return x.copy()


# Factory functions (same API as before)
@public_api
def create_lowpass_filter(cutoff: float, order: int = 4) -> LowpassFilter:
    """
    Create a lowpass filter

    Args:
        cutoff: Cutoff frequency in Hz
        order: Filter order

    Returns:
        A configured LowpassFilter object
    """
    return LowpassFilter(cutoff=cutoff, order=order)


@public_api
def create_highpass_filter(cutoff: float, order: int = 4) -> HighpassFilter:
    """
    Create a highpass filter

    Args:
        cutoff: Cutoff frequency in Hz
        order: Filter order

    Returns:
        A configured HighpassFilter object
    """
    return HighpassFilter(cutoff=cutoff, order=order)


@public_api
def create_bandpass_filter(
    low_cutoff: float, high_cutoff: float, order: int = 4
) -> BandpassFilter:
    """
    Create a bandpass filter

    Args:
        low_cutoff: Lower cutoff frequency in Hz
        high_cutoff: Upper cutoff frequency in Hz
        order: Filter order

    Returns:
        A configured BandpassFilter object
    """
    return BandpassFilter(
        low_cutoff=low_cutoff,
        high_cutoff=high_cutoff,
        order=order,
    )


@public_api
def create_notch_filter(notch_freq: float, quality_factor: float = 30.0) -> NotchFilter:
    """
    Create a notch filter

    Args:
        notch_freq: Notch frequency in Hz (e.g. 50 or 60 for power line)
        quality_factor: Quality factor controlling notch width

    Returns:
        A configured NotchFilter object
    """
    return NotchFilter(notch_freq=notch_freq, quality_factor=quality_factor)


@public_api
class EMGFilter:
    """Container for a sequence of filters to be applied to EMG data"""

    def __init__(self, filters: Optional[List[BaseFilter]] = None):
        self.filters = filters or []
        self.is_initialized = False

    def initialize(self, fs: float) -> None:
        """Initialize all filters with sampling frequency"""
        for filter_obj in self.filters:
            filter_obj.initialize(fs)
        self.is_initialized = True

    def add_filter(self, filter_obj: BaseFilter) -> None:
        """Add a filter to the sequence"""
        self.filters.append(filter_obj)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply all filters in sequence"""
        if not self.filters:
            return tensor

        # Convert to numpy for filtering
        data_np = tensor.numpy()
        filtered_data = np.zeros_like(data_np)

        # Apply filtering to each channel
        for i in range(data_np.shape[0]):
            # Start with a copy of the input data
            filtered = data_np[i].copy()

            # Apply each filter in sequence
            for filter_obj in self.filters:
                filtered = filter_obj(filtered)

            filtered_data[i] = filtered

        # Convert back to torch tensor
        return torch.tensor(filtered_data, dtype=tensor.dtype)


@public_api
def create_emg_filter(filters: Optional[List[BaseFilter]] = None) -> EMGFilter:
    """
    Create an EMG filter with a sequence of filters

    Args:
        filters: List of filter objects to apply in sequence

    Returns:
        A configured EMGFilter object
    """
    return EMGFilter(filters=filters)
