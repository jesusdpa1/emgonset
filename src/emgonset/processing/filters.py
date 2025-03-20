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

    @abstractmethod
    def initialize(self, fs: float) -> None:
        """Initialize filter with sampling frequency"""
        pass

    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply filter to signal"""
        pass


@public_api
class LowpassFilter(BaseFilter):
    """Lowpass filter for EMG signals"""

    def __init__(self, cutoff: float, order: int = 4):
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
        self.is_initialized = True

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if not self.is_initialized:
            raise RuntimeError("Filter not initialized. Call initialize(fs) first.")
        return signal.sosfiltfilt(self.sos, x)


@public_api
class HighpassFilter(BaseFilter):
    """Highpass filter for EMG signals"""

    def __init__(self, cutoff: float, order: int = 4):
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
        self.is_initialized = True

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if not self.is_initialized:
            raise RuntimeError("Filter not initialized. Call initialize(fs) first.")
        return signal.sosfiltfilt(self.sos, x)


@public_api
class BandpassFilter(BaseFilter):
    """Bandpass filter for EMG signals"""

    def __init__(self, low_cutoff: float, high_cutoff: float, order: int = 4):
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
        self.is_initialized = True

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if not self.is_initialized:
            raise RuntimeError("Filter not initialized. Call initialize(fs) first.")
        return signal.sosfiltfilt(self.sos, x)


@public_api
class NotchFilter(BaseFilter):
    """Notch filter for removing power line interference"""

    def __init__(self, notch_freq: float, quality_factor: float = 30.0):
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
        self.is_initialized = True

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if not self.is_initialized:
            raise RuntimeError("Filter not initialized. Call initialize(fs) first.")
        return signal.sosfiltfilt(self.sos, x)


# Factory functions that return filter objects
@public_api
def create_lowpass_filter(cutoff: float, order: int = 4) -> LowpassFilter:
    return LowpassFilter(cutoff=cutoff, order=order)


@public_api
def create_highpass_filter(cutoff: float, order: int = 4) -> HighpassFilter:
    return HighpassFilter(cutoff=cutoff, order=order)


@public_api
def create_bandpass_filter(
    low_cutoff: float, high_cutoff: float, order: int = 4
) -> BandpassFilter:
    return BandpassFilter(low_cutoff=low_cutoff, high_cutoff=high_cutoff, order=order)


def create_notch_filter(notch_freq: float, quality_factor: float = 30.0) -> NotchFilter:
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
    """Create an EMG filter with a sequence of filters"""
    return EMGFilter(filters=filters)
