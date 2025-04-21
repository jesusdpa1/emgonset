from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Union

import numpy as np
import torch
from scipy import signal


class BaseFilter(ABC):
    """Base class for all filters that can be initialized later"""

    def __init__(self):
        self.fs: Optional[float] = None
        self.is_initialized: bool = False
        self.min_signal_length: int = 32  # Minimum signal length to apply filtering
        self.sos: Optional[np.ndarray] = None

    @abstractmethod
    def initialize(self, fs: float) -> None:
        """Initialize filter with sampling frequency"""
        pass

    def _filter_single_channel(self, x: np.ndarray) -> np.ndarray:
        """
        Apply filter to a single channel

        Args:
            x: Input data with shape (samples,)

        Returns:
            Filtered data with same shape
        """
        if not self.is_initialized:
            raise RuntimeError("Filter not initialized. Call initialize(fs) first.")

        # Return original signal if too short for filtering
        if len(x) < self.min_signal_length:
            return x.copy()

        try:
            # Try to use second-order sections for stability
            filtered = signal.sosfiltfilt(self.sos, x, padtype="constant")
            return filtered
        except ValueError:
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

    def __call__(
        self, x: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Apply filter to signal

        Args:
            x: Input data with shape (samples,) or (samples, channels)

        Returns:
            Filtered data with same shape
        """
        # Convert to numpy if torch tensor
        was_tensor = False
        if isinstance(x, torch.Tensor):
            was_tensor = True
            x = x.numpy()

        # Ensure 2D input
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        # Apply filtering to each channel
        filtered_data = np.zeros_like(x)

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor() as executor:
            # Submit filtering tasks for each channel
            future_to_channel = {
                executor.submit(self._filter_single_channel, x[:, i]): i
                for i in range(x.shape[1])
            }

            # Collect results
            for future in as_completed(future_to_channel):
                channel = future_to_channel[future]
                try:
                    filtered_data[:, channel] = future.result()
                except Exception:
                    # Fallback to original channel data if filtering fails
                    filtered_data[:, channel] = x[:, channel]

        # Convert back to original type
        return (
            torch.tensor(filtered_data, dtype=torch.float32)
            if was_tensor
            else filtered_data
        )


class LowpassFilter(BaseFilter):
    """Lowpass filter for signals"""

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

    def initialize(self, fs: float) -> None:
        """
        Initialize filter with sampling frequency

        Args:
            fs: Sampling frequency in Hz
        """
        self.fs = fs
        nyquist = 0.5 * fs
        normalized_cutoff = self.cutoff / nyquist
        self.sos = signal.butter(
            self.order, normalized_cutoff, btype="low", output="sos"
        )
        # Set minimum signal length based on filter order
        self.min_signal_length = max(32, 4 * self.order)
        self.is_initialized = True


class HighpassFilter(BaseFilter):
    """Highpass filter for signals"""

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

    def initialize(self, fs: float) -> None:
        """
        Initialize filter with sampling frequency

        Args:
            fs: Sampling frequency in Hz
        """
        self.fs = fs
        nyquist = 0.5 * fs
        normalized_cutoff = self.cutoff / nyquist
        self.sos = signal.butter(
            self.order, normalized_cutoff, btype="high", output="sos"
        )
        # Set minimum signal length based on filter order
        self.min_signal_length = max(32, 4 * self.order)
        self.is_initialized = True


class BandpassFilter(BaseFilter):
    """Bandpass filter for signals"""

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

    def initialize(self, fs: float) -> None:
        """
        Initialize filter with sampling frequency

        Args:
            fs: Sampling frequency in Hz
        """
        self.fs = fs
        nyquist = 0.5 * fs
        low = self.low_cutoff / nyquist
        high = self.high_cutoff / nyquist
        self.sos = signal.butter(self.order, [low, high], btype="band", output="sos")
        # Bandpass needs more samples due to combined filters
        self.min_signal_length = max(64, 8 * self.order)
        self.is_initialized = True


class NotchFilter(BaseFilter):
    """Notch filter for removing interference"""

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

    def initialize(self, fs: float) -> None:
        """
        Initialize filter with sampling frequency

        Args:
            fs: Sampling frequency in Hz
        """
        self.fs = fs
        nyquist = 0.5 * fs
        w0 = self.notch_freq / nyquist
        b, a = signal.iirnotch(w0, self.quality_factor)
        self.sos = signal.tf2sos(b, a)
        # Notch filters typically need less padding
        self.min_signal_length = 32
        self.is_initialized = True


class CascadeFilter:
    """Container for a sequence of filters to be applied in cascade"""

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

    def __call__(
        self, data: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Apply all filters in sequence

        Args:
            data: Input data with shape (samples, channels)

        Returns:
            Filtered data with same shape
        """
        if not self.filters:
            return data

        # Convert to numpy if tensor
        was_tensor = False
        if isinstance(data, torch.Tensor):
            was_tensor = True
            data_np = data.numpy()
        else:
            data_np = data

        # Ensure 2D input
        if data_np.ndim == 1:
            data_np = data_np.reshape(-1, 1)

        # Apply filtering to each channel in sequence
        filtered_data = data_np.copy()
        for filter_obj in self.filters:
            filtered_data = filter_obj(filtered_data)

        # Convert back to original type
        return (
            torch.tensor(filtered_data, dtype=torch.float32)
            if was_tensor
            else filtered_data
        )


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


def create_cascade_filter(filters: Optional[List[BaseFilter]] = None) -> CascadeFilter:
    """
    Create a cascade filter with a sequence of filters

    Args:
        filters: List of filter objects to apply in sequence

    Returns:
        A configured CascadeFilter object
    """
    return CascadeFilter(filters=filters)
