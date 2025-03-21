from abc import ABC, abstractmethod
from typing import Dict, Optional

import numpy as np
import torch

from ..utils.internals import public_api
from ..utils.padding import calculate_pad_length, mirror_pad_numpy, unpad_numpy
from .filters import (
    create_lowpass_filter,
)
from .rectifiers import (
    create_abs_rectifier,
    create_hilbert_rectifier,
    create_square_rectifier,
)
from .tkeo import (
    create_mtkeo,
    create_tkeo,
    create_tkeo2,
)


class BaseEnvelope(ABC):
    """Base class for all EMG envelope detection methods"""

    def __init__(self, pad_time_ms: float = 100.0):
        self.fs: Optional[float] = None
        self.is_initialized: bool = False
        self.pad_time_ms = pad_time_ms
        self.pad_length = None

    def initialize(self, fs: float) -> None:
        """Initialize envelope detector with sampling frequency"""
        self.fs = fs
        self.pad_length = calculate_pad_length(fs, self.pad_time_ms)
        self.is_initialized = True

    @abstractmethod
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply envelope detection to signal"""
        pass


@public_api
class LowpassEnvelope(BaseEnvelope):
    """
    Envelope detector using rectification and lowpass filtering.

    This applies a rectification method followed by a lowpass filter
    to create a smoothed envelope of the signal.
    """

    def __init__(
        self,
        rectification: str = "abs",
        cutoff_freq: float = 10.0,
        filter_order: int = 4,
        pad_time_ms: float = 100.0,
    ):
        """
        Initialize lowpass envelope detector

        Args:
            rectification: Rectification method ('abs', 'square', 'hilbert')
            cutoff_freq: Cutoff frequency for lowpass filter in Hz
            filter_order: Order of the Butterworth filter
            pad_time_ms: Padding time in milliseconds to reduce edge effects
        """
        super().__init__(pad_time_ms=pad_time_ms)
        self.rectification_type = rectification
        self.cutoff_freq = cutoff_freq
        self.filter_order = filter_order

        # Initialize components
        self._setup_components()

    def _setup_components(self):
        """Set up rectifier and filter components"""
        # Create rectifier based on type
        if self.rectification_type == "abs":
            self.rectifier = create_abs_rectifier()
        elif self.rectification_type == "square":
            self.rectifier = create_square_rectifier()
        elif self.rectification_type == "hilbert":
            self.rectifier = create_hilbert_rectifier()
        else:
            raise ValueError(f"Unknown rectification type: {self.rectification_type}")

        # Create lowpass filter
        self.filter = create_lowpass_filter(
            cutoff=self.cutoff_freq,
            order=self.filter_order,
            pad_time_ms=self.pad_time_ms,
        )

    def initialize(self, fs: float) -> None:
        """Initialize envelope detector with sampling frequency"""
        super().initialize(fs)

        # Initialize filter with sampling frequency
        self.filter.initialize(fs)

        # Initialize rectifier if it has the method
        if hasattr(self.rectifier, "initialize"):
            self.rectifier.initialize(fs)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply lowpass envelope detection to signal

        Args:
            tensor: PyTorch tensor of shape [channels, samples]

        Returns:
            Envelope tensor of shape [channels, samples]
        """
        if not self.is_initialized:
            raise RuntimeError(
                "Envelope detector not initialized. Call initialize(fs) first."
            )

        result = torch.zeros_like(tensor)

        # Process each channel
        for ch in range(tensor.shape[0]):
            # Get signal for this channel
            signal_np = tensor[ch].detach().cpu().numpy()

            # Apply mirror padding before rectification to avoid edge effects in rectification
            padded_signal = mirror_pad_numpy(signal_np, self.pad_length)

            # Apply rectification
            rectified = self.rectifier(padded_signal)

            # Apply lowpass filter (filter will handle its own padding internally)
            envelope = self.filter(rectified)

            # Remove padding
            envelope = unpad_numpy(envelope, self.pad_length)

            # Convert back to tensor - use copy to ensure positive strides
            result[ch] = torch.tensor(np.copy(envelope), dtype=tensor.dtype)

        return result


@public_api
class TKEOEnvelope(BaseEnvelope):
    """
    Envelope detector using TKEO and optional rectification and filtering.

    This applies an optional rectification, followed by TKEO calculation,
    and an optional lowpass filter to create a smoothed envelope.
    """

    def __init__(
        self,
        tkeo_type: str = "classic",
        rectification: Optional[str] = None,
        lowpass_cutoff: Optional[float] = None,
        filter_order: int = 4,
        tkeo_params: Optional[Dict] = None,
        pad_time_ms: float = 100.0,
    ):
        """
        Initialize TKEO envelope detector

        Args:
            tkeo_type: TKEO type ('classic', 'tkeo2', 'mtkeo')
            rectification: Optional rectification method ('abs', 'square', 'hilbert', None)
            lowpass_cutoff: Optional cutoff frequency for lowpass filter in Hz (None for no filtering)
            filter_order: Order of the Butterworth filter if lowpass is used
            tkeo_params: Additional parameters for TKEO (e.g., weights for MTKEO)
            pad_time_ms: Padding time in milliseconds to reduce edge effects
        """
        super().__init__(pad_time_ms=pad_time_ms)
        self.tkeo_type = tkeo_type
        self.rectification_type = rectification
        self.lowpass_cutoff = lowpass_cutoff
        self.filter_order = filter_order
        self.tkeo_params = tkeo_params or {}

        # Initialize components
        self._setup_components()

    def _setup_components(self):
        """Set up rectifier, TKEO and filter components"""
        # Create rectifier if specified
        if self.rectification_type is not None:
            if self.rectification_type == "abs":
                self.rectifier = create_abs_rectifier()
            elif self.rectification_type == "square":
                self.rectifier = create_square_rectifier()
            elif self.rectification_type == "hilbert":
                self.rectifier = create_hilbert_rectifier()
            else:
                raise ValueError(
                    f"Unknown rectification type: {self.rectification_type}"
                )
        else:
            self.rectifier = None

        # Create TKEO based on type
        if self.tkeo_type == "classic":
            self.tkeo = create_tkeo()
        elif self.tkeo_type == "tkeo2":
            self.tkeo = create_tkeo2()
        elif self.tkeo_type == "mtkeo":
            # Extract MTKEO parameters if provided
            k1 = self.tkeo_params.get("k1", 1.0)
            k2 = self.tkeo_params.get("k2", 1.0)
            k3 = self.tkeo_params.get("k3", 1.0)
            self.tkeo = create_mtkeo(k1=k1, k2=k2, k3=k3)
        else:
            raise ValueError(f"Unknown TKEO type: {self.tkeo_type}")

        # Create lowpass filter if cutoff is specified
        if self.lowpass_cutoff is not None:
            self.filter = create_lowpass_filter(
                cutoff=self.lowpass_cutoff,
                order=self.filter_order,
                pad_time_ms=self.pad_time_ms,
            )
        else:
            self.filter = None

    def initialize(self, fs: float) -> None:
        """Initialize envelope detector with sampling frequency"""
        super().initialize(fs)

        # Initialize TKEO
        self.tkeo.initialize(fs)

        # Initialize rectifier if it exists and has the method
        if self.rectifier is not None and hasattr(self.rectifier, "initialize"):
            self.rectifier.initialize(fs)

        # Initialize filter if it exists
        if self.filter is not None:
            self.filter.initialize(fs)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply TKEO envelope detection to signal

        Args:
            tensor: PyTorch tensor of shape [channels, samples]

        Returns:
            Envelope tensor of shape [channels, samples]
        """
        if not self.is_initialized:
            raise RuntimeError(
                "Envelope detector not initialized. Call initialize(fs) first."
            )

        # Process each channel
        result_channels = []

        for ch in range(tensor.shape[0]):
            # Extract signal for this channel
            signal = tensor[ch]

            # Convert to numpy if needed for rectification with padding
            if self.rectifier is not None:
                signal_np = signal.detach().cpu().numpy()

                # Apply mirror padding
                padded_signal = mirror_pad_numpy(signal_np, self.pad_length)

                # Apply rectification
                rectified = self.rectifier(padded_signal)

                # Convert back to tensor
                prepared_signal = torch.tensor(np.copy(rectified), dtype=tensor.dtype)
            else:
                # Use original signal with padding applied when converting to TKEO
                prepared_signal = signal

            # Apply TKEO (TKEO handles tensors)
            # First, pad the signal if not already padded
            if self.rectifier is None:
                from ..utils.padding import mirror_pad_torch

                prepared_signal = mirror_pad_torch(prepared_signal, self.pad_length)

            tkeo_result = self.tkeo(prepared_signal.unsqueeze(0)).squeeze(0)

            # Apply lowpass filter if specified
            if self.filter is not None:
                # Convert to numpy for filtering
                tkeo_np = tkeo_result.detach().cpu().numpy()

                # Filter will handle its own padding
                filtered = self.filter(tkeo_np)

                # Remove padding applied before TKEO
                filtered = unpad_numpy(filtered, self.pad_length)

                # Convert back to tensor
                channel_result = torch.tensor(np.copy(filtered), dtype=tensor.dtype)
            else:
                # Remove padding applied before TKEO
                from ..utils.padding import unpad_torch

                channel_result = unpad_torch(tkeo_result, self.pad_length)

            result_channels.append(channel_result)

        # Stack channels
        return torch.stack(result_channels)


# Factory functions


@public_api
def create_lowpass_envelope(
    rectification: str = "abs",
    cutoff_freq: float = 10.0,
    filter_order: int = 4,
    pad_time_ms: float = 100.0,
) -> LowpassEnvelope:
    """
    Create a lowpass envelope detector

    Args:
        rectification: Rectification method ('abs', 'square', 'hilbert')
        cutoff_freq: Cutoff frequency for lowpass filter in Hz
        filter_order: Order of the Butterworth filter
        pad_time_ms: Padding time in milliseconds to reduce edge effects

    Returns:
        A configured LowpassEnvelope object
    """
    return LowpassEnvelope(
        rectification=rectification,
        cutoff_freq=cutoff_freq,
        filter_order=filter_order,
        pad_time_ms=pad_time_ms,
    )


@public_api
def create_tkeo_envelope(
    tkeo_type: str = "classic",
    rectification: Optional[str] = None,
    lowpass_cutoff: Optional[float] = None,
    filter_order: int = 4,
    tkeo_params: Optional[Dict] = None,
    pad_time_ms: float = 100.0,
) -> TKEOEnvelope:
    """
    Create a TKEO envelope detector

    Args:
        tkeo_type: TKEO type ('classic', 'tkeo2', 'mtkeo')
        rectification: Optional rectification method ('abs', 'square', 'hilbert', None)
        lowpass_cutoff: Optional cutoff frequency for lowpass filter in Hz (None for no filtering)
        filter_order: Order of the Butterworth filter if lowpass is used
        tkeo_params: Additional parameters for TKEO (e.g., weights for MTKEO)
        pad_time_ms: Padding time in milliseconds to reduce edge effects

    Returns:
        A configured TKEOEnvelope object
    """
    return TKEOEnvelope(
        tkeo_type=tkeo_type,
        rectification=rectification,
        lowpass_cutoff=lowpass_cutoff,
        filter_order=filter_order,
        tkeo_params=tkeo_params,
        pad_time_ms=pad_time_ms,
    )


@public_api
def create_envelope_pipeline(transforms: list):
    """
    Create a pipeline of envelope detection transforms using EMGTransformCompose

    This is a convenience function to combine multiple envelope transforms
    or other processing steps into a single pipeline.

    Args:
        transforms: List of transform objects to be applied sequentially

    Returns:
        An EMGTransformCompose object that applies all transforms in sequence
    """
    from ..processing.transforms import EMGTransformCompose

    return EMGTransformCompose(transforms)
