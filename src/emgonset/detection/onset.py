"""
/src/emgonset/detection/onset.py
Onset and offset detection algorithms for EMG signals using thresholding methods
"""

from abc import ABC, abstractmethod
from typing import List, NamedTuple, Optional, Tuple, Union

import numpy as np
import torch

from ..utils.internals import public_api


class OnsetOffsetPoint(NamedTuple):
    """Container for onset/offset time points"""

    sample_index: int  # Sample index in the signal
    time_sec: float  # Time in seconds
    amplitude: float  # Signal amplitude at the detection point


class OnsetOffsetResult(NamedTuple):
    """Container for onset/offset detection results"""

    onsets: List[OnsetOffsetPoint]  # List of detected onset points
    offsets: List[OnsetOffsetPoint]  # List of detected offset points
    threshold_value: float  # The threshold value used for detection
    baseline_value: Optional[float]  # The calculated baseline value (if applicable)


class BaseOnsetDetector(ABC):
    """Base class for all onset/offset detection methods"""

    def __init__(
        self,
        min_duration_ms: float = 100.0,
    ):
        """
        Initialize onset detector

        Args:
            min_duration_ms: Minimum duration in milliseconds for a valid onset-offset pair
        """
        self.fs: Optional[float] = None
        self.is_initialized: bool = False
        self.min_duration_ms = min_duration_ms
        self.min_samples = None  # Will be calculated when fs is known

    def initialize(self, fs: float) -> None:
        """Initialize detector with sampling frequency"""
        self.fs = fs
        self.min_samples = int(self.min_duration_ms * fs / 1000)
        self.is_initialized = True

    @abstractmethod
    def detect(self, signal: np.ndarray) -> OnsetOffsetResult:
        """
        Detect onset and offset points in the signal

        Args:
            signal: 1D numpy array containing the signal (typically an envelope)

        Returns:
            OnsetOffsetResult containing detected onsets, offsets, and threshold info
        """
        pass

    def __call__(self, signal: Union[np.ndarray, torch.Tensor]) -> OnsetOffsetResult:
        """
        Apply onset/offset detection to signal

        Args:
            signal: 1D numpy array or torch tensor containing the signal

        Returns:
            OnsetOffsetResult containing detected onsets, offsets, and threshold info
        """
        if not self.is_initialized:
            raise RuntimeError("Detector not initialized. Call initialize(fs) first.")

        # Convert to numpy if tensor
        if isinstance(signal, torch.Tensor):
            signal_np = signal.detach().cpu().numpy()
        else:
            signal_np = signal

        # Ensure 1D array
        if signal_np.ndim > 1:
            raise ValueError("Expected 1D signal for onset detection")

        return self.detect(signal_np)

    def validate_pair(self, onset_idx: int, offset_idx: int) -> bool:
        """
        Validate if an onset-offset pair meets the minimum duration requirements

        Args:
            onset_idx: Sample index of onset
            offset_idx: Sample index of offset

        Returns:
            True if pair is valid, False otherwise
        """
        if self.min_samples is None:
            raise RuntimeError("Detector not initialized. Call initialize(fs) first.")

        return (offset_idx - onset_idx) >= self.min_samples


@public_api
class FixedThresholdDetector(BaseOnsetDetector):
    """
    Onset/offset detector using a fixed threshold value

    This detector identifies onsets when the signal crosses above the threshold
    and offsets when it crosses below the threshold.
    """

    def __init__(
        self,
        threshold: float,
        min_duration_ms: float = 100.0,
        hysteresis: float = 0.0,
    ):
        """
        Initialize fixed threshold detector

        Args:
            threshold: Fixed threshold value
            min_duration_ms: Minimum duration in ms for a valid onset-offset pair
            hysteresis: Optional hysteresis value to prevent rapid on/off switching
        """
        super().__init__(min_duration_ms=min_duration_ms)
        self.threshold = threshold
        self.hysteresis = hysteresis

    def detect(self, signal: np.ndarray) -> OnsetOffsetResult:
        """
        Detect onset and offset points using fixed threshold

        Args:
            signal: 1D numpy array containing the signal

        Returns:
            OnsetOffsetResult containing detected onsets, offsets, and threshold info
        """
        if not self.is_initialized:
            raise RuntimeError("Detector not initialized. Call initialize(fs) first.")

        # Create time vector
        time = np.arange(len(signal)) / self.fs

        # Binary signal indicating where signal is above threshold
        if self.hysteresis > 0:
            # Use different thresholds for rising and falling edges
            onset_mask = signal >= self.threshold
            offset_mask = signal < (self.threshold - self.hysteresis)

            # Initialize activation state
            active = False
            binary_signal = np.zeros_like(signal, dtype=bool)

            # Apply hysteresis
            for i in range(len(signal)):
                if not active and onset_mask[i]:
                    active = True
                elif active and offset_mask[i]:
                    active = False
                binary_signal[i] = active
        else:
            # Simple thresholding without hysteresis
            binary_signal = signal >= self.threshold

        # Find transitions (0->1 for onsets, 1->0 for offsets)
        transitions = np.diff(binary_signal.astype(int))
        onset_indices = (
            np.where(transitions == 1)[0] + 1
        )  # +1 because diff reduces length by 1
        offset_indices = np.where(transitions == -1)[0] + 1

        # Ensure pairs match (if signal starts above threshold, add onset at beginning)
        if len(offset_indices) > 0 and (
            len(onset_indices) == 0 or offset_indices[0] < onset_indices[0]
        ):
            onset_indices = np.insert(onset_indices, 0, 0)

        # If signal ends above threshold, add offset at end
        if len(onset_indices) > len(offset_indices):
            offset_indices = np.append(offset_indices, len(signal) - 1)

        # Match onset-offset pairs, filtering by minimum duration
        onsets = []
        offsets = []

        for onset_idx, offset_idx in zip(onset_indices, offset_indices):
            if self.validate_pair(onset_idx, offset_idx):
                onsets.append(
                    OnsetOffsetPoint(
                        sample_index=onset_idx,
                        time_sec=time[onset_idx],
                        amplitude=signal[onset_idx],
                    )
                )
                offsets.append(
                    OnsetOffsetPoint(
                        sample_index=offset_idx,
                        time_sec=time[offset_idx],
                        amplitude=signal[offset_idx],
                    )
                )

        return OnsetOffsetResult(
            onsets=onsets,
            offsets=offsets,
            threshold_value=self.threshold,
            baseline_value=None,
        )


@public_api
class StandardDeviationThresholdDetector(BaseOnsetDetector):
    """
    Onset/offset detector using a threshold based on signal standard deviation

    This detector calculates a threshold as baseline + k * std_dev, where
    k is a multiplier and baseline is mean of the first n samples or a specified region.
    """

    def __init__(
        self,
        std_dev_factor: float = 3.0,
        baseline_window_ms: float = 500.0,
        min_duration_ms: float = 100.0,
        hysteresis_factor: float = 0.0,
        baseline_region: Optional[Tuple[int, int]] = None,
    ):
        """
        Initialize standard deviation threshold detector

        Args:
            std_dev_factor: Number of standard deviations above baseline for threshold
            baseline_window_ms: Duration in ms to use for baseline calculation
            min_duration_ms: Minimum duration in ms for a valid onset-offset pair
            hysteresis_factor: Hysteresis as a fraction of the threshold-baseline difference
            baseline_region: Optional explicit baseline region as (start, end) indices.
                            If provided, this overrides baseline_window_ms.
        """
        super().__init__(min_duration_ms=min_duration_ms)
        self.std_dev_factor = std_dev_factor
        self.baseline_window_ms = baseline_window_ms
        self.hysteresis_factor = hysteresis_factor
        self.baseline_region = baseline_region
        self.baseline_samples = None  # Will be calculated when fs is known

    def initialize(self, fs: float) -> None:
        """Initialize detector with sampling frequency"""
        super().initialize(fs)
        if self.baseline_region is None:
            self.baseline_samples = int(self.baseline_window_ms * fs / 1000)

    def detect(self, signal: np.ndarray) -> OnsetOffsetResult:
        """
        Detect onset and offset points using standard deviation threshold

        Args:
            signal: 1D numpy array containing the signal

        Returns:
            OnsetOffsetResult containing detected onsets, offsets, and threshold info
        """
        if not self.is_initialized:
            raise RuntimeError("Detector not initialized. Call initialize(fs) first.")

        # Determine baseline region
        if self.baseline_region is not None:
            start_idx, end_idx = self.baseline_region
        else:
            start_idx, end_idx = 0, min(self.baseline_samples, len(signal))

        # Calculate baseline parameters
        baseline_mean = np.mean(signal[start_idx:end_idx])
        baseline_std = np.std(signal[start_idx:end_idx])

        # Calculate threshold
        threshold = baseline_mean + self.std_dev_factor * baseline_std

        # Calculate hysteresis if needed
        hysteresis = (
            self.hysteresis_factor * (threshold - baseline_mean)
            if self.hysteresis_factor > 0
            else 0
        )

        # Create detector with calculated threshold
        detector = FixedThresholdDetector(
            threshold=threshold,
            min_duration_ms=self.min_duration_ms,
            hysteresis=hysteresis,
        )
        detector.initialize(self.fs)

        # Detect onsets and offsets
        fixed_detector_result = detector.detect(signal)

        # Return with additional baseline information
        return OnsetOffsetResult(
            onsets=fixed_detector_result.onsets,
            offsets=fixed_detector_result.offsets,
            threshold_value=threshold,
            baseline_value=baseline_mean,
        )


@public_api
class AdaptiveThresholdDetector(BaseOnsetDetector):
    """
    Onset/offset detector using an adaptive threshold based on local signal properties

    This detector computes a threshold that adapts to changing signal characteristics
    using a moving average of the signal with a separate background window.
    """

    def __init__(
        self,
        background_window_ms: float = 1000.0,
        detection_window_ms: float = 50.0,
        std_dev_factor: float = 3.0,
        min_duration_ms: float = 100.0,
        min_threshold: Optional[float] = None,
    ):
        """
        Initialize adaptive threshold detector

        Args:
            background_window_ms: Duration in ms for background level calculation
            detection_window_ms: Duration in ms for detection window
            std_dev_factor: Number of standard deviations above background for threshold
            min_duration_ms: Minimum duration in ms for a valid onset-offset pair
            min_threshold: Optional minimum threshold value
        """
        super().__init__(min_duration_ms=min_duration_ms)
        self.background_window_ms = background_window_ms
        self.detection_window_ms = detection_window_ms
        self.std_dev_factor = std_dev_factor
        self.min_threshold = min_threshold

        # Will be calculated when fs is known
        self.background_samples = None
        self.detection_samples = None

    def initialize(self, fs: float) -> None:
        """Initialize detector with sampling frequency"""
        super().initialize(fs)
        self.background_samples = int(self.background_window_ms * fs / 1000)
        self.detection_samples = int(self.detection_window_ms * fs / 1000)

    def detect(self, signal: np.ndarray) -> OnsetOffsetResult:
        """
        Detect onset and offset points using adaptive threshold

        Args:
            signal: 1D numpy array containing the signal

        Returns:
            OnsetOffsetResult containing detected onsets, offsets, and threshold info
        """
        if not self.is_initialized:
            raise RuntimeError("Detector not initialized. Call initialize(fs) first.")

        # Create time vector
        time = np.arange(len(signal)) / self.fs

        # Calculate moving mean and std of background window
        bg_mean = np.zeros_like(signal)
        bg_std = np.zeros_like(signal)

        for i in range(len(signal)):
            # Determine background window (use past samples)
            start_idx = max(0, i - self.background_samples)
            # Use shorter window at beginning of signal
            bg_window = signal[start_idx : i + 1]

            if len(bg_window) > 0:
                bg_mean[i] = np.mean(bg_window)
                bg_std[i] = np.std(bg_window)
            else:
                # Fallback (shouldn't happen with bounds check)
                bg_mean[i] = signal[i]
                bg_std[i] = 0

        # Calculate threshold at each point
        threshold = bg_mean + self.std_dev_factor * bg_std

        # Apply minimum threshold if specified
        if self.min_threshold is not None:
            threshold = np.maximum(threshold, self.min_threshold)

        # Binary signal indicating where signal is above threshold
        binary_signal = np.zeros_like(signal, dtype=bool)

        # Use detection window to find sustained activations
        for i in range(len(signal) - self.detection_samples + 1):
            if (
                np.mean(
                    signal[i : i + self.detection_samples]
                    > threshold[i : i + self.detection_samples]
                )
                > 0.5
            ):
                binary_signal[i : i + self.detection_samples] = True

        # Find transitions (0->1 for onsets, 1->0 for offsets)
        transitions = np.diff(binary_signal.astype(int))
        onset_indices = (
            np.where(transitions == 1)[0] + 1
        )  # +1 because diff reduces length by 1
        offset_indices = np.where(transitions == -1)[0] + 1

        # Ensure pairs match (if signal starts above threshold, add onset at beginning)
        if len(offset_indices) > 0 and (
            len(onset_indices) == 0 or offset_indices[0] < onset_indices[0]
        ):
            onset_indices = np.insert(onset_indices, 0, 0)

        # If signal ends above threshold, add offset at end
        if len(onset_indices) > len(offset_indices):
            offset_indices = np.append(offset_indices, len(signal) - 1)

        # Match onset-offset pairs, filtering by minimum duration
        onsets = []
        offsets = []

        for onset_idx, offset_idx in zip(onset_indices, offset_indices):
            if self.validate_pair(onset_idx, offset_idx):
                onsets.append(
                    OnsetOffsetPoint(
                        sample_index=onset_idx,
                        time_sec=time[onset_idx],
                        amplitude=signal[onset_idx],
                    )
                )
                offsets.append(
                    OnsetOffsetPoint(
                        sample_index=offset_idx,
                        time_sec=time[offset_idx],
                        amplitude=signal[offset_idx],
                    )
                )

        return OnsetOffsetResult(
            onsets=onsets,
            offsets=offsets,
            threshold_value=np.mean(threshold),  # Return average threshold as reference
            baseline_value=np.mean(bg_mean),  # Return average baseline
        )


@public_api
def create_fixed_threshold_detector(
    threshold: float,
    min_duration_ms: float = 100.0,
    hysteresis: float = 0.0,
) -> FixedThresholdDetector:
    """
    Create a fixed threshold onset/offset detector

    Args:
        threshold: Fixed threshold value
        min_duration_ms: Minimum duration in ms for a valid onset-offset pair
        hysteresis: Optional hysteresis value

    Returns:
        Configured FixedThresholdDetector
    """
    return FixedThresholdDetector(
        threshold=threshold,
        min_duration_ms=min_duration_ms,
        hysteresis=hysteresis,
    )


@public_api
def create_std_threshold_detector(
    std_dev_factor: float = 3.0,
    baseline_window_ms: float = 500.0,
    min_duration_ms: float = 100.0,
    hysteresis_factor: float = 0.0,
) -> StandardDeviationThresholdDetector:
    """
    Create a standard deviation based threshold detector

    Args:
        std_dev_factor: Number of standard deviations above baseline
        baseline_window_ms: Duration in ms for baseline calculation
        min_duration_ms: Minimum duration in ms for a valid onset-offset pair
        hysteresis_factor: Hysteresis as a fraction of threshold

    Returns:
        Configured StandardDeviationThresholdDetector
    """
    return StandardDeviationThresholdDetector(
        std_dev_factor=std_dev_factor,
        baseline_window_ms=baseline_window_ms,
        min_duration_ms=min_duration_ms,
        hysteresis_factor=hysteresis_factor,
    )


@public_api
def create_adaptive_threshold_detector(
    background_window_ms: float = 1000.0,
    detection_window_ms: float = 50.0,
    std_dev_factor: float = 3.0,
    min_duration_ms: float = 100.0,
    min_threshold: Optional[float] = None,
) -> AdaptiveThresholdDetector:
    """
    Create an adaptive threshold detector

    Args:
        background_window_ms: Duration in ms for background level calculation
        detection_window_ms: Duration in ms for detection window
        std_dev_factor: Number of standard deviations above background
        min_duration_ms: Minimum duration in ms for a valid onset-offset pair
        min_threshold: Optional minimum threshold value

    Returns:
        Configured AdaptiveThresholdDetector
    """
    return AdaptiveThresholdDetector(
        background_window_ms=background_window_ms,
        detection_window_ms=detection_window_ms,
        std_dev_factor=std_dev_factor,
        min_duration_ms=min_duration_ms,
        min_threshold=min_threshold,
    )
