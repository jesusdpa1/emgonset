"""
/src/emgonset/visualization/detection_plots.py
Visualization tools for EMG onset/offset detection results
"""

from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from ..detection.onset import OnsetOffsetResult
from ..utils.internals import public_api


@public_api
def plot_onset_detection(
    signal: Union[np.ndarray, torch.Tensor],
    detection_result: OnsetOffsetResult,
    fs: float,
    raw_signal: Optional[Union[np.ndarray, torch.Tensor]] = None,
    channel_idx: int = 0,
    title: str = "EMG Onset/Offset Detection",
    figsize: Tuple[int, int] = (12, 8),
    time_range: Optional[Tuple[float, float]] = None,
    dark_mode: bool = True,
    show_threshold: bool = True,
    show_baseline: bool = True,
    save_path: Optional[str] = None,
    dpi: int = 300,
    show: bool = True,
):
    """
    Plot onset and offset detection results on signal (typically envelope)

    Args:
        signal: Processed signal (typically envelope) used for detection
        detection_result: Detection results from onset detector
        fs: Sampling frequency in Hz
        raw_signal: Optional raw signal to show alongside processed signal
        channel_idx: Channel index if signals are multi-channel
        title: Plot title
        figsize: Figure size as (width, height) in inches
        time_range: Optional tuple of (start_time, end_time) in seconds to zoom
        dark_mode: If True, use seaborn dark theme
        show_threshold: Whether to show threshold line
        show_baseline: Whether to show baseline line if available
        save_path: Optional path to save the figure
        dpi: Resolution for saved figure
        show: If True, call plt.show() to display the plot

    Returns:
        The matplotlib figure object
    """
    # Convert to numpy if tensors
    if isinstance(signal, torch.Tensor):
        signal_np = signal.detach().cpu().numpy()
        # Handle multi-channel signal
        if signal_np.ndim > 1:
            signal_np = signal_np[channel_idx]
    else:
        signal_np = signal
        if signal_np.ndim > 1:
            signal_np = signal_np[channel_idx]

    if raw_signal is not None:
        if isinstance(raw_signal, torch.Tensor):
            raw_np = raw_signal.detach().cpu().numpy()
            if raw_np.ndim > 1:
                raw_np = raw_np[channel_idx]
        else:
            raw_np = raw_signal
            if raw_np.ndim > 1:
                raw_np = raw_np[channel_idx]
    else:
        raw_np = None

    # Create time vector
    time = np.arange(len(signal_np)) / fs

    # Apply time range if specified
    if time_range is not None:
        start_idx = max(0, int(time_range[0] * fs))
        end_idx = min(len(signal_np), int(time_range[1] * fs))
        time_slice = slice(start_idx, end_idx)
        time = time[time_slice]
        signal_np = signal_np[time_slice]
        if raw_np is not None:
            # Handle potentially different lengths
            raw_end_idx = min(len(raw_np), int(time_range[1] * fs))
            raw_slice = slice(start_idx, raw_end_idx)
            raw_np = raw_np[raw_slice]
    else:
        time_slice = slice(None)

    # Set theme
    if dark_mode:
        sns.set_theme(style="darkgrid")
    else:
        sns.set_theme(style="whitegrid")

    # Create figure
    fig = plt.figure(figsize=figsize)

    # Determine number of subplots
    n_plots = 2 if raw_np is not None else 1

    # Plot raw signal if provided
    if raw_np is not None:
        ax1 = plt.subplot(n_plots, 1, 1)
        ax1.plot(time, raw_np, color="gray", alpha=0.8, label="Raw Signal")

        # Mark onsets and offsets on raw signal
        for onset in detection_result.onsets:
            if time_range is None or (
                onset.time_sec >= time_range[0] and onset.time_sec <= time_range[1]
            ):
                ax1.axvline(onset.time_sec, color="g", linestyle="--", alpha=0.7)

        for offset in detection_result.offsets:
            if time_range is None or (
                offset.time_sec >= time_range[0] and offset.time_sec <= time_range[1]
            ):
                ax1.axvline(offset.time_sec, color="r", linestyle="--", alpha=0.7)

        ax1.set_title("Raw EMG Signal")
        ax1.set_ylabel("Amplitude")
        ax1.set_xticklabels([])
        ax1.legend(loc="upper right")

    # Plot processed signal with detections
    ax2 = plt.subplot(n_plots, 1, n_plots)
    ax2.plot(time, signal_np, color="blue", label="Processed Signal")

    # Show threshold line if requested
    if show_threshold and detection_result.threshold_value is not None:
        ax2.axhline(
            detection_result.threshold_value,
            color="orange",
            linestyle="-",
            alpha=0.7,
            label=f"Threshold: {detection_result.threshold_value:.3f}",
        )

    # Show baseline if available and requested
    if show_baseline and detection_result.baseline_value is not None:
        ax2.axhline(
            detection_result.baseline_value,
            color="gray",
            linestyle="-.",
            alpha=0.5,
            label=f"Baseline: {detection_result.baseline_value:.3f}",
        )

    # Mark onsets and offsets
    onset_x, onset_y = [], []
    offset_x, offset_y = [], []

    for onset in detection_result.onsets:
        if time_range is None or (
            onset.time_sec >= time_range[0] and onset.time_sec <= time_range[1]
        ):
            ax2.axvline(onset.time_sec, color="g", linestyle="--", alpha=0.7)
            onset_x.append(onset.time_sec)
            onset_y.append(onset.amplitude)

    for offset in detection_result.offsets:
        if time_range is None or (
            offset.time_sec >= time_range[0] and offset.time_sec <= time_range[1]
        ):
            ax2.axvline(offset.time_sec, color="r", linestyle="--", alpha=0.7)
            offset_x.append(offset.time_sec)
            offset_y.append(offset.amplitude)

    # Plot onset and offset points
    ax2.scatter(onset_x, onset_y, color="green", s=50, label="Onset", zorder=5)
    ax2.scatter(offset_x, offset_y, color="red", s=50, label="Offset", zorder=5)

    # Shade the active regions
    for onset, offset in zip(detection_result.onsets, detection_result.offsets):
        if time_range is None or (
            onset.time_sec <= time_range[1] and offset.time_sec >= time_range[0]
        ):
            # Adjust for time range if needed
            start_sec = max(
                onset.time_sec, time_range[0] if time_range else onset.time_sec
            )
            end_sec = min(
                offset.time_sec, time_range[1] if time_range else offset.time_sec
            )
            ax2.axvspan(start_sec, end_sec, alpha=0.2, color="green")

    ax2.set_title("Processed Signal with Onset/Offset Detection")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Amplitude")
    ax2.legend(loc="upper right")

    # Set main title
    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()

    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    # Show figure if requested
    if show:
        plt.show()

    return fig


@public_api
def plot_multi_channel_onsets(
    signals: Union[np.ndarray, torch.Tensor],
    detection_results: List[OnsetOffsetResult],
    fs: float,
    channel_names: Optional[List[str]] = None,
    title: str = "Multi-channel EMG Onset Detection",
    figsize: Tuple[int, int] = (12, 10),
    time_range: Optional[Tuple[float, float]] = None,
    dark_mode: bool = True,
    show_threshold: bool = True,
    save_path: Optional[str] = None,
    dpi: int = 300,
    show: bool = True,
):
    """
    Plot onset and offset detection results for multiple channels

    Args:
        signals: Multi-channel processed signals used for detection [channels, samples]
        detection_results: List of detection results for each channel
        fs: Sampling frequency in Hz
        channel_names: Optional list of channel names
        title: Plot title
        figsize: Figure size as (width, height) in inches
        time_range: Optional tuple of (start_time, end_time) in seconds to zoom
        dark_mode: If True, use seaborn dark theme
        show_threshold: Whether to show threshold lines
        save_path: Optional path to save the figure
        dpi: Resolution for saved figure
        show: If True, call plt.show() to display the plot

    Returns:
        The matplotlib figure object
    """
    # Convert to numpy if tensor
    if isinstance(signals, torch.Tensor):
        signals_np = signals.detach().cpu().numpy()
    else:
        signals_np = signals

    # Get number of channels
    n_channels = signals_np.shape[0]

    # Create default channel names if not provided
    if channel_names is None:
        channel_names = [f"Channel {i + 1}" for i in range(n_channels)]

    # Ensure we have detection results for each channel
    if len(detection_results) != n_channels:
        raise ValueError(
            f"Number of detection results ({len(detection_results)}) does not match number of channels ({n_channels})"
        )

    # Create time vector
    time = np.arange(signals_np.shape[1]) / fs

    # Apply time range if specified
    if time_range is not None:
        start_idx = max(0, int(time_range[0] * fs))
        end_idx = min(signals_np.shape[1], int(time_range[1] * fs))
        time_slice = slice(start_idx, end_idx)
        time = time[time_slice]
        signals_np = signals_np[:, time_slice]
    else:
        time_slice = slice(None)

    # Set theme
    if dark_mode:
        sns.set_theme(style="darkgrid")
    else:
        sns.set_theme(style="whitegrid")

    # Create figure
    fig = plt.figure(figsize=figsize)

    # Create subplots for each channel
    for i in range(n_channels):
        ax = plt.subplot(n_channels, 1, i + 1)

        # Plot signal
        ax.plot(time, signals_np[i], color="blue", label=channel_names[i])

        # Show threshold if requested
        if show_threshold and detection_results[i].threshold_value is not None:
            ax.axhline(
                detection_results[i].threshold_value,
                color="orange",
                linestyle="-",
                alpha=0.7,
                label=f"Threshold: {detection_results[i].threshold_value:.3f}",
            )

        # Mark onsets and offsets
        onset_x, onset_y = [], []
        offset_x, offset_y = [], []

        for onset in detection_results[i].onsets:
            if time_range is None or (
                onset.time_sec >= time_range[0] and onset.time_sec <= time_range[1]
            ):
                ax.axvline(onset.time_sec, color="g", linestyle="--", alpha=0.7)
                onset_x.append(onset.time_sec)
                onset_y.append(onset.amplitude)

        for offset in detection_results[i].offsets:
            if time_range is None or (
                offset.time_sec >= time_range[0] and offset.time_sec <= time_range[1]
            ):
                ax.axvline(offset.time_sec, color="r", linestyle="--", alpha=0.7)
                offset_x.append(offset.time_sec)
                offset_y.append(offset.amplitude)

        # Plot onset and offset points
        ax.scatter(onset_x, onset_y, color="green", s=40, label="Onset", zorder=5)
        ax.scatter(offset_x, offset_y, color="red", s=40, label="Offset", zorder=5)

        # Shade the active regions
        for onset, offset in zip(
            detection_results[i].onsets, detection_results[i].offsets
        ):
            if time_range is None or (
                onset.time_sec <= time_range[1] and offset.time_sec >= time_range[0]
            ):
                # Adjust for time range if needed
                start_sec = max(
                    onset.time_sec, time_range[0] if time_range else onset.time_sec
                )
                end_sec = min(
                    offset.time_sec, time_range[1] if time_range else offset.time_sec
                )
                ax.axvspan(start_sec, end_sec, alpha=0.2, color="green")

        # Set axis labels
        ax.set_ylabel("Amplitude")

        # Only show x-axis label for bottom subplot
        if i == n_channels - 1:
            ax.set_xlabel("Time (s)")
        else:
            ax.set_xticklabels([])

        # Add legend to first subplot only
        if i == 0:
            ax.legend(loc="upper right")

        # Add channel name as title
        ax.set_title(channel_names[i])

    # Set main title
    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()

    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    # Show figure if requested
    if show:
        plt.show()

    return fig


@public_api
def plot_multichannel_comparison(
    signals: Union[np.ndarray, torch.Tensor],
    reference_channel: int,
    detection_results: List[OnsetOffsetResult],
    fs: float,
    channel_names: Optional[List[str]] = None,
    title: str = "Multi-channel Onset Timing Comparison",
    figsize: Tuple[int, int] = (12, 8),
    time_range: Optional[Tuple[float, float]] = None,
    dark_mode: bool = True,
    save_path: Optional[str] = None,
    dpi: int = 300,
    show: bool = True,
):
    """
    Plot a comparison of onset timings across multiple channels relative to a reference channel

    Args:
        signals: Multi-channel processed signals used for detection [channels, samples]
        reference_channel: Index of the reference channel
        detection_results: List of detection results for each channel
        fs: Sampling frequency in Hz
        channel_names: Optional list of channel names
        title: Plot title
        figsize: Figure size as (width, height) in inches
        time_range: Optional tuple of (start_time, end_time) in seconds to zoom
        dark_mode: If True, use seaborn dark theme
        save_path: Optional path to save the figure
        dpi: Resolution for saved figure
        show: If True, call plt.show() to display the plot

    Returns:
        The matplotlib figure object
    """
    # Convert to numpy if tensor
    if isinstance(signals, torch.Tensor):
        signals_np = signals.detach().cpu().numpy()
    else:
        signals_np = signals

    # Get number of channels
    n_channels = signals_np.shape[0]

    # Create default channel names if not provided
    if channel_names is None:
        channel_names = [f"Channel {i + 1}" for i in range(n_channels)]

    # Ensure we have detection results for each channel
    if len(detection_results) != n_channels:
        raise ValueError(
            f"Number of detection results ({len(detection_results)}) does not match number of channels ({n_channels})"
        )

    # Ensure reference channel is valid
    if reference_channel < 0 or reference_channel >= n_channels:
        raise ValueError(
            f"Reference channel index {reference_channel} out of range (0-{n_channels - 1})"
        )

    # Set theme
    if dark_mode:
        sns.set_theme(style="darkgrid")
    else:
        sns.set_theme(style="whitegrid")

    # Create figure
    fig = plt.figure(figsize=figsize)

    # Plot normalized signals and onsets
    ax1 = plt.subplot(2, 1, 1)

    # Normalize signals for better visualization
    norm_signals = np.zeros_like(signals_np)
    for i in range(n_channels):
        chan_min = np.min(signals_np[i])
        chan_max = np.max(signals_np[i])
        if chan_max > chan_min:
            norm_signals[i] = (signals_np[i] - chan_min) / (chan_max - chan_min)
            # Offset each channel for clarity
            norm_signals[i] = norm_signals[i] + i

    # Create time vector
    time = np.arange(signals_np.shape[1]) / fs

    # Apply time range if specified
    if time_range is not None:
        start_idx = max(0, int(time_range[0] * fs))
        end_idx = min(signals_np.shape[1], int(time_range[1] * fs))
        time_slice = slice(start_idx, end_idx)
        time = time[time_slice]
        norm_signals = norm_signals[:, time_slice]
    else:
        time_slice = slice(None)

    # Plot each normalized signal with its onsets
    colors = plt.cm.tab10(np.linspace(0, 1, n_channels))

    for i in range(n_channels):
        # Plot normalized signal
        ax1.plot(
            time, norm_signals[i], color=colors[i], label=channel_names[i], alpha=0.7
        )

        # Mark onsets
        for onset in detection_results[i].onsets:
            if time_range is None or (
                onset.time_sec >= time_range[0] and onset.time_sec <= time_range[1]
            ):
                # Calculate normalized amplitude for the onset point
                chan_min = np.min(signals_np[i])
                chan_max = np.max(signals_np[i])
                if chan_max > chan_min:
                    norm_amp = (onset.amplitude - chan_min) / (chan_max - chan_min) + i
                else:
                    norm_amp = i

                ax1.scatter(
                    onset.time_sec,
                    norm_amp,
                    color=colors[i],
                    marker="o",
                    s=50,
                    zorder=10,
                )
                ax1.axvline(onset.time_sec, color=colors[i], linestyle="--", alpha=0.3)

    ax1.set_title("Normalized Signals with Onset Detection")
    ax1.set_ylabel("Channel")
    ax1.set_yticks(np.arange(n_channels))
    ax1.set_yticklabels(channel_names)
    ax1.legend(loc="upper right")

    # Plot timing differences relative to reference channel
    ax2 = plt.subplot(2, 1, 2)

    # Get reference onsets
    ref_onsets = detection_results[reference_channel].onsets

    # Skip if no reference onsets
    if len(ref_onsets) == 0:
        ax2.text(
            0.5,
            0.5,
            "No onsets detected in reference channel",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax2.transAxes,
        )
    else:
        # Calculate timing differences for each channel relative to reference
        timing_diffs = []
        chan_indices = []
        diff_values = []

        for i in range(n_channels):
            if i == reference_channel:
                continue

            chan_onsets = detection_results[i].onsets

            # Match each reference onset with closest channel onset
            for ref_idx, ref_onset in enumerate(ref_onsets):
                # Skip if out of time range
                if time_range is not None and (
                    ref_onset.time_sec < time_range[0]
                    or ref_onset.time_sec > time_range[1]
                ):
                    continue

                # Find closest onset in this channel
                closest_idx = None
                min_diff = float("inf")

                for ch_idx, ch_onset in enumerate(chan_onsets):
                    diff = ch_onset.time_sec - ref_onset.time_sec
                    abs_diff = abs(diff)

                    # Only consider onsets within reasonable time window (e.g., ±500ms)
                    if abs_diff < 0.5 and abs_diff < min_diff:
                        min_diff = abs_diff
                        closest_idx = ch_idx

                if closest_idx is not None:
                    timing_diff = chan_onsets[closest_idx].time_sec - ref_onset.time_sec
                    timing_diffs.append(timing_diff * 1000)  # Convert to ms
                    chan_indices.append(i)
                    diff_values.append((ref_onset.time_sec, timing_diff * 1000))

        # Plot timing differences as scatter points
        for i in range(n_channels):
            if i == reference_channel:
                continue

            # Extract differences for this channel
            x_vals = [
                t[0] for t in diff_values if chan_indices[diff_values.index(t)] == i
            ]
            y_vals = [
                t[1] for t in diff_values if chan_indices[diff_values.index(t)] == i
            ]

            ax2.scatter(
                x_vals, y_vals, color=colors[i], label=channel_names[i], s=50, alpha=0.7
            )

            # Optionally add trend line
            if len(x_vals) > 1:
                ax2.plot(x_vals, y_vals, color=colors[i], alpha=0.4)

        # Add zero line
        ax2.axhline(
            0,
            color="black",
            linestyle="-",
            alpha=0.5,
            label=f"Reference: {channel_names[reference_channel]}",
        )

        # Set axis limits and labels
        ax2.set_ylim([-200, 200])  # ±200ms or adjust as needed
        ax2.set_ylabel("Timing Difference (ms)")
        ax2.set_xlabel("Time (s)")
        ax2.set_title(
            f"Onset Timing Differences Relative to {channel_names[reference_channel]}"
        )

        # Add legend
        ax2.legend(loc="upper right")

    # Set main title
    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()

    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    # Show figure if requested
    if show:
        plt.show()

    return fig
