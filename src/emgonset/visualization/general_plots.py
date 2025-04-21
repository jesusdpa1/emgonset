from typing import List, Optional, Tuple, Union

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.cm import get_cmap
from matplotlib.figure import Figure

from ..utils.internals import public_api


def plot_multi_channel_data(
    data: Union[np.ndarray, torch.Tensor],
    channels: Optional[List[int]] = None,
    fs: Optional[float] = None,
    figsize: Tuple[int, int] = (15, 6),
    title: str = "Multi-Channel Data",
    channel_names: Optional[List[str]] = None,
    color_mode: str = "colormap",  # "colormap" or "single"
    colormap: str = "Set1",  # Used when color_mode is "colormap"
    color: str = "black",  # Used when color_mode is "single"
    time_window: Optional[Tuple[float, float]] = None,
    y_spread: float = 1.0,  # Vertical spacing between channels
    y_offset: float = 0.0,  # Baseline offset
    line_width: float = 1.0,  # Line width for signals
    alpha: float = 0.8,  # Transparency level
    grid: bool = True,  # Show grid lines
    show_channel_labels: bool = True,  # Show channel labels on y-axis
    normalize: bool = True,  # Normalize each channel's amplitude
    norm_scale: float = 0.4,  # Scale factor for normalized signals
    dark_mode: bool = True,  # Dark mode for plot
    show: bool = True,  # Show the plot
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> Figure:
    """
    Plot multiple channels of time-series data in a single axis with customizable spacing.

    Parameters:
    -----------
    data : numpy array or torch tensor
        Raw signal data (samples × channels)
    channels : list of int, optional
        List of channel indices to plot. If None, plot all channels.
    fs : float, optional
        Sampling frequency in Hz. Required to convert samples to time.
    figsize : tuple, default (15, 6)
        Figure size (width, height)
    title : str, default "Multi-Channel Data"
        Title for the plot
    channel_names : list of str, optional
        Custom names for each channel. If None, uses "Channel {index}"
    color_mode : str, default "colormap"
        How to color channels:
        - "colormap": Use matplotlib colormap to assign different colors
        - "single": Use a single color for all channels
    colormap : str, default "Set1"
        Matplotlib colormap name to use when color_mode is "colormap"
    color : str, default "black"
        Color to use for all channels when color_mode is "single"
    time_window : tuple, optional
        Time window to plot (start_time, end_time) in seconds.
    y_spread : float, default 1.0
        Controls vertical spacing between channels
    y_offset : float, default 0.0
        Baseline offset for all channels
    line_width : float, default 1.0
        Width of signal lines
    alpha : float, default 0.8
        Transparency level for signals
    grid : bool, default True
        Whether to show grid lines
    show_channel_labels : bool, default True
        Whether to show channel labels on y-axis
    normalize : bool, default True
        Whether to normalize each channel's amplitude
    norm_scale : float, default 0.4
        Scale factor for normalized signals
    dark_mode : bool, default True
        Use dark mode theme
    show : bool, default True
        Display the plot
    save_path : str, optional
        Path to save the figure
    dpi : int, default 300
        Resolution for saved figure

    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
    """
    # Convert to numpy if tensor
    if hasattr(data, "detach") and hasattr(data, "cpu"):  # Check for PyTorch tensor
        data_np = data.detach().cpu().numpy()
    else:
        data_np = np.asarray(data)

    # Ensure data is 2D
    if data_np.ndim == 1:
        data_np = data_np.reshape(-1, 1)

    # Validate input shape (samples, channels)
    n_samples, n_channels = data_np.shape

    # Set theme
    if dark_mode:
        sns.set_theme(style="darkgrid")
    else:
        sns.set_theme(style="whitegrid")

    # Default to all channels if not specified
    if channels is None:
        channels = list(range(n_channels))

    # Generate default channel names if not provided
    if channel_names is None:
        channel_names = [f"Channel {i}" for i in channels]

    # Ensure fs is provided
    if fs is None:
        fs = 1000  # Default to 1000 Hz if not specified
        print("Warning: Sampling frequency not provided. Assuming 1000 Hz.")

    # Prepare figure with a single axis
    fig, ax = plt.subplots(figsize=figsize)

    # Color setup based on color_mode
    import matplotlib.cm as cm

    if color_mode == "colormap":
        cmap = cm.get_cmap(colormap)

    # Compute full time array
    time = np.arange(n_samples) / fs

    # Set up the time bounds for the plot
    start_idx = 0
    end_idx = n_samples
    if time_window is not None:
        start_idx = max(0, int(time_window[0] * fs))
        end_idx = min(n_samples, int(time_window[1] * fs))
        ax.set_xlim(time_window)
    else:
        ax.set_xlim(time[start_idx], time[end_idx - 1])

    # Get time segment to plot
    subset_time = time[start_idx:end_idx]

    # Channel positions for plotting
    channel_positions = []

    # Process each channel
    for idx, channel in enumerate(channels):
        # Calculate vertical offset for this channel
        channel_offset = y_offset + (len(channels) - 1 - idx) * y_spread
        channel_positions.append(channel_offset)

        # Get data for this channel
        subset_data = data_np[start_idx:end_idx, channel]

        # Normalize if requested
        if normalize:
            max_amplitude = np.max(np.abs(subset_data))
            if max_amplitude > 0:
                norm_data = (
                    subset_data / max_amplitude * (y_spread * norm_scale)
                    + channel_offset
                )
            else:
                norm_data = np.zeros_like(subset_data) + channel_offset
        else:
            norm_data = subset_data + channel_offset

        # Choose color based on color_mode
        if color_mode == "colormap":
            plot_color = cmap(idx / max(1, len(channels) - 1))
        else:  # "single" color mode
            plot_color = color

        # Plot the channel data
        ax.plot(
            subset_time,
            norm_data,
            color=plot_color,
            alpha=alpha,
            linewidth=line_width,
            label=channel_names[idx],
        )

    # Add y-axis ticks for channel positions if requested
    if show_channel_labels:
        ax.set_yticks(channel_positions)
        ax.set_yticklabels(channel_names)
    else:
        ax.set_yticks([])  # Hide y-axis ticks

    # Add grid if requested
    if grid:
        ax.grid(True, alpha=0.8)

    # Customize appearance
    ax.set_xlabel("Time (s)")
    if show_channel_labels:
        ax.set_ylabel("Channels")

    # Set title
    ax.set_title(title)

    plt.tight_layout()

    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    # Show figure if requested
    if show:
        plt.show()

    return fig


@public_api
def visualize_emg_processing(
    raw_data: torch.Tensor,
    processed_data: torch.Tensor,
    fs: float,
    step_names: Optional[List[str]] = None,
    time_range: Optional[Tuple[float, float]] = None,
    figsize: Tuple[int, int] = (14, 10),
    dark_mode: bool = True,
    save_path: Optional[str] = None,
):
    """
    Visualize raw EMG data alongside processed versions to compare effects of processing steps.

    Args:
        raw_data: Raw EMG data tensor of shape [channels, samples]
        processed_data: List of processed data tensors, each with shape [channels, samples]
        fs: Sampling frequency in Hz
        step_names: Names of processing steps (length should match processed_data)
        time_range: Optional tuple of (start_time, end_time) in seconds to zoom
        figsize: Figure size as (width, height) in inches
        dark_mode: If True, use seaborn dark theme
        save_path: Optional path to save the figure

    Returns:
        The matplotlib figure object
    """
    # Handle single processed data tensor case
    if processed_data.ndim == 2:
        processed_data = [processed_data]

    # Set default step names if not provided
    if step_names is None:
        step_names = [f"Processing Step {i + 1}" for i in range(len(processed_data))]

    # Set theme
    if dark_mode:
        with sns.axes_style("darkgrid"):
            sns.set_theme(style="darkgrid")
    else:
        with sns.axes_style("whitegrid"):
            sns.set_theme(style="whitegrid")

    # Get number of channels
    n_channels = raw_data.shape[0]

    # Create figure
    fig = plt.figure(figsize=figsize)

    # Calculate total number of plots
    n_rows = n_channels
    n_cols = 1 + len(processed_data)

    # Create time vector
    time = np.arange(raw_data.shape[1]) / fs

    # Apply time range if specified
    if time_range is not None:
        start_idx = max(0, int(time_range[0] * fs))
        end_idx = min(raw_data.shape[1], int(time_range[1] * fs))
        time_slice = slice(start_idx, end_idx)
        time = time[time_slice]
    else:
        time_slice = slice(None)

    # Create color palettes
    raw_color = sns.color_palette("Set2")[0]
    processed_colors = sns.color_palette("Set2")[1:]

    # Plot raw and processed data for each channel
    for ch in range(n_channels):
        # Plot raw data
        ax = plt.subplot(n_rows, n_cols, ch * n_cols + 1)
        ax.plot(time, raw_data[ch, time_slice], color=raw_color)
        ax.set_title(f"Raw - Channel {ch + 1}" if ch == 0 else f"Channel {ch + 1}")
        ax.set_ylabel("Amplitude")
        if ch < n_rows - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Time (s)")

        # Plot each processing step
        for i, proc_data in enumerate(processed_data):
            idx = ch * n_cols + i + 2
            ax = plt.subplot(n_rows, n_cols, idx)

            # Handle potentially different lengths of processed data
            proc_time = time
            proc_data_slice = proc_data[ch, time_slice]
            if proc_data_slice.shape[0] != time.shape[0]:
                # Adjust time vector if processed data has different length
                proc_time = np.linspace(time[0], time[-1], proc_data_slice.shape[0])

            ax.plot(
                proc_time,
                proc_data_slice,
                color=processed_colors[i % len(processed_colors)],
            )

            if ch == 0:
                ax.set_title(step_names[i])
            if ch < n_rows - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("Time (s)")

    plt.tight_layout()

    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

    return fig
