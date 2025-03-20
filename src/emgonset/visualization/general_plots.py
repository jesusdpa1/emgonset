from typing import List, Optional, Tuple, Union

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.cm import get_cmap

from ..utils.internals import public_api


@public_api
def plot_time_series(
    data: Union[torch.Tensor, np.ndarray],
    fs: float,
    title: Optional[str] = "EMG Signal",
    channel_names: Optional[List[str]] = None,
    color: Optional[Union[str, List[str]]] = None,
    cmap: Optional[str] = None,
    time_range: Optional[Tuple[float, float]] = None,
    figsize: Tuple[int, int] = (12, 6),
    subplots: bool = True,
    dark_mode: bool = True,
    grid: bool = True,
    show: bool = True,
    save_path: Optional[str] = None,
    dpi: int = 300,
):
    """
    Plot EMG signal data with customizable styling.

    Args:
        data: EMG data tensor or array of shape [channels, samples]
        fs: Sampling frequency in Hz
        title: Main plot title
        channel_names: List of names for each channel
        color: Color(s) for the plots. Can be a single color name/hex or a list for each channel
        cmap: Matplotlib colormap name to use for channels (ignored if color is specified)
        time_range: Optional tuple of (start_time, end_time) in seconds to zoom
        figsize: Figure size as (width, height) in inches
        subplots: If True, plot each channel in separate subplots; if False, overlay them
        dark_mode: If True, use seaborn dark theme
        grid: If True, show grid lines
        show: If True, call plt.show() to display the plot
        save_path: Optional path to save the figure
        dpi: Resolution for saved figure

    Returns:
        The matplotlib figure object
    """
    # Convert to numpy if tensor
    if isinstance(data, torch.Tensor):
        data_np = data.detach().cpu().numpy()
    else:
        data_np = data

    # Get number of channels and samples
    n_channels, n_samples = data_np.shape

    # Create time vector
    time = np.arange(n_samples) / fs

    # Apply time range if specified
    if time_range is not None:
        start_idx = max(0, int(time_range[0] * fs))
        end_idx = min(n_samples, int(time_range[1] * fs))
        data_np = data_np[:, start_idx:end_idx]
        time = time[start_idx:end_idx]

    # Set default channel names if not provided
    if channel_names is None:
        channel_names = [f"Channel {i + 1}" for i in range(n_channels)]
    # Create figure
    fig = plt.figure(figsize=figsize)
    # Set theme
    if dark_mode:
        sns.set_theme(style="darkgrid")
    else:
        sns.set_theme(style="whitegrid")

    # Determine colors
    if color is not None:
        # If single color provided
        if isinstance(color, str):
            colors = [color] * n_channels
        else:
            colors = color
    elif cmap is not None:
        # Use colormap
        cmap_obj = get_cmap(cmap)
        colors = [cmap_obj(i / max(1, n_channels - 1)) for i in range(n_channels)]
    else:
        # Default to tab10 colormap
        cmap_obj = get_cmap("tab10")
        colors = [cmap_obj(i % 10) for i in range(n_channels)]

    # Plot channels
    if subplots:
        for i in range(n_channels):
            ax = plt.subplot(n_channels, 1, i + 1)
            ax.plot(time, data_np[i], color=colors[i])
            ax.set_ylabel("Amplitude")
            ax.set_title(channel_names[i])

            if i < n_channels - 1:  # Only show x-label for bottom subplot
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("Time (s)")

            if grid:
                ax.grid(True, alpha=0.8)
    else:
        # Overlay all channels on a single plot
        ax = plt.subplot(111)
        for i in range(n_channels):
            ax.plot(time, data_np[i], color=colors[i], label=channel_names[i])

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.legend(loc="best")

        if grid:
            ax.grid(True, alpha=0.8)

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
