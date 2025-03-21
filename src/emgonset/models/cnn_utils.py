import numpy as np
from numba import jit


@jit(nopython=True, cache=True)
def calculate_gaussian_activations(
    onset_labels, offset_labels, detection_onsets, detection_offsets, time_steps, fs
):
    """
    Numba-optimized function to calculate Gaussian activations around onset/offset points

    Args:
        onset_labels: NumPy array to store onset activations
        offset_labels: NumPy array to store offset activations
        detection_onsets: List of onset indices
        detection_offsets: List of offset indices
        time_steps: Total number of time steps
        fs: Sampling frequency
    """
    # Create binary activations and Gaussian smoothed labels
    for i in range(len(detection_onsets)):
        onset_idx = min(max(detection_onsets[i], 0), time_steps - 1)
        offset_idx = min(max(detection_offsets[i], 0), time_steps - 1)

        # Create Gaussian activations around onset/offset points
        sigma = int(0.01 * fs)  # 10ms spread
        window_size = 5 * sigma  # Only calculate within 5 standard deviations

        # Onset
        start_t = max(0, onset_idx - window_size)
        end_t = min(time_steps, onset_idx + window_size + 1)
        for t in range(start_t, end_t):
            onset_val = np.exp(-0.5 * ((t - onset_idx) / sigma) ** 2)
            if onset_val > onset_labels[t]:
                onset_labels[t] = onset_val

        # Offset
        start_t = max(0, offset_idx - window_size)
        end_t = min(time_steps, offset_idx + window_size + 1)
        for t in range(start_t, end_t):
            offset_val = np.exp(-0.5 * ((t - offset_idx) / sigma) ** 2)
            if offset_val > offset_labels[t]:
                offset_labels[t] = offset_val

    return onset_labels, offset_labels
