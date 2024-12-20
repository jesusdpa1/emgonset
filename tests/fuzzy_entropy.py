# %%
import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import time


@nb.njit(fastmath=True)
def fast_fuzzy_entropy(x, m, r):
    """
    Optimized Fuzzy Entropy calculation using Numba JIT compilation

    Parameters:
    -----------
    x : ndarray
        Input time series
    m : int
        Embedding dimension
    r : float
        Similarity threshold

    Returns:
    --------
    float
        Fuzzy Entropy value
    """

    # Embedding function
    def embed_sequence(x, m, delay=1):
        n = len(x) - (m - 1) * delay
        X = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                X[i, j] = x[i + j * delay]
        return X

    # Vectorized distance calculation
    def calculate_distance_matrix(x_m):
        n = x_m.shape[0]
        distance_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                distance_matrix[i, j] = np.max(np.abs(x_m[i] - x_m[j]))
        return distance_matrix

    # Fuzzy membership computation
    def fuzzy_membership(distance_matrix, r):
        return np.exp(-np.power(distance_matrix / r, 2))

    # Embedding for m and m+1
    X_m = embed_sequence(x, m)
    X_m1 = embed_sequence(x, m + 1)

    # Distance matrices
    distance_matrix_m = calculate_distance_matrix(X_m)
    distance_matrix_m1 = calculate_distance_matrix(X_m1)

    # Fuzzy membership
    fm_m = fuzzy_membership(distance_matrix_m, r)
    fm_m1 = fuzzy_membership(distance_matrix_m1, r)

    # Compute fuzzy entropy
    n_m = len(X_m)
    phi_m = np.sum(fm_m) / (n_m * (n_m - 1))
    phi_m1 = np.sum(fm_m1) / (len(X_m1) * (len(X_m1) - 1))

    return -np.log(max(phi_m1 / phi_m, 1e-10))


@nb.njit
def compute_window_entropy(signal_norm, window_start, window_size, m, r):
    """
    Compute fuzzy entropy for a single window

    Parameters:
    -----------
    signal_norm : ndarray
        Normalized signal
    window_start : int
        Start index of the window
    window_size : int
        Size of the window
    m : int
        Embedding dimension
    r : float
        Similarity threshold

    Returns:
    --------
    float
        Fuzzy entropy value for the window
    """
    window = signal_norm[window_start : window_start + window_size]
    return fast_fuzzy_entropy(window, m, r)


@nb.njit
def fuzzy_entropy_envelope(signal_norm, window_size, stride, m, r, aggregation="max"):
    """
    Compute fuzzy entropy envelope

    Parameters:
    -----------
    signal_norm : ndarray
        Normalized input signal
    window_size : int
        Size of sliding window
    stride : int
        Stride between windows
    m : int
        Embedding dimension
    r : float
        Similarity threshold
    aggregation : str
        Aggregation method

    Returns:
    --------
    ndarray
        Envelope values
    """
    # Initialize envelope
    envelope = np.zeros_like(signal_norm)

    # Compute windows
    for i in range(0, len(signal_norm) - window_size + 1, stride):
        fe = compute_window_entropy(signal_norm, i, window_size, m, r)

        # Aggregation methods
        if aggregation == "max":
            for j in range(window_size):
                envelope[i + j] = max(envelope[i + j], fe)
        elif aggregation == "mean":
            for j in range(window_size):
                envelope[i + j] = max(envelope[i + j], fe)
        elif aggregation == "min":
            for j in range(window_size):
                envelope[i + j] = min(envelope[i + j], fe)

    return envelope


def extract_envelope_fuzzy_entropy(
    signal_data,
    window_size=50,
    stride=None,
    overlap=0.5,
    m=2,
    r_factor=0.2,
    aggregation="max",
):
    """
    Extract signal envelope using optimized Fuzzy Entropy

    Parameters:
    -----------
    signal_data : ndarray
        Input time series signal
    window_size : int, optional
        Size of the sliding window (default 50)
    stride : int, optional
        Manual stride length
    overlap : float, optional
        Proportion of window overlap (0 to 1, default 0.5)
    m : int, optional
        Embedding dimension (default 2)
    r_factor : float, optional
        Similarity threshold factor (default 0.2)
    aggregation : str, optional
        Method of aggregating window results (default 'max')

    Returns:
    --------
    ndarray
        Extracted envelope
    """
    # Validate inputs
    if not 0 <= overlap < 1:
        raise ValueError("Overlap must be between 0 and 1")

    # Normalize signal
    signal_norm = (signal_data - np.mean(signal_data)) / np.std(signal_data)

    # Compute similarity threshold
    r = r_factor * np.std(signal_norm)

    # Calculate stride if not provided
    if stride is None:
        stride = int(window_size * (1 - overlap))

    # Compute envelope
    envelope = fuzzy_entropy_envelope(
        signal_norm, window_size, stride, m, r, aggregation
    )

    return envelope


def performance_comparison():
    """
    Compare performance of different implementations
    """
    # Generate a large test signal
    np.random.seed(42)
    signal_data = np.random.randn(10000)  # Large signal for performance test

    # Warm-up compilation
    _ = extract_envelope_fuzzy_entropy(signal_data[:1000])

    # Performance test
    start_time = time.time()
    envelope_optimized = extract_envelope_fuzzy_entropy(signal_data)
    optimized_time = time.time() - start_time

    # Plotting for verification
    plt.figure(figsize=(12, 6))
    plt.plot(signal_data, label="Original Signal", alpha=0.7)
    plt.plot(envelope_optimized, label="Fuzzy Entropy Envelope", linewidth=2)
    plt.title("Fuzzy Entropy Envelope Performance")
    plt.legend()
    plt.show()

    print(f"Optimized Implementation Time: {optimized_time:.4f} seconds")


# Performance demonstration
if __name__ == "__main__":
    performance_comparison()
# %%

signal_data = filtered_signal  # Large signal for performance test


# Performance test
start_time = time.time()
envelope_optimized = extract_envelope_fuzzy_entropy(
    signal_data[20000:25000], window_size=1000
)
optimized_time = time.time() - start_time

# Plotting for verification
plt.figure(figsize=(12, 6))
plt.plot(signal_data, label="Original Signal", alpha=0.7)
plt.plot(envelope_optimized, label="Fuzzy Entropy Envelope", linewidth=2)
plt.title("Fuzzy Entropy Envelope Performance")
plt.legend()
plt.show()

print(f"Optimized Implementation Time: {optimized_time:.4f} seconds")

# %%
