# %%
# Get a sample batch
sample_batch = next(iter(dataloader))

# Set up variables for visualization
tsx = sample_batch
fs = fs
channel_idx = 0  # First channel
window_idx = 0  # First window in batch
# %%
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

# Extract single window to visualize and ensure it's on CPU as numpy array
window = tsx[window_idx, channel_idx].detach().clone().cpu().numpy()
ts = np.arange(len(window)) / fs

print(f"Processing window with shape: {window.shape}")

# Create envelope detectors WITH built-in normalization
envelope_detectors = [
    # Classic TKEO with normalization
    create_tkeo_envelope(
        tkeo_type="tkeo2",
        rectification="abs",
        lowpass_cutoff=20.0,
        normalize=True,
    ),
    # Modified TKEO with normalization
    create_tkeo_envelope(
        tkeo_type="mtkeo",
        rectification="abs",
        lowpass_cutoff=20.0,
        normalize=True,
        tkeo_params={"k1": 1.0, "k2": 0.5, "k3": 0.5},
    ),
]

envelope_names = ["tkeo2", "Modified TKEO"]

# Create only the fixed threshold detector
threshold_detectors = [
    # Fixed threshold
    create_fixed_threshold_detector(
        threshold=0.05,
        min_duration_ms=50.0,
        hysteresis=0.01,
        norm="minmax",  # Using internal normalization
    ),
]

threshold_names = ["Fixed Threshold"]

# Initialize all processors
for envelope in envelope_detectors:
    envelope.initialize(fs)

# Initialize threshold detector
for detector in threshold_detectors:
    detector.initialize(fs)

# Create a large figure for visualization
plt.figure(figsize=(15, 12))
gs = GridSpec(
    len(envelope_detectors) + 2, 2
)  # Changed to 2 columns since we only have 1 threshold method

# Plot raw filtered tsx
ax_raw = plt.subplot(gs[0, :])
ax_raw.plot(ts, window)
ax_raw.set_title("Filtered EMG Signal")
ax_raw.set_xlabel("Time (s)")
ax_raw.set_ylabel("Amplitude")

# Process and visualize each combination
all_detections = []

# Final ground truth from ensemble
final_onset_probs = np.zeros_like(window)
final_offset_probs = np.zeros_like(window)

# Process each envelope method
for e_idx, (envelope, env_name) in enumerate(zip(envelope_detectors, envelope_names)):
    # Prepare tsx tensor for envelope processing
    tsx_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(
        0
    )  # Add channel dim only

    try:
        # Apply envelope detection (now with built-in normalization)
        env_result = envelope(tsx_tensor)

        # Check the shape and handle multi-channel output if needed
        if env_result.ndim > 2:
            print(
                f"Warning: Envelope '{env_name}' returned {env_result.shape[0]} channels"
            )
            # Use the first channel or average across channels
            env_signal = env_result[0].cpu().numpy()  # Take first channel
        else:
            # The output is [channels, ts], so we need to get the single channel
            env_signal = env_result.squeeze(0).cpu().numpy()  # Remove channel dimension

        print(
            f"Envelope '{env_name}' output shape: {env_signal.shape}, ts shape: {ts.shape}"
        )

        # Ensure env_signal and ts have matching dimensions
        if env_signal.ndim > 1 and env_signal.shape[0] != ts.shape[0]:
            # Might need to transpose
            if env_signal.shape[1] == ts.shape[0]:
                env_signal = env_signal.T
            else:
                # If the shapes still don't match, use first dimension that matches ts
                for i in range(env_signal.shape[0]):
                    if env_signal[i].shape[0] == ts.shape[0]:
                        env_signal = env_signal[i]
                        break

        # Plot envelope result
        ax_env = plt.subplot(gs[e_idx + 1, 0])
        ax_env.plot(ts, env_signal)
        ax_env.set_title(f"{env_name} (Normalized)")
        ax_env.set_ylabel("Amplitude")

        # Now we apply fixed threshold detection to the normalized envelope
        for t_idx, (detector, thresh_name) in enumerate(
            zip(threshold_detectors, threshold_names)
        ):
            # Apply threshold detection directly to the normalized envelope
            detection_result = detector(env_signal)

            # Store detection for ensemble
            all_detections.append((detection_result, f"{env_name} + {thresh_name}"))

            # Plot detection result
            ax_thresh = plt.subplot(gs[e_idx + 1, t_idx + 1])
            ax_thresh.plot(ts, env_signal, color="blue", alpha=0.7)

            # Mark onsets and offsets
            for onset in detection_result.onsets:
                idx = max(0, min(int(onset.sample_index), len(ts) - 1))
                ax_thresh.axvline(x=ts[idx], color="green", linestyle="--", alpha=0.7)

            for offset in detection_result.offsets:
                idx = max(0, min(int(offset.sample_index), len(ts) - 1))
                ax_thresh.axvline(x=ts[idx], color="red", linestyle="--", alpha=0.7)

            # Show threshold
            ax_thresh.axhline(
                y=detection_result.threshold_value,
                color="orange",
                linestyle="-",
                alpha=0.5,
                label=f"Threshold: {detection_result.threshold_value:.3f}",
            )

            ax_thresh.legend()
            ax_thresh.set_title(f"{env_name} + {thresh_name}")

            # Update combined ground truth with Gaussian activations
            for onset, offset in zip(detection_result.onsets, detection_result.offsets):
                onset_idx = max(0, min(int(onset.sample_index), len(window) - 1))
                offset_idx = max(0, min(int(offset.sample_index), len(window) - 1))

                # Create Gaussian activations around onset/offset points
                sigma = int(0.01 * fs)  # 10ms spread

                for t in range(len(window)):
                    # Gaussian activation for onset
                    onset_val = np.exp(-0.5 * ((t - onset_idx) / sigma) ** 2)
                    if onset_val > final_onset_probs[t]:
                        final_onset_probs[t] = onset_val

                    # Gaussian activation for offset
                    offset_val = np.exp(-0.5 * ((t - offset_idx) / sigma) ** 2)
                    if offset_val > final_offset_probs[t]:
                        final_offset_probs[t] = offset_val
    except Exception as e:
        print(f"Error processing {env_name}: {e}")

# Visualize the final ground truth labels
ax_final = plt.subplot(gs[-1, :])

# Normalize the window to range [-1, 1] for better visualization
normalized_window = window / (np.max(np.abs(window)) + 1e-10)

ax_final.plot(ts, normalized_window, color="gray", alpha=0.5, label="Normalized Signal")
ax_final.plot(ts, final_onset_probs, "g-", label="Onset Probability")
ax_final.plot(ts, final_offset_probs, "r-", label="Offset Probability")

# Mark points where probability exceeds threshold
threshold = 0.5
onset_events = np.where(final_onset_probs > threshold)[0]
offset_events = np.where(final_offset_probs > threshold)[0]

# Mark onset and offset points
if len(onset_events) > 0:
    ax_final.scatter(
        ts[onset_events],
        final_onset_probs[onset_events],
        color="green",
        s=100,
        marker="o",
        label="Onset Events",
    )
if len(offset_events) > 0:
    ax_final.scatter(
        ts[offset_events],
        final_offset_probs[offset_events],
        color="red",
        s=100,
        marker="o",
        label="Offset Events",
    )

ax_final.set_title("Final Ensemble Ground Truth")
ax_final.set_xlabel("Time (s)")
ax_final.set_ylabel("Probability")
ax_final.legend()
ax_final.grid(True)

plt.tight_layout()
plt.show()

print(
    f"Found {np.sum(final_onset_probs > 0.95)} onset events and {np.sum(final_offset_probs > 0.95)} offset events"
)


# %%
