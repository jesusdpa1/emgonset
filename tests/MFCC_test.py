"""
Author: jpenaloza
Description: script to test MFCC
"""

# %%
# load the libraries required to run this script
# filesystem paths
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert, savgol_filter, find_peaks
from sklearn.preprocessing import MinMaxScaler
import spikeinterface.full as si

from spikeinterface.preprocessing import filter
import spikeinterface.widgets as sw
import librosa

# %%
# define home (Drive were data is located)
home = Path.home()
# define the path to the data file
data_path = r"E:\jpenalozaa\testSubject\24-11-03_3870-1_testSubject_Contusion + Baseline\01_baseline\testSubject-241103_01_baseline.Tbk"
# build the path to the data file
working_path = home.joinpath(data_path)
print(working_path)
# %%
recording = si.read_tdt(working_path, "10")
# %%

recording_cmr = si.common_reference(recording, reference="global", operator="median")
recording_f = si.bandpass_filter(recording_cmr, freq_min=300, freq_max=6000)
recording_w = si.whiten(recording_f)

# %%
w_ts = sw.plot_traces(recording_w, time_range=(10, 15))
# %%
# get new recording with the first 4 channels
channel_ids = recording.channel_ids
recording_right = recording_w.channel_slice(channel_ids=channel_ids[0:16])
recording_left = recording_w.channel_slice(channel_ids=channel_ids[16:33])
# %%
sw.plot_traces(recording_right)
# %%
right_sliced = recording_right.time_slice(10, 15)
right_data = right_sliced.get_traces()
plt.plot(np.mean(right_data, 1))
# %%
a = si.bandpass_filter(right_sliced, freq_min=1, freq_max=6000)
signal_ = a.get_traces()
signal_emg = np.median(signal_, 1)
# %%
import pywt

wavelet_ = "db4"
# Wavelet filtering
coeffs = pywt.wavedec(signal_emg, wavelet_)

# Denoise coefficients
coeffs[1:] = [pywt.threshold(c, 0.2, mode="soft") for c in coeffs[1:]]

# Reconstruct filtered signal
filtered_signal = pywt.waverec(coeffs, wavelet_)

# Plot original and filtered signals
plt.figure(figsize=(12, 6))
plt.plot(signal_emg, label="Original Signal")
plt.plot(filtered_signal, label="Filtered Signal")
plt.legend()
plt.show()
# %%
window_size = int(recording.sampling_frequency * 0.055)
scaler = MinMaxScaler()

emg_rect = np.abs(hilbert(filtered_signal))
emg_env = savgol_filter(emg_rect, window_size, 1)  # Savitzky-Golay filter

emg_norm = scaler.fit_transform(emg_env[np.newaxis])
plt.plot(emg_env)

# %%
level = 8
# Perform wavelet decomposition
coeffs = pywt.wavedec(filtered_signal, "db4", level)

# Reconstruct approximation coefficients
reconstruction = pywt.waverec([coeffs[0]] + [None] * level, "db4")

# Return absolute value as envelope
test = np.abs(reconstruction[: len(filtered_signal)])
plt.plot(test)
plt.plot(filtered_signal)
# %%
sr = recording.sampling_frequency
S = librosa.feature.melspectrogram(
    y=filtered_signal, sr=sr, n_mels=128, fmax=16000, hop_length=256
)

mfccs = librosa.feature.mfcc(
    y=filtered_signal, sr=sr, hop_length=256, htk=True, n_mfcc=20
)

# %%
plt.figure(figsize=(15, 6))
fig, ax = plt.subplots(nrows=2, sharex=True)
img = librosa.display.specshow(
    librosa.power_to_db(S, ref=np.max),
    x_axis="time",
    y_axis="mel",
    fmax=16000,
    ax=ax[0],
)
fig.colorbar(img, ax=[ax[0]])
ax[0].set(title="Mel spectrogram")
ax[0].label_outer()

img2 = librosa.display.specshow(mfccs, x_axis="time", ax=ax[1])
ax[1].set(title="HTK-style (dct_type=3)")
fig.colorbar(img2, ax=[ax[1]])
# %%
plt.plot(mfccs[0, :])
plt.plot(mfccs[1, :])
# %%
