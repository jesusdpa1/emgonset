"""
Author: jpenalozaa
Description: testing mels stft pytorch
"""

# %%
from pathlib import Path

# Plotting
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Data loading and manipulation

import numpy as np
import quantities as pq

from neo.io import TdtIO
from emgonset.preprocessing import filters
# %%
# define home (Drive were data is located)
home = Path.home()
# define the path to the data file
data_path = r"E:\jpenalozaa\tdt_data\test_long-recording\testSubject-231101_test_long-recording.tev"
# build the path to the data file
working_path = home.joinpath(data_path)
print(working_path)
# %%
# load the data using CedIO
reader = TdtIO(dirname=working_path)
data = reader.read()
emg_data = data[0].segments[0].analogsignals[6]
fs = float(emg_data.sampling_rate)
# %%
start_time = 10 * pq.s
end_time = 20 * pq.s
#%%
sample_data = emg_data.time_slice(start_time, end_time).as_array()
#%%
plt.plot(sample_data[:,1])
# %%
emg_r = sample_data[:, 0]
emg_r_notch = filters.notch_filter(emg_r, fs = fs) 
emg_r_bp = filters.butter_bandpass_filter(emg_r_notch, lowcut=10, highcut=2000, fs=fs, order=150)
# %%
plt.plot(emg_r_bp)
# %%
import torch
from torchaudio.transforms import Spectrogram, MelSpectrogram
from torchaudio.functional import resample
import librosa
#%%
emg_r_bp_copy = emg_r_bp.copy()
emg_r_Tensor = torch.from_numpy(emg_r_bp_copy)
emg_r_Tensor_resample = resample(emg_r_Tensor, int(fs), fs//2)
#%%
# Define transform
spectrogram = Spectrogram(n_fft=512, hop_length=64)

# Perform transform
spec = spectrogram(emg_r_Tensor_resample)
# %%
def plot_spectrogram(specgram, title=None, ylabel="freq_bin", ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto", interpolation="nearest")
# %%
plot_spectrogram(spec)
# %%
n_fft = 512
win_length = None
hop_length = 64
n_mels = 128

mel_spectrogram = MelSpectrogram(
    sample_rate=int(fs),
    n_fft=n_fft,
    win_length=win_length,
    hop_length=hop_length,
    center=True,
    pad_mode="reflect",
    power=2.0,
    norm="slaney",
    n_mels=n_mels,
    mel_scale="htk",
)
double_test = emg_r_Tensor_resample.float()
#%%
melspec = mel_spectrogram(double_test)
# %%
plot_spectrogram(melspec, title="MelSpectrogram - torchaudio", ylabel="mel freq")
# %%
# calculate the difference
diff = np.diff(spec.T, axis=0)
# keep only the positive differences
pos_diff = np.maximum(0, diff)
# sum everything to get the spectral flux
sf = np.sum(pos_diff, axis=1)
# %%
plot_spectrogram(melspec, title="MelSpectrogram - torchaudio", ylabel="mel freq")
#%%
plt.plot(sf)
# %%
