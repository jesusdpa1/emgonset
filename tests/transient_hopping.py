# %%
import numpy as np
import matplotlib.pyplot as plt

# Generate a sample vibration signal with a transient
t = np.linspace(0, 1, 1000)
x = (
    np.sin(2 * np.pi * 10 * t)
    + 0.5 * np.sin(2 * np.pi * 20 * t)
    + 0.1 * np.sin(2 * np.pi * 30 * t)
)
x[500:550] += 2 * np.sin(2 * np.pi * 50 * t[500:550])


# Define the transient hopping function
def transient_hopping(x, window_size, threshold):
    hopping_output = []
    for i in range(len(x) - window_size + 1):
        window = x[i : i + window_size]
        feature = np.sum(np.abs(window))
        if feature > threshold:
            hopping_output.append(1)
        else:
            hopping_output.append(0)
    return hopping_output


# Apply transient hopping to the signal
window_size = 20
threshold = 10
hopping_output = transient_hopping(x, window_size, threshold)

# Plot the results
plt.plot(t, x)
# %%
plt.plot(hopping_output)
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from scipy.stats import kurtosis


class TransientHopping:
    def __init__(self, window_size, threshold, fs):
        self.window_size = window_size
        self.threshold = threshold
        self.fs = fs

    def butter_bandpass(self, lowcut, highcut, order=5):
        nyq = 0.5 * self.fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype="band")
        return b, a

    def butter_bandpass_filter(self, data, lowcut, highcut, order=5):
        b, a = self.butter_bandpass(lowcut, highcut, order=order)
        y = lfilter(b, a, data)
        return y

    def kurtosis_feature(self, x):
        return kurtosis(x)

    def transient_hopping(self, x):
        hopping_output = np.zeros(len(x))
        for i in range(len(x) - self.window_size + 1):
            window = x[i : i + self.window_size]
            feature = self.kurtosis_feature(window)
            if feature > self.threshold:
                hopping_output[i + self.window_size // 2] = 1
        return hopping_output

    def detect_transients(self, x, lowcut, highcut):
        filtered_x = self.butter_bandpass_filter(x, lowcut, highcut)
        hopping_output = self.transient_hopping(filtered_x)
        return hopping_output


# Generate a sample vibration signal with a transient
t = np.linspace(0, 1, 1000)
x = (
    np.sin(2 * np.pi * 10 * t)
    + 0.5 * np.sin(2 * np.pi * 20 * t)
    + 0.1 * np.sin(2 * np.pi * 30 * t)
)
x[500:550] += 2 * np.sin(2 * np.pi * 50 * t[500:550])

# Create a TransientHopping object
transient_hopper = TransientHopping(window_size=50, threshold=10, fs=1000)

# Detect transients in the signal
lowcut = 1
highcut = 900
hopping_output = transient_hopper.detect_transients(x, lowcut, highcut)

# Plot the results
plt.plot(t, x)
plt.plot(t, hopping_output)
plt.show()
# %%
