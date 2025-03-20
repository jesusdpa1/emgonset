# EMGOnset

A Python library for EMG onset/offset detection using machine learning approaches.

## Overview

EMGOnset is a comprehensive toolkit for processing electromyography (EMG) signals and detecting muscle activation onset/offset points using advanced machine learning techniques. This library provides tools for:

- Loading and preprocessing EMG data from various formats
- Filtering and feature extraction from EMG signals
- Implementation of various onset detection algorithms
- Machine learning models for robust onset/offset detection
- Evaluation and visualization of detection performance

## Installation

```bash
pip install emgonset
```

Or install from source:

```bash
git clone https://github.com/yourusername/emgonset.git
cd emgonset
pip install -e .
```

## Features

### Data Loading and Preprocessing

- Support for loading EMG data from Parquet files with metadata
- Windowing and segmentation of continuous EMG signals
- Efficient lazy loading for large datasets
- PyTorch DataLoader integration for ML model training

### Signal Processing

- Comprehensive filtering options (bandpass, notch, lowpass, highpass)
- Multiple rectification methods (absolute value, square, Hilbert)
- Teager-Kaiser Energy Operator implementations (TKEO, TKEO2, MTKEO)
- Composable transform pipeline

### Onset Detection

- Traditional threshold-based algorithms
- Feature-based machine learning models
- Deep learning approaches for adaptive detection
- Ensemble methods for improved robustness

## Quick Start

Here's a simple example of how to use EMGOnset:

```python
import torch
from emgonset.processing.filters import create_bandpass_filter, create_notch_filter, create_emg_filter
from emgonset.processing.rectifiers import create_abs_rectifier
from emgonset.processing.tkeo import create_tkeo
from emgonset.processing.transforms import EMGTransformCompose
from emgonset.utils.io import create_emg_dataloader

# Create a processing pipeline
transform = EMGTransformCompose([
    create_emg_filter([
        create_notch_filter(notch_freq=60),
        create_bandpass_filter(low_cutoff=20, high_cutoff=2000)
    ]),
    create_abs_rectifier(),
    create_tkeo()
])

# Load EMG data
dataloader, fs = create_emg_dataloader(
    data_dir="path/to/data",
    window_size_sec=1.0,
    window_stride_sec=0.5,
    batch_size=32,
    transform=transform,
    num_workers=0,
    shuffle=False
)

# Simple visualization
import matplotlib.pyplot as plt
import numpy as np

# Get a batch of data
batch = next(iter(dataloader))
sample = batch[0]  # First sample in batch

# Plot original and processed signals
plt.figure(figsize=(12, 6))
time = np.linspace(0, sample.shape[1]/fs, sample.shape[1])
for i in range(sample.shape[0]):
    plt.plot(time, sample[i].numpy(), label=f'Channel {i+1}')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.show()
```

## Project Structure

```
emgonset/
├── processing/
│   ├── filters.py       # Signal filtering
│   ├── rectifiers.py    # Signal rectification
│   ├── tkeo.py          # Teager-Kaiser Energy Operators
│   ├── transforms.py    # Transform compositions
│   └── dataset.py       # Data loading and windowing
├── detection/
│   ├── threshold.py     # Threshold-based algorithms
│   ├── features.py      # Feature extraction
│   ├── ml_models.py     # Machine learning models
│   └── evaluation.py    # Performance metrics
├── utils/
│   ├── io.py            # I/O utilities
│   └── internals.py     # Internal utilities
└── visualization/
    └── general_plots.py         # Visualization tools
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.