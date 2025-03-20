# EMGOnset

A Python library for EMG onset/offset detection using machine learning approaches.

## Overview

EMGOnset is a comprehensive toolkit for processing electromyography (EMG) signals and detecting muscle activation onset/offset points using advanced machine learning techniques. This library provides tools for:

- Loading and preprocessing EMG data from various formats
- Filtering and feature extraction from EMG signals
- Implementation of various onset detection algorithms
- Machine learning models for robust onset/offset detection
- Evaluation and visualization of detection performance

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