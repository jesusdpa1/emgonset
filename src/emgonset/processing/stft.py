from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torchaudio.transforms as T

from ..utils.internals import public_api


class BaseSTFT(ABC):
    """Base class for all STFT transform methods"""

    def __init__(self):
        self.fs: Optional[float] = None
        self.is_initialized: bool = False

    def initialize(self, fs: float) -> None:
        """Initialize STFT with sampling frequency"""
        self.fs = fs
        self.is_initialized = True

    @abstractmethod
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply STFT to signal"""
        pass


@public_api
class TorchAudioSTFT(BaseSTFT):
    """
    STFT implementation using torchaudio.transforms.Spectrogram

    This implementation uses the torchaudio library's spectrogram transform
    which internally uses STFT.
    """

    def __init__(
        self,
        n_fft: int = 512,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        window_fn: callable = torch.hann_window,
        power: Optional[float] = 2.0,
        normalized: bool = False,
        center: bool = True,
        pad_mode: str = "reflect",
        return_complex: bool = False,
    ):
        """
        Initialize TorchAudioSTFT

        Args:
            n_fft: Size of FFT
            hop_length: Hop length between frames. Default: n_fft // 4
            win_length: Window length. Default: n_fft
            window_fn: Window function
            power: Power of the magnitude. None for complex STFT, 1.0 for magnitude, 2.0 for power
            normalized: Whether to normalize the STFT
            center: Whether to pad the signal on both sides
            pad_mode: Padding mode for centered signal
            return_complex: Whether to return complex tensor (overrides power=None)
        """
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length if hop_length is not None else n_fft // 4
        self.win_length = win_length if win_length is not None else n_fft
        self.window_fn = window_fn
        self.power = None if return_complex else power
        self.normalized = normalized
        self.center = center
        self.pad_mode = pad_mode
        self.spectrogram = None  # Will be initialized in initialize()

    def initialize(self, fs: float) -> None:
        """Initialize STFT with sampling frequency"""
        super().initialize(fs)

        # Initialize the spectrogram transform
        self.spectrogram = T.Spectrogram(
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            pad_mode=self.pad_mode,
            window_fn=self.window_fn,
            power=self.power,
            normalized=self.normalized,
            center=self.center,
            onesided=True,
        )

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply STFT to signal

        Args:
            tensor: PyTorch tensor of shape [channels, samples]

        Returns:
            STFT result tensor of shape [channels, frequencies, frames]
        """
        if not self.is_initialized:
            raise RuntimeError("STFT not initialized. Call initialize(fs) first.")

        result = []

        # Process each channel
        for ch in range(tensor.shape[0]):
            # Apply STFT to this channel
            stft_result = self.spectrogram(tensor[ch])
            result.append(stft_result)

        # Stack results from all channels
        return torch.stack(result)


@public_api
class MelSpectrogram(BaseSTFT):
    """
    Mel Spectrogram transform using torchaudio

    This transforms the signal into a mel spectrogram using torchaudio.
    """

    def __init__(
        self,
        n_fft: int = 512,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        window_fn: callable = torch.hann_window,
        n_mels: int = 64,
        center: bool = True,
        pad_mode: str = "reflect",
        norm: Optional[str] = "slaney",
        mel_scale: str = "htk",
        log_mel: bool = False,
    ):
        """
        Initialize MelSpectrogram

        Args:
            n_fft: Size of FFT
            hop_length: Hop length between frames. Default: n_fft // 4
            win_length: Window length. Default: n_fft
            window_fn: Window function
            n_mels: Number of mel filterbanks
            center: Whether to pad the signal on both sides
            pad_mode: Padding mode for centered signal
            norm: Normalization mode for mel filterbanks
            mel_scale: Scale to use for mel conversion
            log_mel: Whether to apply log to mel spectrograms
        """
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length if hop_length is not None else n_fft // 4
        self.win_length = win_length if win_length is not None else n_fft
        self.window_fn = window_fn
        self.n_mels = n_mels
        self.center = center
        self.pad_mode = pad_mode
        self.norm = norm
        self.mel_scale = mel_scale
        self.log_mel = log_mel
        self.mel_spectrogram = None  # Will be initialized in initialize()

    def initialize(self, fs: float) -> None:
        """Initialize MelSpectrogram with sampling frequency"""
        super().initialize(fs)

        # Initialize the mel spectrogram transform
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=self.fs,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            center=self.center,
            pad_mode=self.pad_mode,
            window_fn=self.window_fn,
            n_mels=self.n_mels,
            norm=self.norm,
            mel_scale=self.mel_scale,
        )

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply MelSpectrogram to signal

        Args:
            tensor: PyTorch tensor of shape [channels, samples]

        Returns:
            Mel spectrogram tensor of shape [channels, n_mels, frames]
        """
        if not self.is_initialized:
            raise RuntimeError(
                "MelSpectrogram not initialized. Call initialize(fs) first."
            )

        result = []

        # Process each channel
        for ch in range(tensor.shape[0]):
            # Apply mel spectrogram to this channel
            mel_spec = self.mel_spectrogram(tensor[ch])

            # Apply log if requested
            if self.log_mel:
                # Add small constant to avoid log(0)
                mel_spec = torch.log(mel_spec + 1e-10)

            result.append(mel_spec)

        # Stack results from all channels
        return torch.stack(result)


@public_api
class MFCC(BaseSTFT):
    """
    MFCC (Mel-Frequency Cepstral Coefficients) transform using torchaudio

    This transforms the signal into MFCCs using torchaudio.
    """

    def __init__(
        self,
        n_fft: int = 512,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        window_fn: callable = torch.hann_window,
        n_mels: int = 40,
        n_mfcc: int = 13,
        center: bool = True,
        pad_mode: str = "reflect",
        norm: Optional[str] = "slaney",
        mel_scale: str = "htk",
        dct_norm: str = "ortho",
    ):
        """
        Initialize MFCC

        Args:
            n_fft: Size of FFT
            hop_length: Hop length between frames. Default: n_fft // 4
            win_length: Window length. Default: n_fft
            window_fn: Window function
            n_mels: Number of mel filterbanks
            n_mfcc: Number of MFCC coefficients to return
            center: Whether to pad the signal on both sides
            pad_mode: Padding mode for centered signal
            norm: Normalization mode for mel filterbanks
            mel_scale: Scale to use for mel conversion
            dct_norm: Normalization for DCT-II
        """
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length if hop_length is not None else n_fft // 4
        self.win_length = win_length if win_length is not None else n_fft
        self.window_fn = window_fn
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.center = center
        self.pad_mode = pad_mode
        self.norm = norm
        self.mel_scale = mel_scale
        self.dct_norm = dct_norm
        self.mfcc_transform = None  # Will be initialized in initialize()

    def initialize(self, fs: float) -> None:
        """Initialize MFCC with sampling frequency"""
        super().initialize(fs)

        # Initialize the MFCC transform
        self.mfcc_transform = T.MFCC(
            sample_rate=self.fs,
            n_mfcc=self.n_mfcc,
            melkwargs={
                "n_fft": self.n_fft,
                "n_mels": self.n_mels,
                "hop_length": self.hop_length,
                "win_length": self.win_length,
                "window_fn": self.window_fn,
                "center": self.center,
                "pad_mode": self.pad_mode,
                "norm": self.norm,
                "mel_scale": self.mel_scale,
            },
            norm=self.dct_norm,
        )

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply MFCC to signal

        Args:
            tensor: PyTorch tensor of shape [channels, samples]

        Returns:
            MFCC tensor of shape [channels, n_mfcc, frames]
        """
        if not self.is_initialized:
            raise RuntimeError("MFCC not initialized. Call initialize(fs) first.")

        result = []

        # Process each channel
        for ch in range(tensor.shape[0]):
            # Apply MFCC to this channel
            mfcc = self.mfcc_transform(tensor[ch])
            result.append(mfcc)

        # Stack results from all channels
        return torch.stack(result)


# Factory functions


@public_api
def create_stft(
    n_fft: int = 512,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    window_fn: callable = torch.hann_window,
    power: Optional[float] = 2.0,
    normalized: bool = False,
    center: bool = True,
    pad_mode: str = "reflect",
    return_complex: bool = False,
) -> TorchAudioSTFT:
    """
    Create a STFT transform using torchaudio

    Args:
        n_fft: Size of FFT
        hop_length: Hop length between frames. Default: n_fft // 4
        win_length: Window length. Default: n_fft
        window_fn: Window function
        power: Power of the magnitude. None for complex STFT, 1.0 for magnitude, 2.0 for power
        normalized: Whether to normalize the STFT
        center: Whether to pad the signal on both sides
        pad_mode: Padding mode for centered signal
        return_complex: Whether to return complex tensor (overrides power=None)

    Returns:
        A configured TorchAudioSTFT object
    """
    return TorchAudioSTFT(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window_fn=window_fn,
        power=power,
        normalized=normalized,
        center=center,
        pad_mode=pad_mode,
        return_complex=return_complex,
    )


@public_api
def create_mel_spectrogram(
    n_fft: int = 512,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    window_fn: callable = torch.hann_window,
    n_mels: int = 64,
    center: bool = True,
    pad_mode: str = "reflect",
    norm: Optional[str] = "slaney",
    mel_scale: str = "htk",
    log_mel: bool = False,
) -> MelSpectrogram:
    """
    Create a Mel Spectrogram transform

    Args:
        n_fft: Size of FFT
        hop_length: Hop length between frames. Default: n_fft // 4
        win_length: Window length. Default: n_fft
        window_fn: Window function
        n_mels: Number of mel filterbanks
        center: Whether to pad the signal on both sides
        pad_mode: Padding mode for centered signal
        norm: Normalization mode for mel filterbanks
        mel_scale: Scale to use for mel conversion
        log_mel: Whether to apply log to mel spectrograms

    Returns:
        A configured MelSpectrogram object
    """
    return MelSpectrogram(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window_fn=window_fn,
        n_mels=n_mels,
        center=center,
        pad_mode=pad_mode,
        norm=norm,
        mel_scale=mel_scale,
        log_mel=log_mel,
    )


@public_api
def create_mfcc(
    n_fft: int = 512,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    window_fn: callable = torch.hann_window,
    n_mels: int = 40,
    n_mfcc: int = 13,
    center: bool = True,
    pad_mode: str = "reflect",
    norm: Optional[str] = "slaney",
    mel_scale: str = "htk",
    dct_norm: str = "ortho",
) -> MFCC:
    """
    Create an MFCC transform

    Args:
        n_fft: Size of FFT
        hop_length: Hop length between frames. Default: n_fft // 4
        win_length: Window length. Default: n_fft
        window_fn: Window function
        n_mels: Number of mel filterbanks
        n_mfcc: Number of MFCC coefficients to return
        center: Whether to pad the signal on both sides
        pad_mode: Padding mode for centered signal
        norm: Normalization mode for mel filterbanks
        mel_scale: Scale to use for mel conversion
        dct_norm: Normalization for DCT-II

    Returns:
        A configured MFCC object
    """
    return MFCC(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window_fn=window_fn,
        n_mels=n_mels,
        n_mfcc=n_mfcc,
        center=center,
        pad_mode=pad_mode,
        norm=norm,
        mel_scale=mel_scale,
        dct_norm=dct_norm,
    )
