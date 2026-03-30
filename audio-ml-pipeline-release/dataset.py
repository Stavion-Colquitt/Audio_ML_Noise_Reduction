"""
DNS Challenge Dataset Loader
==============================
PyTorch Dataset class that:
1. Loads clean speech + noise audio files
2. Mixes them at random SNR levels to create noisy training pairs
3. Converts to magnitude spectrograms for the U-Net
4. Returns (noisy_spec, clean_spec) pairs for training
"""

import os
import random
import glob
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Audio settings
SAMPLE_RATE = 16000   # 16kHz standard for speech ML
DURATION = 3.0        # seconds per training clip
N_FFT = 512           # FFT size
HOP_LENGTH = 128      # hop between frames
N_FRAMES = int(SAMPLE_RATE * DURATION / HOP_LENGTH) + 1

# SNR range for mixing clean + noise
SNR_MIN_DB = -5       # very noisy
SNR_MAX_DB = 20       # barely noisy


def load_audio(path, target_sr=SAMPLE_RATE, duration=DURATION):
    """Load audio file, resample, convert to mono, trim/pad to duration."""
    waveform, sr = torchaudio.load(path)

    # Resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)

    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Trim or pad to target duration
    target_len = int(target_sr * duration)
    if waveform.shape[1] > target_len:
        start = random.randint(0, waveform.shape[1] - target_len)
        waveform = waveform[:, start:start + target_len]
    else:
        pad = target_len - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, pad))

    return waveform.squeeze(0)  # [T]

def mix_at_snr(clean, noise, snr_db):
    """
    Mix clean speech with noise at a specific SNR level.
    SNR = 10 * log10(P_clean / P_noise)
    Higher SNR = cleaner audio, lower SNR = noisier audio.
    """
    # Calculate signal powers
    clean_power = clean.pow(2).mean()
    noise_power = noise.pow(2).mean()

    if noise_power < 1e-10:
        return clean, clean  # silence noise, return clean pair

    # Scale noise to achieve target SNR
    target_noise_power = clean_power / (10 ** (snr_db / 10))
    noise_scale = torch.sqrt(target_noise_power / noise_power)
    noisy = clean + noise * noise_scale

    # Normalize to prevent clipping
    max_val = noisy.abs().max()
    if max_val > 0:
        noisy = noisy / max_val
        clean = clean / max_val

    return noisy, clean


def to_spectrogram(waveform):
    """
    Convert waveform to magnitude spectrogram.
    This is the 2D representation the U-Net processes —
    time on x-axis, frequency on y-axis.
    Same thing you'd see in a DAW's spectrum analyzer.
    """
    window = torch.hann_window(N_FFT)
    stft = torch.stft(
        waveform,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        window=window,
        return_complex=True
    )
    magnitude = stft.abs()

    # Log scale — compresses dynamic range, better for neural nets
    # Same reason we use dBFS in audio — log scale matches perception
    log_magnitude = torch.log1p(magnitude)

    return log_magnitude.unsqueeze(0)  # [1, F, T]

class DNSDataset(Dataset):
    """
    DNS Challenge training dataset.
    Dynamically mixes clean speech + noise at random SNR during loading.
    This data augmentation means every epoch sees different noise mixes —
    much more diverse training than fixed noisy files.
    """
    def __init__(self, clean_dir, noise_dir, max_files=None):
        # Find all audio files recursively
        self.clean_files = glob.glob(
            os.path.join(clean_dir, "**", "*.wav"), recursive=True
        ) + glob.glob(
            os.path.join(clean_dir, "**", "*.flac"), recursive=True
        )

        self.noise_files = glob.glob(
            os.path.join(noise_dir, "**", "*.wav"), recursive=True
        ) + glob.glob(
            os.path.join(noise_dir, "**", "*.flac"), recursive=True
        )

        if max_files:
            self.clean_files = self.clean_files[:max_files]

        print(f"Dataset: {len(self.clean_files)} clean files, {len(self.noise_files)} noise files")

        if len(self.clean_files) == 0:
            raise ValueError(f"No clean audio files found in {clean_dir}")
        if len(self.noise_files) == 0:
            raise ValueError(f"No noise files found in {noise_dir}")

    def __len__(self):
        return len(self.clean_files)

    def __getitem__(self, idx):
        # Load clean speech
        clean = load_audio(self.clean_files[idx])

        # Load random noise clip
        noise_path = random.choice(self.noise_files)
        noise = load_audio(noise_path)

        # Mix at random SNR
        snr = random.uniform(SNR_MIN_DB, SNR_MAX_DB)
        noisy, clean = mix_at_snr(clean, noise, snr)

        # Convert to spectrograms
        noisy_spec = to_spectrogram(noisy)
        clean_spec = to_spectrogram(clean)

        return noisy_spec, clean_spec


def get_dataloaders(clean_dir, noise_dir, batch_size=16, val_split=0.1):
    """Create train and validation dataloaders."""
    dataset = DNSDataset(clean_dir, noise_dir)

    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size

    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Training samples: {train_size} | Validation samples: {val_size}")
    return train_loader, val_loader

