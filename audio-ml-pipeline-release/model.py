"""
U-Net Denoising Model Architecture
===================================
A lightweight U-Net designed for real-time speech enhancement.

Architecture:
- Input: magnitude spectrogram (1 x F x T)
- Encoder: 4 blocks, each doubles channels and halves time/freq dims
- Bottleneck: deepest compressed representation
- Decoder: 4 blocks with skip connections from encoder
- Output: mask applied to input spectrogram

Skip connections preserve fine detail that would otherwise be lost
during compression — same principle as skip connections in ResNet.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Basic conv block: Conv2d -> BatchNorm -> LeakyReLU"""
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class UpBlock(nn.Module):
    """Decoder block: upsample + concat skip connection + conv"""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # Pad if sizes don't match exactly
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:])
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetDenoiser(nn.Module):
    """
    Lightweight U-Net for speech denoising.

    Input:  magnitude spectrogram [B, 1, F, T]
    Output: clean magnitude spectrogram [B, 1, F, T]

    The model learns a soft mask to apply to the noisy input.
    This is called masking-based enhancement — more stable than
    directly predicting the clean spectrogram.
    """
    def __init__(self, base_ch=32):
        super().__init__()

        # Encoder — progressively compress
        self.enc1 = ConvBlock(1, base_ch)
        self.enc2 = ConvBlock(base_ch, base_ch * 2, stride=2)
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4, stride=2)
        self.enc4 = ConvBlock(base_ch * 4, base_ch * 8, stride=2)

        # Bottleneck — deepest representation
        self.bottleneck = ConvBlock(base_ch * 8, base_ch * 16, stride=2)

        # Decoder — progressively reconstruct with skip connections
        self.dec4 = UpBlock(base_ch * 16, base_ch * 8, base_ch * 8)
        self.dec3 = UpBlock(base_ch * 8, base_ch * 4, base_ch * 4)
        self.dec2 = UpBlock(base_ch * 4, base_ch * 2, base_ch * 2)
        self.dec1 = UpBlock(base_ch * 2, base_ch, base_ch)

        # Output head — predict a mask between 0 and 1
        self.output = nn.Sequential(
            nn.Conv2d(base_ch, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Encode
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # Bottleneck
        b = self.bottleneck(e4)

        # Decode with skip connections
        d4 = self.dec4(b, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)

        # Generate mask and apply to input
        mask = self.output(d1)
        return x * mask

def count_parameters(model):
    """Count trainable parameters — useful for estimating model size."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick architecture test
    model = UNetDenoiser(base_ch=32)
    print(f"Model parameters: {count_parameters(model):,}")

    # Test forward pass with dummy input
    # Simulating a spectrogram: batch=2, channels=1, freq=256, time=128
    dummy = torch.randn(2, 1, 256, 128)
    output = model(dummy)
    print(f"Input shape:  {dummy.shape}")
    print(f"Output shape: {output.shape}")
    print("Architecture test passed.")
