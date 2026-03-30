"""
U-Net Denoising Model Trainer
================================
Trains the denoising model on DNS Challenge data.

Loss function: SI-SNR (Scale-Invariant Signal-to-Noise Ratio)
Higher SI-SNR = cleaner output.

Usage:
    python train.py
"""

import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from model import UNetDenoiser, count_parameters
from dataset import get_dataloaders

BASE_DIR       = os.path.dirname(__file__)
DATA_DIR       = os.path.join(BASE_DIR, "data")
CLEAN_DIR      = os.path.join(DATA_DIR, "clean_fullband")
NOISE_DIR      = os.path.join(DATA_DIR, "noise_fullband")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Training config
EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
BASE_CHANNELS = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def si_snr_loss(predicted, target, eps=1e-8):
    """SI-SNR loss — standard metric for speech enhancement."""
    predicted = predicted.view(predicted.shape[0], -1)
    target = target.view(target.shape[0], -1)
    predicted = predicted - predicted.mean(dim=1, keepdim=True)
    target = target - target.mean(dim=1, keepdim=True)
    dot = (predicted * target).sum(dim=1, keepdim=True)
    target_power = (target * target).sum(dim=1, keepdim=True) + eps
    proj = (dot / target_power) * target
    noise = predicted - proj
    signal_power = (proj * proj).sum(dim=1)
    noise_power = (noise * noise).sum(dim=1) + eps
    si_snr = 10 * torch.log10(signal_power / noise_power + eps)
    return -si_snr.mean()


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch_idx, (noisy, clean) in enumerate(loader):
        noisy, clean = noisy.to(device), clean.to(device)
        optimizer.zero_grad()
        predicted = model(noisy)
        loss = si_snr_loss(predicted, clean)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        if batch_idx % 50 == 0:
            print(f"  Batch {batch_idx}/{len(loader)} | Loss: {loss.item():.4f}")
    return total_loss / len(loader)


def validate(model, loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for noisy, clean in loader:
            noisy, clean = noisy.to(device), clean.to(device)
            predicted = model(noisy)
            loss = si_snr_loss(predicted, clean)
            total_loss += loss.item()
    return total_loss / len(loader)

def main():
    print("=" * 60)
    print("U-Net Speech Denoising Trainer")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

    print("\nLoading dataset...")
    train_loader, val_loader = get_dataloaders(CLEAN_DIR, NOISE_DIR, batch_size=BATCH_SIZE)

    print("\nBuilding model...")
    model = UNetDenoiser(base_ch=BASE_CHANNELS).to(DEVICE)
    print(f"Parameters: {count_parameters(model):,}")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    best_val_loss = float('inf')
    print(f"\nStarting training for {EPOCHS} epochs...")
    print("=" * 60)

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        train_loss = train_epoch(model, train_loader, optimizer, DEVICE)
        val_loss = validate(model, val_loader, DEVICE)
        scheduler.step()
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"Saved best model (val loss: {val_loss:.4f})")

        if epoch % 10 == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch}.pth")
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, checkpoint_path)

    print("\n" + "=" * 60)
    print(f"Training complete. Best val loss: {best_val_loss:.4f}")
    print(f"Best model: {os.path.join(CHECKPOINT_DIR, 'best_model.pth')}")
    print("Next: run export_model.py to convert to ONNX")


if __name__ == "__main__":
    main()
