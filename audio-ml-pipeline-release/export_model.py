"""
Export Trained Model to ONNX
==============================
After training, converts the PyTorch model to ONNX format
for fast inference in the real-time pipeline.

ONNX is a universal format that runs on any platform.
INT8 quantization reduces model size and speeds up inference
without significant quality loss.

Usage:
    python export_model.py
"""

import torch
import torch.onnx
import onnx
import onnxruntime as ort
import numpy as np
import os

from model import UNetDenoiser
from dataset import N_FFT, HOP_LENGTH, SAMPLE_RATE

BASE_DIR        = os.path.dirname(__file__)
CHECKPOINT_PATH = os.path.join(BASE_DIR, "checkpoints", "best_model.pth")
ONNX_PATH       = os.path.join(BASE_DIR, "denoiser.onnx")
DEVICE = torch.device("cpu")  # Export on CPU for compatibility

# Spectrogram dimensions
FREQ_BINS = N_FFT // 2 + 1
TIME_FRAMES = 94  # ~3 seconds at 16kHz with hop=128

def export_to_onnx():
    print("Loading trained model...")
    model = UNetDenoiser(base_ch=32)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Model loaded from epoch {checkpoint['epoch']}")

    # Create dummy input matching inference input shape
    dummy_input = torch.randn(1, 1, FREQ_BINS, TIME_FRAMES)

    print("Exporting to ONNX...")
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['noisy_spectrogram'],
        output_names=['clean_spectrogram'],
        dynamic_axes={
            'noisy_spectrogram': {3: 'time_frames'},
            'clean_spectrogram': {3: 'time_frames'}
        }
    )
    print(f"ONNX model saved to: {ONNX_PATH}")

    print("Verifying ONNX model...")
    onnx_model = onnx.load(ONNX_PATH)
    onnx.checker.check_model(onnx_model)
    print("ONNX model is valid.")

    print("Testing ONNX inference...")
    ort_session = ort.InferenceSession(ONNX_PATH)
    dummy_np = dummy_input.numpy()
    outputs = ort_session.run(None, {'noisy_spectrogram': dummy_np})
    print(f"ONNX inference output shape: {outputs[0].shape}")
    print("Export complete. Model ready for real-time pipeline.")

if __name__ == "__main__":
    export_to_onnx()
