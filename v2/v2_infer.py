"""
v2 model stage, shared by the live pipeline and the evaluators so they can never drift apart.

Matches training exactly: torch.stft(center=True) == reflect-pad n_fft/2, Hann, n_fft 512 / hop 128;
model input log1p|STFT|; output = mask in [0, 1] applied to the LINEAR complex STFT; no per-window rescaling.
"""
import os
import numpy as np
import onnxruntime as ort

N_FFT, HOP = 512, 128
MASK_FLOOR = 0.05          # never remove more than ~26 dB in any bin (keeps a little air on consonants)
_WIN = np.hanning(N_FFT + 1)[:-1].astype(np.float64)       # periodic Hann == torch.hann_window
ONNX_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "denoiser_v2.onnx")
_sess = None


def session(path=ONNX_PATH):
    global _sess
    if _sess is None:
        _sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return _sess


def stft(x):
    x = np.pad(x.astype(np.float64), N_FFT // 2, mode="reflect")
    n = 1 + (len(x) - N_FFT) // HOP
    frames = np.stack([x[i * HOP:i * HOP + N_FFT] * _WIN for i in range(n)])
    return np.fft.rfft(frames, n=N_FFT, axis=1)               # [T, F] complex


def istft(S, length):
    fr = np.fft.irfft(S, n=N_FFT, axis=1)
    L = (len(fr) - 1) * HOP + N_FFT
    out, nrm = np.zeros(L), np.zeros(L)
    for i, f in enumerate(fr):
        out[i * HOP:i * HOP + N_FFT] += f * _WIN; nrm[i * HOP:i * HOP + N_FFT] += _WIN ** 2
    out = out / np.where(nrm > 1e-8, nrm, 1.0)
    return out[N_FFT // 2:N_FFT // 2 + length].astype(np.float32)


def run_model(chunk, floor=MASK_FLOOR):
    """One analysis window (normally 3200 samples = 200 ms) -> enhanced samples of the same length."""
    S = stft(chunk)
    feat = np.log1p(np.abs(S)).T.astype(np.float32)[None, None]          # [1, 1, F, T]
    mask = session().run(None, {"noisy_spectrogram": feat})[0][0, 0].T    # [T, F]
    return istft(np.maximum(mask, floor) * S, len(chunk))
