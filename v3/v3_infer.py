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
ONNX_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "denoiser_v3.onnx")
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


COMP = 0.3                 # must match train.COMP
# PHASE = "model": full complex mask (v3 as designed).  PHASE = "noisy": use only |mask| and keep the noisy phase,
# which scored better on VoiceBank+DEMAND (see README). Set with the env var V3_PHASE.
PHASE = os.environ.get("V3_PHASE", "model")


def features(S):           # complex [T, F] -> float32 [1, 3, F, T]  (== train.features)
    mag = np.abs(S); unit = S / (mag + 1e-8); c = (mag + 1e-8) ** COMP
    return np.stack([np.log1p(mag), c * unit.real, c * unit.imag])[None].transpose(0, 1, 3, 2).astype(np.float32)


def run_model(chunk, floor=MASK_FLOOR):
    """One analysis window (normally 3200 samples = 200 ms) -> enhanced samples of the same length.
    v3: complex ratio mask; the floor limits how far a bin's MAGNITUDE can be pulled down, phase untouched."""
    S = stft(chunk)
    m = session().run(None, {"noisy_spectrogram": features(S)})[0][0]       # [2, F, T]
    M = (m[0] + 1j * m[1]).T                                                # [T, F]
    if PHASE == "noisy":
        M = np.abs(M)                                                       # magnitude only, noisy phase
    if floor > 0:
        a = np.abs(M); M = M * (np.maximum(a, floor) / np.maximum(a, 1e-8))
    return istft(M * S, len(chunk))
