"""
ML Denoiser Pipeline
====================
Real-time speech enhancement pipeline. Captures microphone input,
runs it through a trained U-Net denoising model via ONNX inference,
and routes the clean output to a virtual audio cable so any application
(Teams, Zoom, DAW) sees a clean microphone.

Signal flow:
    Microphone → STFT → U-Net (ONNX) → iSTFT → OLA → EQ → Soft Gate → Virtual Cable

Architecture decisions:
- Native 16kHz throughout — matches the model's training sample rate,
  eliminates resampling and the boundary artifacts it introduces.
- Overlap-Add (OLA) synthesis at 50% overlap — Hann windowed, prevents
  seam artifacts between consecutive inference windows.
- Soft noise gate with hold — attenuates only true silence, preserves
  consonants that have low RMS but are still speech.
- Parallel bandpass EQ (100Hz–1kHz) — compensates for the mild presence
  attenuation caused by noisy-phase ISTFT reconstruction.
"""

import threading
import queue as qmod
import time
import os
import numpy as np
import sounddevice as sd
import onnxruntime as ort  # type: ignore
from scipy.signal import butter, sosfilt, sosfilt_zi

# ── Devices ───────────────────────────────────────────────────────────────────
# Run list_devices.py to find the correct indexes for your system.
MIC_INDEX   = 0    # index of your microphone input device
CABLE_INDEX = 1    # index of your virtual cable input device (e.g. VB-Audio CABLE Input)

# ── Audio ─────────────────────────────────────────────────────────────────────
SAMPLE_RATE = 16000   # native model rate — no resampling needed
CHUNK_SIZE  = 256
CHANNELS    = 1
ONNX_PATH   = os.path.join(os.path.dirname(__file__), "denoiser.onnx")

# ── STFT ──────────────────────────────────────────────────────────────────────
N_FFT      = 512
HOP_LENGTH = 128

# ── OLA parameters ────────────────────────────────────────────────────────────
WINDOW_SAMP = 3200   # 200ms @ 16kHz (22 STFT frames)
HOP_SAMP    = 1600   # 100ms hop — 50% overlap

# ── Spectral floor ────────────────────────────────────────────────────────────
# Minimum fraction of noisy energy kept per bin.
# Prevents over-suppression of unvoiced consonants (s, f, sh, th).
SPECTRAL_FLOOR = 0.12

# ── Soft noise gate ───────────────────────────────────────────────────────────
# The model handles most noise suppression — the gate only catches true silence.
# GATE_FLOOR is intentionally low so consonants are never mistaken for noise.
GATE_FLOOR   = 0.0003   # ~-70dBFS — below this: attenuated
GATE_CEIL    = 0.004    # ~-48dBFS — above this: full pass
GATE_ATTACK  = 0.30
GATE_RELEASE = 0.04
GATE_HOLD    = 8        # hops to hold open after signal drops (~800ms)
_gate_gain    = 0.0
_gate_hold_ct = 0

# ── Gain staging ─────────────────────────────────────────────────────────────
INPUT_GAIN  = 0.75   # pre-model headroom
OUTPUT_GAIN = 3.2    # post-gate output level

# ── Post-processing EQ ───────────────────────────────────────────────────────
# Parallel bandpass boost (100Hz–1kHz, +5dB).
# Restores voice body and formants attenuated by noisy-phase ISTFT.
_EQ_GAIN_LIN = 10 ** (5.0 / 20) - 1
_sos_lo   = butter(2,  100 / (SAMPLE_RATE / 2), btype='high', output='sos')
_sos_hi   = butter(2, 1000 / (SAMPLE_RATE / 2), btype='low',  output='sos')
_eq_zi_lo = sosfilt_zi(_sos_lo).astype(np.float64)
_eq_zi_hi = sosfilt_zi(_sos_hi).astype(np.float64)

# ── Hann OLA synthesis window ─────────────────────────────────────────────────
_hann_syn  = np.hanning(WINDOW_SAMP).astype(np.float32)
_ola_accum = np.zeros(WINDOW_SAMP, dtype=np.float32)
_ola_norm  = np.zeros(WINDOW_SAMP, dtype=np.float32)

print("Loading model...")
session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
print("Model loaded.")

# Pre-warm OLA buffer so the first utterance is not subject to a startup ramp.
for _ in range(WINDOW_SAMP // HOP_SAMP - 1):
    _ola_norm  += _hann_syn
    _ola_accum[:-HOP_SAMP] = _ola_accum[HOP_SAMP:].copy()
    _ola_accum[-HOP_SAMP:] = 0.0
    _ola_norm[:-HOP_SAMP]  = _ola_norm[HOP_SAMP:].copy()
    _ola_norm[-HOP_SAMP:]  = 0.0

# ── DSP helpers ───────────────────────────────────────────────────────────────
def _stft(waveform):
    win      = np.hanning(N_FFT)
    n_frames = max(0, 1 + (len(waveform) - N_FFT) // HOP_LENGTH)
    frames   = np.stack([waveform[i*HOP_LENGTH:i*HOP_LENGTH+N_FFT] * win
                         for i in range(n_frames)], axis=0)
    spectra  = np.fft.rfft(frames, n=N_FFT, axis=1)
    return np.log1p(np.abs(spectra).T), spectra

def _istft(magnitude, phase_ref):
    reconstructed = magnitude.T * np.exp(1j * np.angle(phase_ref))
    frames        = np.fft.irfft(reconstructed, n=N_FFT, axis=1)
    win     = np.hanning(N_FFT)
    out_len = (frames.shape[0] - 1) * HOP_LENGTH + N_FFT
    output  = np.zeros(out_len, dtype=np.float64)
    norm    = np.zeros(out_len, dtype=np.float64)
    for i, frame in enumerate(frames):
        s = i * HOP_LENGTH
        output[s:s+N_FFT] += frame * win
        norm[s:s+N_FFT]   += win ** 2
    return (output / np.where(norm > 1e-8, norm, 1.0)).astype(np.float32)

def _run_model(chunk):
    log_mag, phase_ref = _stft(chunk)
    scale = float(np.percentile(log_mag, 99))
    if scale < 0.01:
        return np.zeros(len(chunk), dtype=np.float32)
    norm_log   = (log_mag / scale).astype(np.float32)
    clean_norm = session.run(
        None, {"noisy_spectrogram": norm_log[np.newaxis, np.newaxis]})[0][0, 0]
    clean_mag  = np.expm1(np.clip(clean_norm * scale, 0, None))
    clean_mag  = np.maximum(clean_mag, SPECTRAL_FLOOR * np.expm1(log_mag))
    out = _istft(clean_mag, phase_ref)
    return out[:len(chunk)] if len(out) >= len(chunk) \
           else np.pad(out, (0, len(chunk) - len(out)))

# ── Thread-safe buffers ───────────────────────────────────────────────────────
_in_q        = qmod.Queue()
_out_buf     = np.zeros(0, dtype=np.float32)
_out_lock    = threading.Lock()
_running     = True
_underruns   = 0
_MIN_STARTUP = HOP_SAMP * 3   # hold output until 300ms is buffered

def _processing_thread():
    global _out_buf, _ola_accum, _ola_norm, _gate_gain, _gate_hold_ct, _eq_zi_lo, _eq_zi_hi, _running
    pending = np.zeros(0, dtype=np.float32)

    while _running:
        chunks = []
        try:
            while True:
                chunks.append(_in_q.get_nowait())
        except qmod.Empty:
            pass

        if chunks:
            pending = np.concatenate([pending, np.concatenate(chunks) * INPUT_GAIN])

        while len(pending) >= WINDOW_SAMP:
            chunk   = pending[:WINDOW_SAMP]
            pending = pending[HOP_SAMP:]
            clean   = _run_model(chunk)

            # Hann OLA synthesis
            _ola_accum += clean * _hann_syn
            _ola_norm  += _hann_syn
            safe_norm   = np.where(_ola_norm[:HOP_SAMP] > 1e-6, _ola_norm[:HOP_SAMP], 1.0)
            out         = (_ola_accum[:HOP_SAMP] / safe_norm).astype(np.float32)
            _ola_accum[:-HOP_SAMP] = _ola_accum[HOP_SAMP:].copy()
            _ola_accum[-HOP_SAMP:] = 0.0
            _ola_norm[:-HOP_SAMP]  = _ola_norm[HOP_SAMP:].copy()
            _ola_norm[-HOP_SAMP:]  = 0.0

            # Parallel bandpass EQ — boosts 100Hz–1kHz voice body
            sig64 = out.astype(np.float64)
            hp, _eq_zi_lo  = sosfilt(_sos_lo, sig64, zi=_eq_zi_lo)
            band, _eq_zi_hi = sosfilt(_sos_hi, hp,   zi=_eq_zi_hi)
            out = out + band.astype(np.float32) * _EQ_GAIN_LIN

            # Soft noise gate with hold
            rms = float(np.sqrt(np.mean(out ** 2)))
            if rms <= GATE_FLOOR:
                target = 0.0
            elif rms >= GATE_CEIL:
                target = 1.0
            else:
                target = (rms - GATE_FLOOR) / (GATE_CEIL - GATE_FLOOR)
            if rms > GATE_FLOOR:
                _gate_hold_ct = GATE_HOLD
            elif _gate_hold_ct > 0:
                _gate_hold_ct -= 1
                target = max(target, 0.9)
            coeff      = GATE_ATTACK if target > _gate_gain else GATE_RELEASE
            _gate_gain = _gate_gain + coeff * (target - _gate_gain)

            with _out_lock:
                _out_buf = np.concatenate([_out_buf, out * _gate_gain * OUTPUT_GAIN])

        time.sleep(0)   # yield without Windows 15ms timer penalty


def callback(indata, outdata, frames, time, status):
    global _underruns
    if status:
        print(f"[Status] {status}")
    _in_q.put(indata[:, 0].copy())
    with _out_lock:
        buf_len = len(_out_buf)
        if buf_len < _MIN_STARTUP:
            outdata[:, 0] = 0.0
            return
        if buf_len >= frames:
            out = _out_buf[:frames].copy()
            globals()['_out_buf'] = _out_buf[frames:]
        else:
            out = np.zeros(frames, dtype=np.float32)
            if buf_len > 0:
                out[:buf_len] = _out_buf
                globals()['_out_buf'] = np.zeros(0, dtype=np.float32)
            _underruns += 1
    outdata[:, 0] = np.clip(out, -1.0, 1.0)

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    stft_frames = 1 + (WINDOW_SAMP - N_FFT) // HOP_LENGTH

    print("=" * 60)
    print("  ML Denoiser Pipeline")
    print("=" * 60)
    print(f"  Mic    : device {MIC_INDEX}")
    print(f"  Output : device {CABLE_INDEX}")
    print(f"  Rate   : {SAMPLE_RATE} Hz   Window: {WINDOW_SAMP/SAMPLE_RATE*1000:.0f}ms   Hop: {HOP_SAMP/SAMPLE_RATE*1000:.0f}ms")
    print(f"  Floor  : {SPECTRAL_FLOOR*100:.0f}%   Gate: {GATE_FLOOR:.4f}–{GATE_CEIL:.3f}")
    print()
    print("  Set your app's microphone to 'CABLE Output (VB-Audio Virtual Cable)'")
    print("  Press Ctrl+C to stop.")
    print()

    _proc = threading.Thread(target=_processing_thread, daemon=True)
    _proc.start()

    try:
        with sd.Stream(
            device=(MIC_INDEX, CABLE_INDEX),
            samplerate=SAMPLE_RATE,
            blocksize=CHUNK_SIZE,
            channels=CHANNELS,
            dtype='float32',
            callback=callback,
        ):
            print("  Active — 300ms startup buffer, then continuous.\n")
            sd.sleep(int(1e9))
    except KeyboardInterrupt:
        _running = False
        print(f"\n  Stopped.  Underruns: {_underruns}")
    except Exception as e:
        _running = False
        raise
