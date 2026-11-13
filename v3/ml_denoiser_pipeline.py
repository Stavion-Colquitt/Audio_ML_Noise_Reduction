"""
ML Denoiser Pipeline  v10
=========================
v9 CHANGES:
- Removed all resampling. Stream now runs natively at 16kHz — the same
  sample rate the U-Net model was trained on. Eliminates both resample
  calls, the overlap buffer warm-start complexity, and resampler boundary
  pops entirely.
- SPECTRAL_FLOOR raised 0.08 → 0.12 to reduce muffling on unvoiced
  consonants (s, f, sh, th). Phase reuse in ISTFT attenuates these
  frequencies; a higher floor preserves more of the original spectral
  content in those bins.
- time.sleep(0.001) → time.sleep(0): Windows timer granularity means
  0.001s actually sleeps ~15ms, draining the output buffer and causing
  the loud/quiet oscillation. sleep(0) yields without a minimum wait.
- Startup gate in callback: outputs silence until 300ms of audio is
  buffered so the stream never starts in an underrun state.
"""

import threading
import queue as qmod
import time
import math
import numpy as np
import sounddevice as sd
import onnxruntime as ort  # type: ignore
from scipy.signal import butter, sosfilt, sosfilt_zi
from math import gcd

# ── Devices ───────────────────────────────────────────────────────────────────
MIC_INDEX   = 2    # Razer Barracuda X Chroma  (MME)
CABLE_INDEX = 13   # CABLE Input VB-Audio       (MME)

# ── Audio — native 16kHz, no resampling ──────────────────────────────────────
SAMPLE_RATE = 16000
CHUNK_SIZE  = 256
CHANNELS    = 1
import v3_infer                     # v2 model stage: shared with the evaluators so they can't drift
ONNX_PATH   = v3_infer.ONNX_PATH

# ── STFT ──────────────────────────────────────────────────────────────────────
N_FFT      = 512
HOP_LENGTH = 128

# ── OLA parameters ────────────────────────────────────────────────────────────
WINDOW_SAMP = 3200   # 200ms @ 16kHz (22 STFT frames)
HOP_SAMP    = 1600   # 100ms hop → 50% overlap

# ── Spectral floor ────────────────────────────────────────────────────────────
SPECTRAL_FLOOR = 0.12   # 12% — preserves unvoiced consonants (s, f, sh, th)

# ── Soft noise gate ───────────────────────────────────────────────────────────
# Gate is intentionally lenient — the U-Net model handles noise suppression.
# Gate only catches genuine silence. Consonants (p,t,k,s,f) have very low RMS
# and must pass through — GATE_FLOOR is set low enough that only true silence
# (no speech activity at all) gets attenuated.
GATE_FLOOR    = 0.0003   # ~-70dBFS — only dead silence triggers attenuation
GATE_CEIL     = 0.004    # ~-48dBFS — full pass above this
GATE_ATTACK   = 0.30     # fast open
GATE_RELEASE  = 0.04     # slow close — trails off naturally
GATE_HOLD     = 8        # hops to hold open after signal drops (~800ms)
_gate_gain    = 0.0
_gate_hold_ct = 0

# ── Gain staging ─────────────────────────────────────────────────────────────
INPUT_GAIN  = 0.75
OUTPUT_GAIN = 3.2   # raised — previous EQ shelf was inflating apparent loudness

# ── Post-processing EQ ───────────────────────────────────────────────────────
# Voice body boost: bandpass peak centered at 500Hz, covering ~100Hz-1kHz.
# This is where speaking voice fundamentals, first/second formants, and
# vowel body all live. Boosting here restores warmth and intelligibility
# that the spectral suppression model attenuates.
# Implementation: difference of two high-pass filters creates a band.
_EQ_GAIN_DB  = 5.0                            # +5dB boost in the band
_EQ_GAIN_LIN = 10 ** (_EQ_GAIN_DB / 20) - 1  # additive gain (parallel EQ)
_sos_lo  = butter(2,  100 / (SAMPLE_RATE / 2), btype='high', output='sos')
_sos_hi  = butter(2, 1000 / (SAMPLE_RATE / 2), btype='low',  output='sos')
_eq_zi_lo = sosfilt_zi(_sos_lo).astype(np.float64)
_eq_zi_hi = sosfilt_zi(_sos_hi).astype(np.float64)

# ── Hann OLA synthesis window ─────────────────────────────────────────────────
_hann_syn  = np.hanning(WINDOW_SAMP).astype(np.float32)
_ola_accum = np.zeros(WINDOW_SAMP, dtype=np.float32)
_ola_norm  = np.zeros(WINDOW_SAMP, dtype=np.float32)

print("Loading model...")
session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
print("Model loaded.")

# ── Pre-warm OLA buffer ───────────────────────────────────────────────────────
_warmup_iters = WINDOW_SAMP // HOP_SAMP - 1   # = 1 for 50% overlap
for _ in range(_warmup_iters):
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
    log_mag  = np.log1p(np.abs(spectra).T)
    return log_mag, spectra

def _istft(magnitude, phase_ref):
    phase         = np.angle(phase_ref)
    reconstructed = magnitude.T * np.exp(1j * phase)
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
    # v2: linear-magnitude mask, training-matched STFT, no per-window rescaling (see v3_infer.py).
    # The v1 body below is kept for reference but no longer runs.
    return v3_infer.run_model(chunk)
    log_mag, phase_ref = _stft(chunk)
    scale = float(np.percentile(log_mag, 99))
    if scale < 0.01:
        return np.zeros(len(chunk), dtype=np.float32)
    norm_log   = (log_mag / scale).astype(np.float32)
    clean_norm = session.run(
        None, {"noisy_spectrogram": norm_log[np.newaxis, np.newaxis]})[0][0, 0]
    clean_log = clean_norm * scale
    noisy_mag = np.expm1(log_mag)
    clean_mag = np.expm1(np.clip(clean_log, 0, None))
    clean_mag = np.maximum(clean_mag, SPECTRAL_FLOOR * noisy_mag)
    out = _istft(clean_mag, phase_ref)
    return out[:len(chunk)] if len(out) >= len(chunk) \
           else np.pad(out, (0, len(chunk) - len(out)))

# ── Thread-safe buffers ───────────────────────────────────────────────────────
_in_q      = qmod.Queue()
_out_buf   = np.zeros(0, dtype=np.float32)
_out_lock  = threading.Lock()
_running   = True
_underruns = 0
# Startup gate: hold output until 300ms of audio is buffered
_MIN_STARTUP = HOP_SAMP * 3   # 300ms @ 16kHz

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
            raw     = np.concatenate(chunks) * INPUT_GAIN
            pending = np.concatenate([pending, raw])

        while len(pending) >= WINDOW_SAMP:
            chunk   = pending[:WINDOW_SAMP]
            pending = pending[HOP_SAMP:]

            clean = _run_model(chunk)

            # ── Hann OLA synthesis ────────────────────────────────────────
            _ola_accum += clean * _hann_syn
            _ola_norm  += _hann_syn
            safe_norm   = np.where(_ola_norm[:HOP_SAMP] > 1e-6,
                                   _ola_norm[:HOP_SAMP], 1.0)
            out         = (_ola_accum[:HOP_SAMP] / safe_norm).astype(np.float32)
            _ola_accum[:-HOP_SAMP] = _ola_accum[HOP_SAMP:].copy()
            _ola_accum[-HOP_SAMP:] = 0.0
            _ola_norm[:-HOP_SAMP]  = _ola_norm[HOP_SAMP:].copy()
            _ola_norm[-HOP_SAMP:]  = 0.0

            # ── Voice body EQ boost (100Hz–1kHz, +5dB) ───────────────────
            # Bandpass: high-pass at 100Hz then low-pass at 1kHz.
            # Boosts voice fundamentals and formants suppressed by the model.
            sig64 = out.astype(np.float64)
            hp, _eq_zi_lo = sosfilt(_sos_lo, sig64, zi=_eq_zi_lo)
            band, _eq_zi_hi = sosfilt(_sos_hi, hp,   zi=_eq_zi_hi)
            out = (out + band.astype(np.float32) * _EQ_GAIN_LIN)

            # ── Soft noise gate with hold ─────────────────────────────────
            frame_rms = float(np.sqrt(np.mean(out ** 2)))
            if frame_rms <= GATE_FLOOR:
                target = 0.0
            elif frame_rms >= GATE_CEIL:
                target = 1.0
            else:
                target = (frame_rms - GATE_FLOOR) / (GATE_CEIL - GATE_FLOOR)
            # Hold: any signal above the floor resets the hold counter.
            # This means the gate stays open through brief silent gaps
            # between syllables and across short consonants.
            if frame_rms > GATE_FLOOR:
                _gate_hold_ct = GATE_HOLD
            elif _gate_hold_ct > 0:
                _gate_hold_ct -= 1
                target = max(target, 0.9)   # stay near-open during hold
            coeff      = GATE_ATTACK if target > _gate_gain else GATE_RELEASE
            _gate_gain = _gate_gain + coeff * (target - _gate_gain)
            out        = out * _gate_gain * OUTPUT_GAIN

            with _out_lock:
                _out_buf = np.concatenate([_out_buf, out])

        time.sleep(0)   # yield — time.sleep(0.001) sleeps ~15ms on Windows

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
stft_frames = 1 + (WINDOW_SAMP - N_FFT) // HOP_LENGTH

print("=" * 68)
print("  ML Denoiser Pipeline  v10  --  native 16kHz, U-Net ONNX")
print("=" * 68)
print(f"  Input  : device {MIC_INDEX}  Razer Barracuda X Chroma")
print(f"  Output : device {CABLE_INDEX}  CABLE Input (VB-Audio)")
print(f"  Rate   : {SAMPLE_RATE} Hz (native — no resampling)")
print(f"  Window : {WINDOW_SAMP/SAMPLE_RATE*1000:.0f} ms  ({stft_frames} STFT frames)  "
      f"Hop: {HOP_SAMP/SAMPLE_RATE*1000:.0f} ms")
print(f"  Floor  : {SPECTRAL_FLOOR*100:.0f}%   Gate: {GATE_FLOOR:.3f} to {GATE_CEIL:.3f}")
print(f"  Gain   : in={INPUT_GAIN}x  out={OUTPUT_GAIN}x")
print()
print("  In Teams/Zoom: mic = 'CABLE Output (VB-Audio Virtual Cable)'")
print("  Press Ctrl+C to stop.")
print()

_proc = threading.Thread(target=_processing_thread, daemon=True)
_proc.start()
print("  Processing thread started.")

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
    print(f"\n  Error: {e}")
    raise
