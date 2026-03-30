"""
DSP Filter Pipeline
====================
Lightweight fallback pipeline using only scipy signal processing.
No ML model required — useful for low-latency situations or
testing audio routing without the full denoiser.

Signal flow:
    Microphone → High-pass filter → Noise gate → Gain → Virtual Cable
"""

import sounddevice as sd
import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

# ── Devices ───────────────────────────────────────────────────────────────────
# Run list_devices.py to find the correct indexes for your system.
MIC_INDEX   = 0   # index of your microphone input device
CABLE_INDEX = 1   # index of your virtual cable input device

# ── Audio ─────────────────────────────────────────────────────────────────────
SAMPLE_RATE = 44100
CHUNK_SIZE  = 512
CHANNELS    = 1

# ── Controls ──────────────────────────────────────────────────────────────────
GAIN            = 1.2    # output level multiplier
GATE_THRESHOLD  = 0.01   # RMS below this is silenced
HIGHPASS_HZ     = 90     # cut frequencies below this (removes rumble/hum)

# ── Filter setup ──────────────────────────────────────────────────────────────
_sos     = butter(4, HIGHPASS_HZ, btype='highpass', fs=SAMPLE_RATE, output='sos')
_zi      = sosfilt_zi(_sos)[..., np.newaxis]

def callback(indata, outdata, frames, time, status):
    global _zi
    if status:
        print(f"[Status] {status}")

    audio            = indata[:, 0]
    filtered, _zi    = sosfilt(_sos, audio, zi=_zi[..., 0])
    _zi              = _zi[..., np.newaxis]

    if np.sqrt(np.mean(filtered ** 2)) < GATE_THRESHOLD:
        filtered = np.zeros_like(filtered)

    outdata[:, 0] = np.clip(filtered * GAIN, -1.0, 1.0)


if __name__ == "__main__":
    print("DSP Filter Pipeline")
    print(f"High-pass: {HIGHPASS_HZ}Hz | Gate: {GATE_THRESHOLD} | Gain: {GAIN}x")
    print("Set your app's microphone to 'CABLE Output (VB-Audio Virtual Cable)'")
    print("Press Ctrl+C to stop.\n")

    try:
        with sd.Stream(
            device=(MIC_INDEX, CABLE_INDEX),
            samplerate=SAMPLE_RATE,
            blocksize=CHUNK_SIZE,
            channels=CHANNELS,
            dtype='float32',
            callback=callback
        ):
            sd.sleep(int(1e9))
    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception as e:
        print(f"Error: {e}")
