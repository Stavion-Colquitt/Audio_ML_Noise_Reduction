# Audio ML Noise Reduction

> Real-time background noise suppression for Teams, Zoom, and any communication app.
> Built from scratch — custom-trained U-Net model, ONNX inference, virtual audio routing.
> No cloud. No subscription. Runs entirely on your machine.

---

## The Problem

Background noise on calls is a solved problem for big tech but a black box — you
can't control it, tune it, or understand it. This project builds the full pipeline
from scratch: dataset preparation, model architecture, training loop, ONNX export,
and a real-time inference pipeline that integrates with any app that accepts a
microphone input.

The result is a noise suppressor that runs locally at ~100ms latency, costs nothing
per call, and exposes every parameter so you can tune it to your environment.

---

## How It Works

```
Your mic (16 kHz)
    │
    ▼  Short-Time Fourier Transform (STFT)
       Window: 32ms  |  Hop: 8ms  |  Hann window
       Output: log-magnitude spectrogram [257 freq bins × 22 frames]
    │
    ▼  Custom U-Net (ONNX, CPU inference, ~15–25ms per window)
       Predicts a soft mask [0–1] per time-frequency bin
       Applies mask to noisy spectrogram → clean magnitude
    │
    ▼  Inverse STFT + Overlap-Add (OLA)
       50% Hann window overlap — constant-power reconstruction, no seam artifacts
    │
    ▼  Post-processing EQ
       +5 dB bandpass boost (100 Hz – 1 kHz) — restores voice body
       attenuated by noisy-phase ISTFT reconstruction
    │
    ▼  Soft noise gate with 800ms hold
       Attenuates only true silence — consonants always pass through
    │
    ▼  Virtual audio cable → your app sees a clean mic
```

The pipeline runs natively at 16 kHz — the model's training sample rate — which
eliminates resampling and the boundary artifacts it introduces.

---

## Quick Start

**1. Install dependencies**
```bash
pip install sounddevice numpy scipy onnxruntime torch torchaudio tqdm
```

**2. Install a virtual audio cable**
- Windows: [VB-Audio Virtual Cable](https://vb-audio.com/Cable/) (free)
- Mac: [BlackHole](https://existential.audio/blackhole/) (free)

**3. Find your audio device indexes**
```bash
python list_devices.py
```
Set `MIC_INDEX` and `CABLE_INDEX` at the top of `ml_denoiser_pipeline.py`.

**4. Get the model**

Option A — Download pretrained weights *(recommended)*:
Download `denoiser.onnx` from [**Releases**](../../releases) and place it in the
project root.

Option B — Train from scratch:
```bash
python download_dataset.py   # ~4–5 GB DNS Challenge 4 subset
python train.py              # 50 epochs — GPU recommended, CPU works
python export_model.py       # PyTorch checkpoint → denoiser.onnx
```

**5. Run**
```bash
python ml_denoiser_pipeline.py
# Windows launcher:
start_pipeline.bat
```

In Teams / Zoom / your DAW, set the microphone to
**`CABLE Output (VB-Audio Virtual Cable)`**.

---

## Two Pipelines

This project ships two runnable pipelines. Use the one that fits your situation.

### ML Denoiser Pipeline *(recommended)*
**File:** `ml_denoiser_pipeline.py`
**Command:** `python ml_denoiser_pipeline.py` or `start_pipeline.bat`

The full pipeline — U-Net ONNX inference, Overlap-Add synthesis, post-processing
EQ, and soft noise gate. Best output quality. Requires `denoiser.onnx` (download
from Releases or train from scratch).

- ✅ Best noise suppression quality
- ✅ Preserves voice naturally
- ✅ Handles non-stationary noise (keyboard, HVAC, background voices)
- ⚠️ Requires the ONNX model file
- ⚠️ ~220ms algorithmic latency

### DSP Filter Pipeline *(fallback)*
**File:** `dsp_filter_pipeline.py`
**Command:** `python dsp_filter_pipeline.py` or `start_pipeline.bat b`

A lightweight alternative using only scipy signal processing — no model required.
A 4th-order Butterworth high-pass filter removes low-frequency rumble and hum,
followed by a simple RMS noise gate. Runs on any machine instantly.

- ✅ No model needed — works out of the box
- ✅ Near-zero latency (~5ms)
- ✅ Minimal CPU usage
- ⚠️ Only removes stationary low-frequency noise (hum, rumble)
- ⚠️ Does not suppress broadband noise or background voices

### Switching Between Pipelines

```bash
start_pipeline.bat        # ML Denoiser Pipeline (default)
start_pipeline.bat b      # DSP Filter Pipeline
start_pipeline.bat list   # List audio device indexes
```

Or run either script directly with `python`. The launcher kills any existing
instance before starting the new one so you never end up with two pipelines
writing to the same virtual cable simultaneously.

---

The denoiser is a **U-Net trained from scratch** on the
[DNS Challenge 4](https://github.com/microsoft/DNS-Challenge) dataset —
the same benchmark used by Microsoft Teams, Krisp, and NVIDIA RTX Voice to
evaluate their noise suppression systems.

### Architecture

```
Input: noisy log-magnitude spectrogram  [B, 1, 257, T]

Encoder (stride-2 convolutions)         Decoder (transposed conv + skip)
────────────────────────────────────────────────────────────────────────
Conv(1  → 32 )                          UpBlock(512→256) ← skip e4
Conv(32 → 64 , stride=2)               UpBlock(256→128) ← skip e3
Conv(64 → 128, stride=2)               UpBlock(128→64 ) ← skip e2
Conv(128→ 256, stride=2)               UpBlock( 64→32 ) ← skip e1
              ↓
      Bottleneck Conv(256→512, stride=2)

Output head: Conv(32→1) + Sigmoid → soft mask ∈ [0, 1]
Final output: input × mask  (masking-based enhancement)
```

Masking-based enhancement constrains the output to be a scaled version of the
noisy input — the model can only suppress, never hallucinate spectral content
not already present in the signal.

### Training

| Property | Value |
|----------|-------|
| Parameters | ~3.8M (`base_ch=32`) |
| Dataset | DNS Challenge 4 — clean speech + Freesound noise |
| Mixing | Dynamic SNR augmentation: −5 dB to +20 dB per epoch |
| Loss | SI-SNR (Scale-Invariant Signal-to-Noise Ratio) |
| Optimizer | Adam, lr=1e-3, cosine annealing to 1e-5 |
| Epochs | 50 |
| Hardware | GPU recommended (CPU training ~10× slower) |

**Why SI-SNR over MSE:**
SI-SNR is invariant to signal scaling and correlates more strongly with
perceptual quality metrics (PESQ/STOI) than magnitude-domain MSE. It is the
standard loss function for speech enhancement research.

---

## Results

```bash
pip install pesq pystoi
```

| Metric | Description | Score |
|--------|-------------|-------|
| PESQ (WB) | Perceptual speech quality — ITU-T P.862 |
| STOI | Short-time objective intelligibility |
| SI-SNR improvement | SNR gain over noisy input |
| End-to-end latency | Mic to virtual cable output | ~220ms ongoing |

Target baselines for reference: PESQ > 3.0, STOI > 0.85, SI-SNR improvement > 10 dB.

---

## Latency

**Ongoing algorithmic latency: ~220ms**

| Component | Time | Notes |
|-----------|------|-------|
| Analysis window | 200ms | Model requires a full window before inference can run |
| ONNX inference | ~15–25ms | CPU, single-threaded |
| OLA hop | 100ms | Output emitted every hop — dominates perceived smoothness |
| EQ + soft gate | ~2ms | Negligible |
| **Total** | **~220ms** | After 300ms startup buffer fills |

The 300ms startup buffer is a one-time hold on launch — it fills once and then
the pipeline runs continuously at ~220ms latency.

**This is acceptable for communication apps.** Microsoft Teams, Zoom, and most
WebRTC systems already introduce 50–150ms of network + encoding latency. The
~220ms added by this pipeline stays below the ITU G.114 400ms total threshold
for acceptable conversational delay.

### Reducing Latency

The latency is dominated by the 200ms analysis window — not the EQ or gate.

**Quick win — remove EQ and gate (~2ms saving, negligible):**
In `ml_denoiser_pipeline.py`, comment out the EQ and gate blocks in
`_processing_thread`. The model's spectral suppression handles most noise
without them. This saves ~2ms which is imperceptible.

**Real reduction — halve the window size:**

```python
WINDOW_SAMP = 1600   # 100ms window (was 200ms)
HOP_SAMP    = 800    # 50ms hop    (was 100ms)
```

This cuts algorithmic latency to ~120ms at the cost of fewer STFT frames per
inference (11 vs 22). Model quality degrades somewhat because it has less
temporal context per window. Tune `SPECTRAL_FLOOR` up slightly (0.15–0.18) to
compensate for less confident suppression at the shorter window.

**Sub-10ms latency** requires replacing the STFT-domain pipeline entirely with
a time-domain model (Conv-TasNet, DTLN, or similar) that processes shorter
frames without a large analysis window. That is a model retrain, not a
parameter change.

---

**Why native 16 kHz?**
The model was trained at 16 kHz. Running at 44.1 kHz and resampling around the
model introduces polyphase filter transients at every chunk boundary — audible
as periodic pops. Running natively at 16 kHz eliminates both resamples. Speech
intelligibility lives entirely below 8 kHz (the 16 kHz Nyquist), so nothing
relevant is lost.

**Why Overlap-Add at 50% overlap?**
The Hann window has the Constant Overlap-Add (COLA) property at 50% — consecutive
windowed frames sum to a constant value, giving artifact-free reconstruction at
every chunk boundary without any explicit crossfade logic.

**Why a soft gate instead of hard silence detection?**
Unvoiced consonants (p, t, k, s, f) have very low RMS but are essential for
intelligibility. A hard RMS threshold set high enough to cut background noise
will also cut these consonants. The soft gate uses a slow release (GATE_RELEASE)
and an 800ms hold period so the gate stays open through natural speech gaps and
quiet phonemes.

**Why post-processing EQ?**
Masking-based models reuse the noisy phase in iSTFT reconstruction. This
combination of clean magnitude + noisy phase is technically an invalid STFT —
it attenuates some frequency content, particularly in the 100Hz–1kHz voice body
range. A stateful Butterworth bandpass filter adds +5 dB in that range to
compensate without affecting higher frequencies.

**`time.sleep(0)` instead of `time.sleep(0.001)`:**
On Windows, any `sleep(t > 0)` waits a minimum of ~15.6ms due to the system
timer resolution. In a 100ms-hop pipeline that means the processing thread
effectively sleeps 15× longer than intended, causing buffer drain and periodic
loudness oscillation. `sleep(0)` yields the thread's CPU timeslice without any
minimum wait.

---

## Known Limitations

- **Phase reuse muffling:** The model predicts clean magnitude but reconstruction
  uses the noisy phase. Unvoiced consonants can sound slightly muffled on some
  inputs. Fix: retrain with a complex-mask architecture (DCUNet) that predicts
  both magnitude and phase jointly.

- **Latency:** ~300ms algorithmic latency (200ms analysis window + startup buffer).
  Acceptable for communication apps but too high for live monitoring during
  recording. Sub-10ms latency requires a time-domain model (Conv-TasNet, DTLN).

- **Fixed noise model:** Trained on DNS Challenge noise types (HVAC, traffic,
  keyboard, office). Degrades on out-of-distribution noise. Fine-tuning on
  domain-specific recordings improves this significantly.

---

## Future Work

- [ ] Run PESQ/STOI/DNSMOS evaluation and publish scores
- [ ] Fine-tune on domain-specific conference room noise
- [ ] Complex-mask U-Net (DCUNet) to address phase reuse muffling
- [ ] Voice Activity Detection (VAD) to replace RMS-based gate
- [ ] INT8 quantization (ONNX Runtime) for lower inference latency
- [ ] Mac/Linux shell launcher
- [ ] DAW-optimized variant (<10ms latency target)

---

## Files

| File | Description |
|------|-------------|
| `ml_denoiser_pipeline.py` | **Main pipeline** — real-time inference, OLA, EQ, gate |
| `dsp_filter_pipeline.py` | Fallback — high-pass filter + gate, no model needed |
| `model.py` | U-Net architecture |
| `dataset.py` | DNS Challenge loader with dynamic SNR mixing |
| `train.py` | Training loop — SI-SNR loss, cosine LR schedule |
| `export_model.py` | PyTorch → ONNX export |
| `download_dataset.py` | DNS Challenge 4 subset downloader |
| `list_devices.py` | List all audio device indexes |
| `start_pipeline.bat` | Windows launcher |

---

## Tuning Parameters

| Parameter | Default | What it does |
|-----------|---------|--------------|
| `SPECTRAL_FLOOR` | 0.12 | Minimum energy kept per frequency bin — raise if voice sounds thin |
| `GATE_FLOOR` | 0.0003 | ~−70 dBFS — below this RMS the gate attenuates |
| `GATE_CEIL` | 0.004 | ~−48 dBFS — above this the gate is fully open |
| `GATE_HOLD` | 8 hops | How long gate stays open after signal drops (~800ms) |
| `INPUT_GAIN` | 0.75 | Pre-model level — reduce if you hear distortion |
| `OUTPUT_GAIN` | 3.2 | Post-gate output level — raise if output is quiet |
| `WINDOW_SAMP` | 3200 | Analysis window — 200ms at 16kHz |
| `HOP_SAMP` | 1600 | OLA hop — 100ms, 50% overlap |

---

## Tech Stack

**ML / Inference:** PyTorch · ONNX Runtime · NumPy

**Audio DSP:** SciPy (Butterworth filters, sosfilt) · sounddevice · torchaudio

**Dataset:** DNS Challenge 4 (Microsoft Deep Noise Suppression Challenge)

---

## License

MIT
