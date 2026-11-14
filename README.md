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
- **v4 (best quality, 10 ms streaming):** download `call_denoiser_v4.onnx` from [**Release V4.0**](../../releases/tag/V4.0)
  and place it in `v4/`. It runs through the block engine in `v4/stream.py` (80-sample / 5 ms blocks); wiring it
  into `ml_denoiser_pipeline.py` is the next step, so the live pipeline below still uses v2.
- **v2 (recommended for the live pipeline):** download `denoiser_v2.onnx` from [**Release V2.0**](../../releases/tag/V2.0) and place it in `v2/`, next to
  `ml_denoiser_pipeline.py`. Run the pipeline from `v2/`.
- **v3 (experimental):** download `denoiser_v3.onnx` from [**Release V3.0**](../../releases/tag/V3.0) and place it in `v3/`.
  Run with `V3_PHASE=noisy` for the better-scoring mode (see Results).
- **v1:** download `denoiser.onnx` from [**Release V1.0**](../../releases/tag/V1.0) and place it in `audio-ml-pipeline-release/`, next to
  `ml_denoiser_pipeline.py`.

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

Evaluated on the **VoiceBank+DEMAND test set** (824 utterances, 16 kHz), the standard speech-enhancement
benchmark. Neither model saw it in training (both were trained on DNS Challenge 4 only). Enhancement runs exactly
the live pipeline's model stage (200 ms windows, 100 ms hop, Hann overlap-add); the EQ, gate and output gain are
left out, so the numbers describe the models themselves.

| Metric | Noisy input | v1 | **v2** |
|--------|-------------|----|--------|
| SI-SDR (dB) | 8.45 | 3.42 (-5.03) | **13.36 (+4.91)** |
| STOI | 0.921 | 0.798 (-0.123) | **0.869 (-0.052)** |
| PESQ (WB, ITU-T P.862.2) | 1.97 | 1.58 (-0.39) | **1.65 (-0.32)** |
| DNSMOS P.835 SIG (speech) | 3.32 | 3.11 (-0.21) | 3.25 (-0.07) |
| DNSMOS P.835 BAK (background) | 3.11 | 3.48 (+0.37) | 3.18 (+0.07) |
| DNSMOS P.835 OVRL (overall) | 2.68 | 2.63 (-0.05) | 2.63 (-0.05) |
| Real-time factor (desktop CPU, ONNX Runtime) | | 0.077 (13x real time) | **0.063 (16x real time)** |

Algorithmic latency for both: 200 ms analysis window, 100 ms hop.

### What changed in v2 (code in [`v2/`](v2/))

v1 computed its SI-SNR loss on log1p **spectrograms**. SI-SNR ignores scale, and a scale on a log spectrogram is a
power law on the audio, so the model could shrink the whole spectrum almost for free; on real audio that showed up
as over-suppression (SI-SDR fell 5 dB below the noisy input). v2 fixes the training objective and the
train/inference mismatches:

- **Loss on the reconstructed waveform:** SI-SDR + multi-resolution STFT loss after iSTFT.
- **Mask on the linear magnitude** (v1 masked the log1p magnitude).
- **Training matches inference:** 200 ms segments, the same STFT, and no per-window 99th-percentile rescaling.
- **Data:** SNR -5 to +30 dB (was -5 to +20), 10% clean-only examples, randomised input level.
- **Honest model selection:** fixed train/dev split by file (5% held out); checkpoints chosen by dev score; the test
  set run once at the end. Runs tracked in MLflow.
- **Parity check:** `v2/check_parity.py` confirms the numpy/ONNX inference path reproduces the PyTorch training
  path (138 dB parity SNR).

**What v2 says:** removing the loss bug turned a 5 dB SI-SDR loss into a **4.9 dB gain**, and more than halved the
intelligibility (STOI) drop. Speech quality (PESQ) is still slightly below the noisy input: the remaining
suspects are the reused noisy phase and deep suppression, which a complex-mask v3 targets.

**Reproduce:**
```bash
pip install pesq pystoi
python eval_vbdemand.py <clean_testset_wav dir> <noisy_testset_wav dir>   # PESQ / STOI / SI-SDR / speed
python eval_dnsmos.py <noisy_testset_wav dir> 824                          # DNSMOS (needs microsoft/DNS-Challenge DNSMOS models in ./DNS-Challenge/DNSMOS)
python eval_variants.py <clean dir> <noisy dir> 20                         # v1 only: sanity checks (identity pass-through, variants)
```
Run them from `audio-ml-pipeline-release/` for v1 or `v2/` for v2. The test set is on
[Edinburgh DataShare](https://datashare.ed.ac.uk/handle/10283/2791).

---

### v3: complex mask, a negative result (code in [`v3/`](v3/))

**Hypothesis:** v2 is still below the noisy input on PESQ because it reuses the noisy phase. v3 tests that by
predicting a **complex ratio mask** (tanh-bounded real and imaginary parts) from three input channels
(`log1p|S|`, `|S|^0.3 cos phase`, `|S|^0.3 sin phase`). Everything else is identical to v2 (data, loss, 200 ms
segments, 60 epochs, model size ~3.8M parameters), so the comparison isolates the mask type.

| Metric | Noisy input | v2 | v3 (model phase) | v3 (noisy phase) |
|--------|-------------|----|------------------|------------------|
| SI-SDR (dB) | 8.45 | **13.36** | 13.02 | 13.17 |
| STOI | 0.921 | 0.869 | 0.868 | **0.874** |
| PESQ (WB) | 1.97 | 1.65 | 1.61 | **1.70** |
| DNSMOS SIG | 3.32 | 3.25 | 3.26 | 3.27 |
| DNSMOS BAK | 3.11 | 3.18 | 3.07 | 3.14 |
| DNSMOS OVRL | 2.68 | 2.63 | 2.58 | 2.62 |

"v3 (noisy phase)" uses only the magnitude of v3's mask and keeps the noisy phase (`V3_PHASE=noisy`).

**What v3 says:**
- **The phase hypothesis did not hold.** Applying v3's own phase scores *worse* than discarding it and keeping the
  noisy phase (PESQ 1.61 vs 1.70). The model learned a better magnitude mask than v2, but its phase estimate hurts.
- **The 200 ms windowing is not the cause either.** Running the complex mask over whole utterances in one pass gives
  the same PESQ as the windowed pipeline (diagnostic on 150 files: `v3/eval_v3_variants.py`).
- **Every version is still below the noisy input on PESQ.** The remaining suspects are the training objective
  (SI-SDR + spectral loss rewards removing noise, not natural-sounding speech) and the short 200 ms context.
  Those are the next experiments (see Future Work).

Parity check (numpy/ONNX path vs PyTorch training path): 133.5 dB.

---

### v4: streaming GRU, 10 ms latency, first version above the noisy input (code in [`v4/`](v4/))

v4 drops the 200 ms-window U-Net for the causal streaming design built for the
[DAW vocal denoiser](https://github.com/Stavion-Colquitt/daw-vocal-denoiser), retuned for 16 kHz calls:

- **Asymmetric windows** (Wang et al. 2021): 32 ms analysis window for frequency resolution, 10 ms synthesis window
  for latency, 5 ms hop. Algorithmic latency drops from ~200 ms to **10 ms**.
- **2-layer GRU (256)** magnitude mask, 988k parameters: context comes from the GRU's memory instead of a long window.
- **Noisy phase kept** (the v3 lesson).
- **Compressed spectral loss** (Braun & Tashev, power 0.3) + multi-resolution STFT: tracks perceived quality better
  than v2's SI-SDR objective. Loss is level-normalised; the model sees the real input level.
- **Training data:** 60% VoiceBank-DEMAND **train** pairs (28 speakers; speakers p226/p287 held out as dev) + 40% DNS
  speech + DNS noise mixed on the fly (SNR -5 to 20 dB, 5% clean-only). EMA weights, warmup + cosine LR, 4 s segments.
- Trained on an RTX 2060 (26,208 steps, about 4 hours); the best dev checkpoint (step 26,000) was evaluated once.

| Metric (VoiceBank-DEMAND test, 824 files, 16 kHz) | Noisy input | v2 | v3 (noisy phase) | **v4** |
|--------|-------------|----|------------------|--------|
| PESQ (WB) | 1.97 | 1.65 | 1.70 | **2.58 (+0.61)** |
| STOI | 0.921 | 0.869 | 0.874 | **0.930 (+0.009)** |
| SI-SDR (dB) | 8.45 | 13.36 | 13.17 | **17.21 (+8.76)** |
| Algorithmic latency | | 200 ms | 200 ms | **10 ms** |
| CPU per 5 ms block (ONNX Runtime) | | | | **0.15 ms** (real-time factor 0.03) |

Streaming vs offline parity: 108.3 dB.

**What v4 says:**
- **It is the first version to beat the noisy input on every metric**, including PESQ, while cutting latency 20x.
- **Fair-comparison note:** v1-v3 never saw VoiceBank-DEMAND; v4 trains on its *training* split (different speakers
  from the test set, same recording setup and noise families). That is the standard protocol for this benchmark, and
  closing that domain gap is a large part of the jump. The architecture, loss and latency changes are the other part;
  separating the two is a clean follow-up ablation (v4 trained on DNS only).
- For context, published lightweight causal models report roughly 2.8-3.0 PESQ on this benchmark, so v4 is in range
  but not yet at the state of the art.

**Reproduce:**
```bash
python v4/prep_vbd.py                       # unzip + resample VoiceBank-DEMAND (train + test) to 16 kHz
python v4/train.py --hours 8                # or v4/run_v4_2060.ps1 for the whole unattended run
python v4/export_onnx.py && python v4/check_parity.py
python v4/eval_vbd.py                       # PESQ / STOI / SI-SDR on the 824 test files
```

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

- [x] Run PESQ/STOI/DNSMOS evaluation and publish scores
- [x] v2: waveform loss, linear-magnitude mask, milder-SNR mixtures, inference-matched training
- [x] v3: complex mask (phase): did not lift PESQ; v3's phase estimate hurts, its magnitude helps (see Results)
- [x] v4: streaming GRU, compressed (perceptual) loss, in-domain training data: first version above the noisy input on PESQ (2.58 vs 1.97)
- [ ] Wire the v4 streaming engine into `ml_denoiser_pipeline.py` (10 ms end to end instead of ~220 ms)
- [ ] Fine-tune on domain-specific conference room noise
- [ ] Complex-mask U-Net (DCUNet) to address phase reuse muffling
- [ ] Voice Activity Detection (VAD) to replace RMS-based gate
- [ ] INT8 quantization (ONNX Runtime) for lower inference latency
- [ ] Mac/Linux shell launcher
- [x] DAW-optimized variant: [daw-vocal-denoiser](https://github.com/Stavion-Colquitt/daw-vocal-denoiser) (48 kHz, 10 ms latency)

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
| `eval_vbdemand.py` | PESQ / STOI / SI-SDR and real-time factor on VoiceBank+DEMAND |
| `eval_dnsmos.py` | DNSMOS P.835 (SIG / BAK / OVRL), no clean reference needed |
| `eval_variants.py` | Evaluation sanity checks (pass-through, inference variants) |
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
