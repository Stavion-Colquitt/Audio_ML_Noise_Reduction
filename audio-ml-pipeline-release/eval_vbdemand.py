"""
Measure the denoiser on the VoiceBank+DEMAND test set (824 utterances, the standard speech-enhancement benchmark;
never used in training -- the model was trained on DNS Challenge 4 data only).

Enhancement = exactly the live pipeline's model stage (_run_model): 200 ms windows, 100 ms hop, Hann OLA,
per-window log-magnitude normalisation, 12% spectral floor. The pipeline's EQ / gate / output gain are left out
(they are level shaping for calls, not denoising), so the numbers describe the model itself.

Metrics (noisy input vs enhanced output, both against the clean reference, 16 kHz):
  PESQ (wide-band, ITU-T P.862.2), STOI, SI-SDR (dB); plus real-time factor on this CPU and algorithmic latency.

    py -3.12 eval_vbdemand.py <clean_testset_wav dir> <noisy_testset_wav dir> [max_files]
"""
import sys, os, glob, time, json
import numpy as np, soundfile as sf
from scipy.signal import resample_poly
import onnxruntime as ort
from pesq import pesq
from pystoi import stoi

SR, N_FFT, HOP = 16000, 512, 128
WIN, HOP_SAMP, FLOOR = 3200, 1600, 0.12
ONNX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "denoiser.onnx")
sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])


def stft(x):
    w = np.hanning(N_FFT); n = max(0, 1 + (len(x) - N_FFT) // HOP)
    S = np.fft.rfft(np.stack([x[i * HOP:i * HOP + N_FFT] * w for i in range(n)]), n=N_FFT, axis=1)
    return np.log1p(np.abs(S).T), S


def istft(mag, ref):
    fr = np.fft.irfft(mag.T * np.exp(1j * np.angle(ref)), n=N_FFT, axis=1); w = np.hanning(N_FFT)
    L = (fr.shape[0] - 1) * HOP + N_FFT; o = np.zeros(L); nm = np.zeros(L)
    for i, f in enumerate(fr):
        o[i * HOP:i * HOP + N_FFT] += f * w; nm[i * HOP:i * HOP + N_FFT] += w ** 2
    return (o / np.where(nm > 1e-8, nm, 1.0)).astype(np.float32)


def run_model(chunk):                        # == ml_denoiser_pipeline._run_model
    lm, ref = stft(chunk); s = float(np.percentile(lm, 99))
    if s < 0.01: return np.zeros(len(chunk), np.float32)
    out = sess.run(None, {"noisy_spectrogram": (lm / s).astype(np.float32)[None, None]})[0][0, 0] * s
    cm = np.maximum(np.expm1(np.clip(out, 0, None)), FLOOR * np.expm1(lm))
    y = istft(cm, ref)
    return y[:len(chunk)] if len(y) >= len(chunk) else np.pad(y, (0, len(chunk) - len(y)))


def enhance(x):                              # offline OLA over the same windows the live stream uses
    pad = np.concatenate([np.zeros(WIN - HOP_SAMP, np.float32), x, np.zeros(WIN, np.float32)])
    acc = np.zeros(len(pad)); nrm = np.zeros(len(pad)); h = np.hanning(WIN)
    for s in range(0, len(pad) - WIN + 1, HOP_SAMP):
        acc[s:s + WIN] += run_model(pad[s:s + WIN]) * h; nrm[s:s + WIN] += h
    y = acc / np.where(nrm > 1e-6, nrm, 1.0)
    return y[WIN - HOP_SAMP:WIN - HOP_SAMP + len(x)].astype(np.float32)


def si_sdr(ref, est):
    ref = ref - ref.mean(); est = est - est.mean()
    a = np.dot(est, ref) / (np.dot(ref, ref) + 1e-12); t = a * ref
    return 10 * np.log10((t ** 2).sum() / (((est - t) ** 2).sum() + 1e-12) + 1e-12)


def load(p):
    x, sr = sf.read(p, dtype="float32")
    if x.ndim > 1: x = x.mean(1)
    return resample_poly(x, SR, sr).astype(np.float32) if sr != SR else x


def main():
    cdir, ndir = sys.argv[1], sys.argv[2]; lim = int(sys.argv[3]) if len(sys.argv) > 3 else None
    files = sorted(glob.glob(os.path.join(ndir, "*.wav")))[:lim]
    R = {k: [] for k in ("pesq_n", "pesq_e", "stoi_n", "stoi_e", "sisdr_n", "sisdr_e")}; t_audio = t_cpu = 0.0
    for i, f in enumerate(files):
        c, n = load(os.path.join(cdir, os.path.basename(f))), load(f); L = min(len(c), len(n)); c, n = c[:L], n[:L]
        t0 = time.perf_counter(); e = enhance(n); t_cpu += time.perf_counter() - t0; t_audio += L / SR
        for tag, y in (("n", n), ("e", e)):
            R["pesq_" + tag].append(pesq(SR, c, y, "wb")); R["stoi_" + tag].append(stoi(c, y, SR, extended=False))
            R["sisdr_" + tag].append(si_sdr(c, y))
        if i % 100 == 0: print(f"{i}/{len(files)}", flush=True)
    m = {k: float(np.mean(v)) for k, v in R.items()}
    print(f"\nVoiceBank+DEMAND test, {len(files)} utterances (16 kHz)\n")
    print(f"{'':10s}{'noisy':>9s}{'enhanced':>10s}{'change':>9s}")
    for name, k in (("PESQ-WB", "pesq"), ("STOI", "stoi"), ("SI-SDR dB", "sisdr")):
        print(f"{name:10s}{m[k + '_n']:9.3f}{m[k + '_e']:10.3f}{m[k + '_e'] - m[k + '_n']:+9.3f}")
    print(f"\nreal-time factor on this CPU: {t_cpu / t_audio:.3f}  ({t_audio:.0f} s of audio in {t_cpu:.0f} s)")
    print(f"algorithmic latency: {WIN / SR * 1000:.0f} ms window, {HOP_SAMP / SR * 1000:.0f} ms hop")
    json.dump({"n": len(files), "means": m, "rtf": t_cpu / t_audio}, open("eval_vbdemand.json", "w"), indent=1)


if __name__ == "__main__":
    main()
