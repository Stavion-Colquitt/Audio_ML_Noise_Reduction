"""Sanity checks for eval_vbdemand.py on a few files: is a drop the model or the harness?
  identity  : STFT -> iSTFT with no model (must score ~ the noisy input)
  pipeline  : the live pipeline's model stage (per-window 99th-percentile normalisation, 12% floor)
  trained   : whole utterance, peak-normalised to 1 like training, raw log1p magnitude (no percentile scaling), no floor
    py -3.12 eval_variants.py <clean dir> <noisy dir> [n]
"""
import sys, os, glob
import numpy as np
import eval_vbdemand as E

def whole(x, use_model=True, floor=0.0):
    pk = float(np.abs(x).max()) or 1.0; y = x / pk
    lm, ref = E.stft(y)
    if use_model:
        T = lm.shape[1]; T16 = int(np.ceil(T / 16) * 16)                 # 4 stride-2 levels -> multiple of 16 frames
        inp = np.pad(lm, ((0, 0), (0, T16 - T)), mode="edge").astype(np.float32)
        out = E.sess.run(None, {"noisy_spectrogram": inp[None, None]})[0][0, 0][:, :T]
        mag = np.maximum(np.expm1(np.clip(out, 0, None)), floor * np.expm1(lm))
    else:
        mag = np.expm1(lm)
    z = E.istft(mag, ref) * pk
    return np.pad(z, (0, max(0, len(x) - len(z))))[:len(x)]

def main():
    cdir, ndir = sys.argv[1], sys.argv[2]; n = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    files = sorted(glob.glob(os.path.join(ndir, "*.wav")))[:n]
    modes = {"noisy": lambda x: x, "identity": lambda x: whole(x, use_model=False),
             "pipeline": E.enhance, "trained": lambda x: whole(x), "trained+floor": lambda x: whole(x, floor=0.12)}
    R = {m: [[], [], []] for m in modes}
    for f in files:
        c, x = E.load(os.path.join(cdir, os.path.basename(f))), E.load(f); L = min(len(c), len(x)); c, x = c[:L], x[:L]
        for m, fn in modes.items():
            y = fn(x).astype(np.float32)
            R[m][0].append(E.pesq(E.SR, c, y, "wb")); R[m][1].append(E.stoi(c, y, E.SR)); R[m][2].append(E.si_sdr(c, y))
    print(f"{n} files      PESQ-WB   STOI   SI-SDR")
    for m, (p, s, d) in R.items(): print(f"{m:14s} {np.mean(p):6.3f} {np.mean(s):6.3f} {np.mean(d):7.2f}")

if __name__ == "__main__":
    main()
