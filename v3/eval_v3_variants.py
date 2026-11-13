"""Diagnose v3 vs v2 on a VoiceBank+DEMAND subset (PESQ / SI-SDR), same files for every variant.
  v2            : v2 model, magnitude mask, noisy phase (the shipped v2)
  v3_crm        : v3 complex mask, floor 0.05 (the shipped v3 path)
  v3_crm_nofl   : v3 complex mask, no floor
  v3_magonly    : |v3 mask| with the NOISY phase  -> isolates whether v3's phase estimate helps or hurts
  v3_crm_whole  : v3 complex mask over the whole utterance in one pass (no outer 200 ms windows / OLA)
    py -3.12 eval_v3_variants.py <clean dir> <noisy dir> [n]
"""
import sys, os, glob, numpy as np, onnxruntime as ort
from pesq import pesq
import eval_vbdemand as E, v3_infer as V

V2_ONNX = os.environ.get("V2_ONNX", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "v2", "denoiser_v2.onnx"))
V2 = ort.InferenceSession(V2_ONNX, providers=["CPUExecutionProvider"])

def mask_v3(S):  # S [T, F] -> complex mask [T, F]
    m = V.session().run(None, {"noisy_spectrogram": V.features(S)})[0][0]
    return (m[0] + 1j * m[1]).T

def floor_c(M, fl):
    a = np.abs(M); return M * (np.maximum(a, fl) / np.maximum(a, 1e-8)) if fl > 0 else M

def rm_v2(ch):
    S = V.stft(ch); m = V2.run(None, {"noisy_spectrogram": np.log1p(np.abs(S)).T.astype(np.float32)[None, None]})[0][0, 0].T
    return V.istft(np.maximum(m, 0.05) * S, len(ch))
def rm_crm(ch, fl=0.05):
    S = V.stft(ch); return V.istft(floor_c(mask_v3(S), fl) * S, len(ch))
def rm_mag(ch):
    S = V.stft(ch); return V.istft(np.maximum(np.abs(mask_v3(S)), 0.05) * S, len(ch))
def whole(x):
    S = V.stft(x); return V.istft(floor_c(mask_v3(S), 0.05) * S, len(x))

VARIANTS = {"v2": lambda x: ola(x, rm_v2), "v3_crm": lambda x: ola(x, rm_crm),
            "v3_crm_nofl": lambda x: ola(x, lambda c: rm_crm(c, 0.0)), "v3_magonly": lambda x: ola(x, rm_mag),
            "v3_crm_whole": whole}

def ola(x, rm):
    E.run_model = rm; return E.enhance(x)

def main():
    cdir, ndir = sys.argv[1], sys.argv[2]; n = int(sys.argv[3]) if len(sys.argv) > 3 else 150
    files = sorted(glob.glob(os.path.join(ndir, "*.wav")))[::max(1, 824 // n)][:n]
    R = {k: [] for k in ["noisy"] + list(VARIANTS)}; D = {k: [] for k in R}
    for i, f in enumerate(files):
        c, x = E.load(os.path.join(cdir, os.path.basename(f))), E.load(f); L = min(len(c), len(x)); c, x = c[:L], x[:L]
        R["noisy"].append(pesq(E.SR, c, x, "wb")); D["noisy"].append(E.si_sdr(c, x))
        for k, fn in VARIANTS.items():
            y = fn(x)[:L]; R[k].append(pesq(E.SR, c, y, "wb")); D[k].append(E.si_sdr(c, y))
        if i % 25 == 0: print(f"{i}/{len(files)}", flush=True)
    print(f"\n{len(files)} VoiceBank+DEMAND test files        PESQ-WB   SI-SDR dB")
    for k in R: print(f"{k:34s} {np.mean(R[k]):7.3f}   {np.mean(D[k]):7.2f}")

if __name__ == "__main__":
    main()
