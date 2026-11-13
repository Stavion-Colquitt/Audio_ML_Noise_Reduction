"""DNSMOS P.835 (Microsoft's no-reference perceptual score: SIG speech, BAK background, OVRL overall) on the
VoiceBank+DEMAND noisy test files vs the same files through the live pipeline's model stage.
    py -3.12 eval_dnsmos.py <noisy dir> [n]
"""
import sys, os, glob, tempfile
import numpy as np, soundfile as sf
import eval_vbdemand as E                  # E.enhance runs the v3 model stage (v3_infer; V3_PHASE picks the phase mode)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "DNS-Challenge", "DNSMOS"))
import dnsmos_local as D

def main():
    ndir = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    M = os.path.join(HERE, "DNS-Challenge", "DNSMOS", "DNSMOS")
    cs = D.ComputeScore(os.path.join(M, "sig_bak_ovr.onnx"), os.path.join(M, "model_v8.onnx"))
    files = sorted(glob.glob(os.path.join(ndir, "*.wav")))[:n]; tmp = tempfile.mkdtemp()
    R = {"noisy": [], "enhanced": []}
    for f in files:
        x = E.load(f); y = E.enhance(x)
        for tag, sig in (("noisy", x), ("enhanced", y)):
            p = os.path.join(tmp, f"{tag}.wav"); sf.write(p, sig, E.SR)
            s = cs(p, E.SR, False); R[tag].append((s["SIG"], s["BAK"], s["OVRL"]))
    print(f"DNSMOS P.835, {len(files)} files      SIG    BAK   OVRL")
    for tag, v in R.items():
        a = np.mean(v, 0); print(f"{tag:28s} {a[0]:5.2f}  {a[1]:5.2f}  {a[2]:5.2f}")

if __name__ == "__main__":
    main()
