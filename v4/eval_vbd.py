"""Score v4 on the full VoiceBank-DEMAND test set (824 files, 16 kHz) against the noisy input and earlier versions.

Metrics: PESQ (wideband), STOI, SI-SDR. Same benchmark as v2/v3 (noisy 1.97 / v2 1.65 / v3 1.70 PESQ).
    py -3.12 eval_vbd.py            # uses checkpoints\best.pth (EMA weights)
    py -3.12 eval_vbd.py 100        # quick: first 100 files
"""
import os, sys, glob, json, numpy as np, soundfile as sf, torch
from pesq import pesq
from pystoi import stoi
from model import StreamDenoiser, enhance, SR

HERE = os.path.dirname(os.path.abspath(__file__))
VBD = os.environ.get("VBD16", r"D:\Datasets\voicebank-demand\16k")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def si_sdr(e, r):
    e, r = e - e.mean(), r - r.mean(); t = np.dot(e, r) / (np.dot(r, r) + 1e-12) * r
    return 10 * np.log10(np.sum(t ** 2) / (np.sum((e - t) ** 2) + 1e-12) + 1e-12)


def scores(est, ref):
    try: p = pesq(SR, ref, est, "wb")
    except Exception: p = float("nan")
    return p, stoi(ref, est, SR, extended=False), si_sdr(est, ref)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else None
    files = sorted(glob.glob(os.path.join(VBD, "clean_test", "*.wav")))[:n]
    ck = torch.load(os.path.join(HERE, "checkpoints", "best.pth"), map_location=DEV, weights_only=False)
    model = StreamDenoiser().to(DEV).eval(); model.load_state_dict(ck["model"])
    print(f"checkpoint step {ck.get('step')}  dev loss {ck.get('dev_loss')}  | {len(files)} test files", flush=True)
    rows = {"noisy": [], "v4": []}
    with torch.no_grad():
        for i, f in enumerate(files):
            ref, _ = sf.read(f, dtype="float32"); noisy, _ = sf.read(f.replace("clean_test", "noisy_test"), dtype="float32")
            L = min(len(ref), len(noisy)); ref, noisy = ref[:L], noisy[:L]
            est = enhance(model, torch.from_numpy(noisy)[None].to(DEV))[0].cpu().numpy()
            rows["noisy"].append(scores(noisy, ref)); rows["v4"].append(scores(est, ref))
            if (i + 1) % 100 == 0: print(f"  {i + 1}/{len(files)}", flush=True)
    out = {}
    for k, v in rows.items():
        a = np.nanmean(np.array(v), 0); out[k] = {"pesq_wb": round(float(a[0]), 3), "stoi": round(float(a[1]), 3), "si_sdr": round(float(a[2]), 2)}
    print(f"{'':8}{'PESQ':>8}{'STOI':>8}{'SI-SDR':>9}")
    for k, m in out.items(): print(f"{k:8}{m['pesq_wb']:8.3f}{m['stoi']:8.3f}{m['si_sdr']:9.2f}")
    print("for reference (earlier eval, 16 kHz): v2 PESQ 1.65 / STOI 0.869 / SI-SDR 13.36 ; v3 1.70 / 0.874 / 13.17")
    json.dump(out, open(os.path.join(HERE, "eval_vbd.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
