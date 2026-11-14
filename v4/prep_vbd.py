r"""Unzip the VoiceBank-DEMAND train set and resample train + test from 48 kHz to 16 kHz (the usual benchmark rate).

Source: Valentini-Botinhao et al., Edinburgh DataShare 10283/2791 (CC BY 4.0). Output: <root>\16k\{clean,noisy}_{train,test}
    py -3.12 prep_vbd.py
"""
import os, sys, zipfile, glob, numpy as np, soundfile as sf
from scipy.signal import resample_poly
from concurrent.futures import ProcessPoolExecutor

ROOT = os.environ.get("VBD_ROOT", r"D:\Datasets\voicebank-demand")
OUT = os.path.join(ROOT, "16k")
JOBS = [("clean_trainset_28spk_wav", "clean_train"), ("noisy_trainset_28spk_wav", "noisy_train"),
        ("clean_testset_wav", "clean_test"), ("noisy_testset_wav", "noisy_test")]


def one(args):
    src, dst = args
    if os.path.exists(dst): return 0
    x, sr = sf.read(src, dtype="float32")
    if x.ndim > 1: x = x.mean(1)
    g = np.gcd(16000, sr); y = resample_poly(x, 16000 // g, sr // g).astype(np.float32)
    sf.write(dst, y, 16000, subtype="PCM_16"); return 1


def main():
    tasks = []
    for name, out in JOBS:
        d = os.path.join(ROOT, name)
        if not os.path.isdir(d):
            z = os.path.join(ROOT, name + ".zip")
            if not os.path.exists(z): sys.exit(f"missing {z}")
            print("unzipping", z, flush=True); zipfile.ZipFile(z).extractall(ROOT)
        os.makedirs(os.path.join(OUT, out), exist_ok=True)
        files = sorted(glob.glob(os.path.join(d, "*.wav")))
        print(name, len(files), "files", flush=True)
        tasks += [(f, os.path.join(OUT, out, os.path.basename(f))) for f in files]
    with ProcessPoolExecutor(max_workers=6) as ex:
        done = sum(ex.map(one, tasks, chunksize=64))
    print(f"resampled {done} new files -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
