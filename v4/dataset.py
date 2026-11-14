"""Training data for call denoiser v4 (16 kHz).

Why v2/v3 lost to the noisy input on PESQ: they only ever saw DNS synthetic mixtures, then were tested on
VoiceBank-DEMAND (different speakers, mics, noises and levels). v4 trains on both:
  60%  VoiceBank-DEMAND TRAIN pairs (28 speakers; never the 824-file test set). Speakers p226 and p287 are held out
       as the dev set (the usual split), so dev numbers track the real test conditions.
  40%  DNS Challenge clean speech + DNS noise mixed on the fly, SNR -5..20 dB, for variety (5% clean-only, so the
       model learns to leave clean speech alone).
Random gain on both (same factor for input and target) so the model handles any call level.
Data is pre-resampled to 16 kHz by prep_vbd.py (VBD) or resampled on the fly (DNS, 48 kHz -> 16 kHz).
"""
import os, glob, random, zlib, numpy as np, soundfile as sf, torch
from scipy.signal import resample_poly
from torch.utils.data import Dataset, DataLoader

SR = 16000
SEG = 4 * SR
VBD = os.environ.get("VBD16", r"D:\Datasets\voicebank-demand\16k")
DNS = os.environ.get("DNS_DATA", r"C:\Users\stavi\Projects\audio-ml-pipeline\data")
DEV_SPK = ("p226", "p287")
P_VBD, P_CLEAN_ONLY = 0.60, 0.05


def _is_dev(path, pct=5): return zlib.crc32(os.path.basename(path).encode()) % 100 < pct
def rms(x): return float(np.sqrt(np.mean(x ** 2) + 1e-12))


def vbd_pairs(split):
    files = sorted(glob.glob(os.path.join(VBD, "clean_train", "*.wav")))
    keep = [f for f in files if (os.path.basename(f).split("_")[0] in DEV_SPK) == (split != "train")]
    return [(os.path.join(VBD, "noisy_train", os.path.basename(f)), f) for f in keep]


def dns_pools(split):
    g = lambda pat: sorted(glob.glob(pat, recursive=True))
    sp = [f for f in g(os.path.join(DNS, "clean_fullband", "**", "*.wav")) if _is_dev(f) == (split != "train")]
    nz = [f for f in g(os.path.join(DNS, "noise_fullband", "**", "*.wav")) if _is_dev(f) == (split != "train")]
    return sp, nz


def read_crop(path, n, rng, min_rms=1e-3, tries=6):
    """Random n-sample mono crop at 16 kHz (loops short files); skips near-silent crops."""
    info = sf.info(path); sr = info.samplerate; need = int(np.ceil(n * sr / SR))
    for _ in range(tries):
        if info.frames > need:
            s = rng.randint(0, info.frames - need); x, _ = sf.read(path, start=s, frames=need, dtype="float32", always_2d=True)
        else:
            x, _ = sf.read(path, dtype="float32", always_2d=True)
        x = x.mean(1)
        if sr != SR:
            g = np.gcd(SR, sr); x = resample_poly(x, SR // g, sr // g).astype(np.float32)
        if len(x) < n: x = np.tile(x, int(np.ceil(n / max(len(x), 1))))
        x = x[:n]
        if rms(x) > min_rms: return x
    return x


def read_pair(noisy_path, clean_path, n, rng):
    """Same random n-sample window from an aligned VBD pair; short utterances are zero-padded (never looped)."""
    y, _ = sf.read(noisy_path, dtype="float32"); x, _ = sf.read(clean_path, dtype="float32")
    L = min(len(x), len(y)); x, y = x[:L], y[:L]
    if L > n:
        s = rng.randint(0, L - n); return y[s:s + n], x[s:s + n]
    pad = n - L; a = rng.randint(0, pad)
    return np.pad(y, (a, pad - a)), np.pad(x, (a, pad - a))


class Mixtures(Dataset):
    def __init__(self, split, length, seed=None):
        self.pairs = vbd_pairs(split); self.speech, self.noise = dns_pools(split)
        self.length, self.seed = length, seed

    def __len__(self): return self.length

    def __getitem__(self, i):
        rng = random.Random(self.seed * 100003 + i) if self.seed is not None else random.Random()
        if self.pairs and (rng.random() < P_VBD or not self.speech):
            noisy, clean = read_pair(*rng.choice(self.pairs), SEG, rng)
            g = 10 ** (rng.uniform(-10, 6) / 20)                          # same gain on both
            noisy, clean = noisy * g, clean * g
        else:
            clean = read_crop(rng.choice(self.speech), SEG, rng)
            clean = clean * 10 ** (rng.uniform(-35, -15) / 20) / rms(clean)
            if rng.random() < P_CLEAN_ONLY:
                noisy = clean.copy()
            else:
                nz = read_crop(rng.choice(self.noise), SEG, rng, 1e-4)
                nz = nz * rms(clean) / rms(nz) / 10 ** (rng.uniform(-5, 20) / 20)
                noisy = clean + nz
        pk = np.abs(noisy).max()
        if pk > 0.99: noisy, clean = noisy * (0.99 / pk), clean * (0.99 / pk)
        return torch.from_numpy(np.asarray(noisy, np.float32)), torch.from_numpy(np.asarray(clean, np.float32))


def loaders(batch=32, steps_per_epoch=1000, dev_items=256, workers=6):
    tr = DataLoader(Mixtures("train", batch * steps_per_epoch), batch_size=batch, shuffle=False, num_workers=workers,
                    pin_memory=True, persistent_workers=workers > 0, drop_last=True)
    dv = DataLoader(Mixtures("dev", dev_items, seed=1234), batch_size=batch, shuffle=False, num_workers=0)
    return tr, dv


if __name__ == "__main__":
    for s in ("train", "dev"):
        sp, nz = dns_pools(s); print(s, "vbd pairs", len(vbd_pairs(s)), "dns speech", len(sp), "dns noise", len(nz))
    x, y = Mixtures("train", 4)[0]; print(x.shape, y.shape, float(x.abs().max()))
