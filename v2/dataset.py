"""
DNS Challenge Dataset Loader  (v2)
==================================
v2 changes vs v1:
- Returns WAVEFORMS (noisy, clean), not spectrograms: the loss is now computed on reconstructed audio.
- 200 ms segments: exactly the live pipeline's analysis window, so training sees what inference sees.
- SNR -5..+30 dB (was -5..+20) and 10% clean-only examples, so the model learns to leave clean speech alone.
- Random input level (-35..-15 dBFS RMS) instead of per-window rescaling at inference.
- Fixed train/dev split BY FILE (crc32 of the file name), so dev speakers and noises are never trained on.
- Several random crops per file per epoch (CROPS_PER_FILE), since one 200 ms crop per file is very little audio.
"""

import os, random, glob, zlib
import torch, torchaudio
from torch.utils.data import Dataset, DataLoader

SAMPLE_RATE = 16000
SEG = 3200                    # 200 ms = the live pipeline's window
N_FFT, HOP_LENGTH = 512, 128
SNR_MIN_DB, SNR_MAX_DB = -5, 30
P_CLEAN_ONLY = 0.10
LEVEL_DB = (-35, -15)
CROPS_PER_FILE = 4
DEV_PCT = 5


def _is_dev(path, pct=DEV_PCT):
    return zlib.crc32(os.path.basename(path).encode()) % 100 < pct


_resamplers = {}
def load_segment(path, n=SEG):
    """Random n-sample mono crop at 16 kHz (reads only the needed frames)."""
    info = torchaudio.info(path); sr = info.sample_rate
    need = int(n * sr / SAMPLE_RATE) + 64
    off = random.randint(0, max(0, info.num_frames - need))
    w, sr = torchaudio.load(path, frame_offset=off, num_frames=need)
    if w.shape[0] > 1: w = w.mean(0, keepdim=True)
    if sr != SAMPLE_RATE:
        if sr not in _resamplers: _resamplers[sr] = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
        w = _resamplers[sr](w)
    w = w.squeeze(0)[:n]
    return torch.nn.functional.pad(w, (0, n - len(w)))


def mix_at_snr(clean, noise, snr_db):
    cp, np_ = clean.pow(2).mean(), noise.pow(2).mean()
    if np_ < 1e-10: return clean.clone(), clean
    return clean + noise * torch.sqrt(cp / (10 ** (snr_db / 10)) / np_), clean


class DNSDataset(Dataset):
    def __init__(self, clean_files, noise_files, crops_per_file=CROPS_PER_FILE):
        self.clean_files, self.noise_files, self.k = clean_files, noise_files, crops_per_file
        if not clean_files or not noise_files: raise ValueError("no audio files")

    def __len__(self):
        return len(self.clean_files) * self.k

    def __getitem__(self, idx):
        f = self.clean_files[idx // self.k]
        for _ in range(6):                                   # skip near-silent crops (SI-SDR undefined)
            clean = load_segment(f)
            if clean.pow(2).mean() > 1e-6: break
        if random.random() < P_CLEAN_ONLY:
            noisy = clean.clone()
        else:
            noise = load_segment(random.choice(self.noise_files))
            noisy, clean = mix_at_snr(clean, noise, random.uniform(SNR_MIN_DB, SNR_MAX_DB))
        g = 10 ** (random.uniform(*LEVEL_DB) / 20) / (noisy.pow(2).mean().sqrt() + 1e-8)
        return noisy * g, clean * g


def _files(d):
    return sorted(glob.glob(os.path.join(d, "**", "*.wav"), recursive=True) +
                  glob.glob(os.path.join(d, "**", "*.flac"), recursive=True))


def get_dataloaders(clean_dir, noise_dir, batch_size=64, max_files=None, workers=4):
    C, N = _files(clean_dir), _files(noise_dir)
    ctr, cdv = [f for f in C if not _is_dev(f)], [f for f in C if _is_dev(f)]
    ntr, ndv = [f for f in N if not _is_dev(f)], [f for f in N if _is_dev(f)]
    if max_files:
        random.Random(0).shuffle(ctr); random.Random(0).shuffle(cdv)
        ctr, cdv = ctr[:max_files], cdv[:max(50, max_files // 10)]
    print(f"train: {len(ctr)} clean / {len(ntr)} noise files | dev: {len(cdv)} clean / {len(ndv)} noise files")
    tr = DataLoader(DNSDataset(ctr, ntr), batch_size=batch_size, shuffle=True, num_workers=workers,
                    pin_memory=True, persistent_workers=workers > 0)
    # dev: single process so train.validate() can reseed and every epoch scores the SAME crops/mixes
    dv = DataLoader(DNSDataset(cdv, ndv, crops_per_file=2), batch_size=batch_size, shuffle=False,
                    num_workers=0)
    return tr, dv
