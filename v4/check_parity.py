"""Parity + speed: the streaming ONNX engine (block by block, causal) must reproduce the PyTorch training path
(whole-signal STFT/GRU/iSTFT). Also reports how long one 5 ms block takes on this CPU."""
import time, numpy as np, torch
from model import StreamDenoiser, enhance, SR, M as HOP
from stream import process_file, StreamEngine

ck = torch.load("checkpoints/best.pth", map_location="cpu", weights_only=False)
m = StreamDenoiser(); m.load_state_dict(ck["model"]); m.eval()
x = (np.random.default_rng(0).standard_normal(3 * SR) * 0.05).astype(np.float32)
with torch.no_grad(): ref = enhance(m, torch.from_numpy(x)[None])[0].numpy()
y = process_file(x)
a, b = 2 * HOP, len(x) - 2 * HOP           # skip the very first/last frame (training pads by reflection there)
d = y[a:b] - ref[a:b]
print(f"max |diff| {np.abs(d).max():.2e}   parity SNR {10 * np.log10((ref[a:b] ** 2).sum() / ((d ** 2).sum() + 1e-20)):.1f} dB")
e = StreamEngine(); blk = np.zeros(HOP, np.float32); [e.process(blk) for _ in range(50)]
t = time.perf_counter(); N = 2000
for _ in range(N): e.process(blk)
ms = (time.perf_counter() - t) / N * 1000
print(f"per 5 ms block: {ms:.3f} ms on CPU  (real-time factor {ms / 5:.3f}); algorithmic latency 10 ms")
