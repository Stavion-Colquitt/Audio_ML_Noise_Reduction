"""Parity check: the live/eval numpy+ONNX path (v2_infer.run_model) must reproduce training's torch path
(train.enhance) on the same 200 ms window. Prints the max difference and the SNR between them."""
import sys, numpy as np, torch
sys.argv = [sys.argv[0]]                       # train.py parses args at import
import train as T, v2_infer as V
from model import UNetDenoiser
from dataset import get_dataloaders

ck = torch.load(T.os.path.join(T.CKPT_DIR, "best_model.pth"), map_location="cpu")
m = UNetDenoiser(base_ch=32); m.load_state_dict(ck["model_state_dict"]); m.eval()
_, dv = get_dataloaders(T.CLEAN_DIR, T.NOISE_DIR, batch_size=4, max_files=200, workers=0)
noisy, _ = next(iter(dv))
with torch.no_grad():
    ref = T.enhance(m, noisy).numpy()
out = np.stack([V.run_model(x.numpy(), floor=0.0) for x in noisy])
d = out - ref
print(f"max |diff| {np.abs(d).max():.2e}   parity SNR {10 * np.log10((ref ** 2).sum() / ((d ** 2).sum() + 1e-20)):.1f} dB")
