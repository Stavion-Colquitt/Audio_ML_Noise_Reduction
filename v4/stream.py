"""Real-time streaming engine: 80-sample (5 ms) blocks in, 80-sample blocks out, 10 ms algorithmic latency.

Each call: shift the block into a 512-sample (32 ms) history, apply the long analysis window, FFT, run one GRU step
(state kept between calls), mask, inverse FFT, apply the short synthesis window, and overlap-add. The emitted block is
the PREVIOUS 5 ms of audio, now complete, so input -> output delay is 2 x 80 = 160 samples = 10 ms. This is the
same math as model.analysis/synthesis (check_parity.py confirms it), i.e. what a VST/AU plugin would run.
    eng = StreamEngine(); y_block = eng.process(x_block)   # x_block: float32[80]
"""
import os, numpy as np, onnxruntime as ort
from model import windows, K, M, SR

ONNX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "call_denoiser_v4.onnx")
HOP = M


class StreamEngine:
    def __init__(self, path=ONNX, layers=2, hidden=256, mask_floor=0.0):
        so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
        self.s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        wa, ws = windows(); self.wa = wa.astype(np.float32); self.ws = ws[K - 2 * M:].astype(np.float32)
        self.h = np.zeros((layers, 1, hidden), np.float32); self.floor = mask_floor
        self.inbuf = np.zeros(K, np.float32); self.tail = np.zeros(M, np.float32)

    def reset(self):
        self.h[:] = 0; self.inbuf[:] = 0; self.tail[:] = 0

    def process(self, x):                       # x: float32[80]
        self.inbuf[:-M] = self.inbuf[M:]; self.inbuf[-M:] = x
        S = np.fft.rfft(self.inbuf * self.wa)
        m, self.h = self.s.run(None, {"mag": np.abs(S).astype(np.float32)[None, None], "h": self.h})
        m = np.maximum(m[0, 0], self.floor)
        seg = np.fft.irfft(S * m, K)[K - 2 * M:].astype(np.float32) * self.ws
        out = self.tail + seg[:M]; self.tail = seg[M:].copy()
        return out


def process_file(x, **kw):
    """Whole signal through the streaming engine, re-aligned to the input (the engine itself runs 10 ms late)."""
    e = StreamEngine(**kw); n = len(x)
    xp = np.concatenate([x, np.zeros(2 * M, np.float32)]); xp = xp[: (len(xp) // M) * M]
    y = np.concatenate([e.process(xp[i:i + M]) for i in range(0, len(xp), M)])
    return y[M: M + n]
