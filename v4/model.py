"""Call denoiser v4: causal streaming speech denoiser, 16 kHz, 10 ms algorithmic latency (port of the DAW model).

Asymmetric analysis/synthesis windows (Wang et al. 2021, arXiv:2106.11794): a LONG 32 ms analysis window
(512 samples -> 257 bins, 31 Hz resolution) with a SHORT 10 ms
synthesis window (160 samples). Latency is set by the synthesis window, so we get long-window frequency resolution
at short-window latency. Hop = 5 ms (80). analysis * synthesis = Hann(160) on the last 160 samples, so the
overlap-add reconstructs perfectly when the mask is 1.

Per frame: |S|^0.3 -> Linear -> LayerNorm -> 2-layer GRU (causal context) -> Linear -> sigmoid magnitude mask.
The mask keeps the input phase (the call-denoiser v3 experiment showed predicted phase hurt quality).
"""
import numpy as np, torch, torch.nn as nn

SR, K, M = 16000, 512, 80            # analysis length, hop (synthesis window = 2M = 160 = 10 ms)
BINS = K // 2 + 1
COMP = 0.3
D_ZEROS = 21                          # leading zeros in the analysis window (reduce time aliasing)


def windows():
    """Asymmetric pair (numpy float64): returns (w_analysis[K], w_synthesis[K])."""
    def hann(n): return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)           # periodic Hann
    L = 2 * (K - M - D_ZEROS); hL, h2M = hann(L), hann(2 * M)
    wa = np.zeros(K); ws = np.zeros(K)
    wa[D_ZEROS:K - M] = np.sqrt(hL[: K - M - D_ZEROS])                               # long rising half
    wa[K - M:] = np.sqrt(h2M[M:])                                                    # short falling half
    seg = slice(K - 2 * M, K - M)
    ws[seg] = h2M[:M] / np.maximum(wa[seg], 1e-12)                                   # so wa * ws = Hann(2M)
    ws[K - M:] = np.sqrt(h2M[M:])
    return wa, ws


_WA, _WS = (torch.tensor(w, dtype=torch.float32) for w in windows())


def analysis(x):
    """x [B, N] -> complex [B, T, BINS]; frame k covers x[(k+1)M - K, (k+1)M) (causal, zero history)."""
    xp = torch.nn.functional.pad(x, (K - M, 0))
    T = (xp.shape[-1] - K) // M + 1
    fr = xp.unfold(-1, K, M)[:, :T] * _WA.to(x.device)
    return torch.fft.rfft(fr, dim=-1)


def synthesis(Y, n):
    """complex [B, T, BINS] -> [B, n]. Block j of the output = B_j + A_{j+1}, where (A_k, B_k) are the two halves
    of the last 2M samples of frame k after the synthesis window (exactly what the stream engine emits)."""
    fr = torch.fft.irfft(Y, n=K, dim=-1)[..., K - 2 * M:] * _WS.to(Y.device)[K - 2 * M:]
    A, Bh = fr[..., :M], fr[..., M:]
    out = Bh + torch.cat([A[:, 1:], torch.zeros_like(A[:, :1])], 1)                  # [B, T, M]
    y = out.reshape(out.shape[0], -1)
    return torch.nn.functional.pad(y, (0, max(0, n - y.shape[-1])))[:, :n]


class StreamDenoiser(nn.Module):
    def __init__(self, hidden=256, layers=2):
        super().__init__()
        self.hidden, self.layers = hidden, layers
        self.inp = nn.Sequential(nn.Linear(BINS, hidden), nn.LayerNorm(hidden), nn.PReLU())
        self.gru = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True)
        self.out = nn.Sequential(nn.Linear(hidden, hidden), nn.PReLU(), nn.Linear(hidden, BINS), nn.Sigmoid())
        with torch.no_grad(): self.out[2].bias.fill_(2.0)           # start near "pass everything" (mask ~0.88)

    def forward(self, mag, h=None):                                  # mag [B, T, F] linear magnitude
        z, h = self.gru(self.inp(mag.clamp_min(1e-8) ** COMP), h)
        return self.out(z), h


def enhance(model, noisy):      # [B, N] -> [B, N]   (input at its real level: no normalisation, like the plugin)
    S = analysis(noisy); m, _ = model(S.abs()); return synthesis(m * S, noisy.shape[-1])


def count_parameters(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)


if __name__ == "__main__":
    x = torch.randn(2, SR)
    ident = synthesis(analysis(x), SR)
    err = (ident[:, M:-M] - x[:, M:-M]).abs().max()                  # offline path is time-aligned with the input
    print(f"perfect-reconstruction check (mask = 1): max error {err:.2e}")
    m = StreamDenoiser(); print(f"parameters: {count_parameters(m):,}"); print(enhance(m, x).shape)
